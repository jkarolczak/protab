import glob
import json
import os

import click
import pandas as pd
import torch

from p2tab.data.named_data import TNamedData
from p2tab.training.config import fetch_best_run


def fetch_logged_table(run, table_name: str) -> pd.DataFrame:
    artifacts = run.logged_artifacts()
    target_artifact = None
    for artifact in artifacts:
        if table_name in artifact.name and artifact.type == "run_table":
            target_artifact = artifact
            break

    if target_artifact is None:
        raise ValueError(f"Table '{table_name}' not found in run artifacts")

    dir_path = target_artifact.download()

    json_files = glob.glob(os.path.join(dir_path, "**", "*.table.json"), recursive=True)

    if not json_files:
        json_files = glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)

    with open(json_files[0], "r") as f:
        table_dict = json.load(f)

    return pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--sample-idx", type=int, default=0, help="Index of the test sample to visualize.")
@click.option("--markdown", is_flag=True, help="Print the table in Markdown format for easy copying to a manuscript.")
def main(dataset_name: str, sample_idx: int, markdown: bool) -> None:
    print(f"Fetching best model and configs for '{dataset_name}'...")

    best_run, data_container, model_config, trainer_config, model = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=True
    )
    model.eval()

    df_prototypes = fetch_logged_table(best_run, "prototypical_parts")
    df_cls_matrix = fetch_logged_table(best_run, "classification_matrix")

    proto_importance = df_cls_matrix.abs().sum(axis=0).values
    feature_active_matrix = df_prototypes.notna().astype(float)
    feature_importance_raw = feature_active_matrix.T.dot(proto_importance)
    if feature_importance_raw.max() > 0:
        feature_importance_norm = (feature_importance_raw / feature_importance_raw.max())
    else:
        feature_importance_norm = feature_importance_raw

    feature_importance_str = feature_importance_norm.apply(lambda x: f"{x:.4f}")

    _, _, test_dataset = data_container.to_simple_datasets()
    x, y = test_dataset[sample_idx]
    x_batch = x.unsqueeze(0)

    with torch.no_grad():
        patches = model.patching(x_batch)

        logits, patches_embeddings = model(x_batch, return_embeddings=True)
        prototype_dist, patches_idcs = model.prototypes(patches_embeddings)

        distances = prototype_dist.squeeze(0)
        idcs = patches_idcs.squeeze(0)

    pred_class = torch.argmax(logits, dim=-1).item()
    true_class = torch.argmax(y, dim=-1).item()

    descaled_sample = data_container.descale(x_batch).iloc[0]

    D = data_container.n_features
    append_masks = model_config.patching.append_masks

    def get_descaled_patch(p_idx):
        raw_patch = patches[0, p_idx].clone()
        if append_masks:
            feat_vals = raw_patch[:D]
            mask = raw_patch[D:].bool()
            feat_vals[~mask] = torch.nan
        else:
            feat_vals = raw_patch

        return data_container.descale(feat_vals.unsqueeze(0)).iloc[0]

    sorted_indices = torch.argsort(distances)
    top_3_idx = sorted_indices[:3].tolist()
    bottom_3_idx = sorted_indices[-3:].tolist()

    data_dict = {
        "Feature Importance": feature_importance_str,
        "Original Sample": descaled_sample
    }

    distances_dict = {
        "Feature Importance": "-",
        "Original Sample": "-"
    }

    for i, idx in enumerate(top_3_idx):
        proto_col = f"Top {i + 1} (Proto ID: {idx})"
        patch_col = f"Top {i + 1} (Sample Patch)"

        data_dict[proto_col] = df_prototypes.iloc[idx]
        distances_dict[proto_col] = f"{distances[idx]:.4f}"

        sample_patch_idx = idcs[idx].item()
        data_dict[patch_col] = get_descaled_patch(sample_patch_idx)
        distances_dict[patch_col] = "-"

    for i, idx in enumerate(bottom_3_idx):
        rank = len(distances) - 3 + i
        proto_col = f"Bottom {i + 1} (Proto Rank {rank - i})"
        patch_col = f"Bottom {i + 1} (Sample Patch)"

        data_dict[proto_col] = df_prototypes.iloc[idx]
        distances_dict[proto_col] = f"{distances[idx]:.4f}"

        sample_patch_idx = idcs[idx].item()
        data_dict[patch_col] = get_descaled_patch(sample_patch_idx)
        distances_dict[patch_col] = "-"

    comp_df = pd.DataFrame(data_dict)

    pair_cols = [col for col in comp_df.columns if col.startswith("Top") or col.startswith("Bottom")]
    comp_df = comp_df.dropna(how="all", subset=pair_cols)

    dist_df = pd.DataFrame([distances_dict], index=[">> Distance to Sample <<"])
    final_table = pd.concat([dist_df, comp_df])

    print("\n" + "=" * 80)
    print(f"TOY EXAMPLE SUMMARY: {dataset_name.upper()}")
    print("=" * 80)
    print(f"Sample Index    : {sample_idx}")
    print(f"True Class      : {true_class}")
    print(f"Predicted Class : {pred_class}")
    print("=" * 80 + "\n")

    if markdown:
        print(final_table.fillna("-").to_markdown())
    else:
        print(final_table.fillna("-").to_string())


if __name__ == "__main__":
    main()
