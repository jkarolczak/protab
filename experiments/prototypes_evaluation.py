import glob
import json
import os
import platform

import click
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from protab.data.named_data import TNamedBoolData
from protab.training.config import fetch_best_run


def fetch_logged_table(run, table_name: str):
    artifacts = run.logged_artifacts()
    target_artifact = None
    for artifact in artifacts:
        if table_name in artifact.name and artifact.type == "run_table":
            target_artifact = artifact
            break

    if target_artifact is None:
        raise ValueError(f"Table '{table_name}' not found in run artifacts")

    dir_path = target_artifact.download()
    json_files = glob.glob(os.path.join(dir_path, "*.table.json"))

    with open(json_files[0], "r") as f:
        table_dict = json.load(f)

    df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
    return df


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedBoolData.__args__))
@click.option("--device", type=str, default="cpu")
@click.option("--batch-size", type=int, default=256)
def main(dataset_name: TNamedBoolData, device: str, batch_size: int) -> None:
    best_run, data_container, model_config, trainer_config, model = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=True
    )

    platform_name = platform.node()

    wandb.init(
        project="ProTab",
        entity="jacek-karolczak",
        tags=["prototypes_evaluation"],
        name=f"{dataset_name}_prototypes_analysis",
        config={
            "architecture": "ProTab",
            "model": model_config.__dict__,
            "trainer": trainer_config.__dict__,
            "data": data_container.config.__dict__,
            "platform": platform_name
        },
    )

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    proto_df = fetch_logged_table(best_run, "prototypical_parts")
    proto_vals, proto_masks = data_container.scale(proto_df)
    proto_vals = proto_vals.to(device)
    proto_masks = proto_masks.to(device)

    x_eval_tensor = torch.tensor(data_container.x_eval.values, dtype=torch.float32)

    y_eval_indices_np = np.argmax(data_container.y_eval.values, axis=1)
    y_eval_indices = torch.tensor(y_eval_indices_np, dtype=torch.long)

    class_names = data_container.y_eval.columns.tolist()
    unique_class_indices = sorted(list(set(y_eval_indices_np)))

    dataset = TensorDataset(x_eval_tensor, y_eval_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_min_dists = []
    all_fs_dists = []
    all_labels = []

    p_exp = proto_vals.unsqueeze(0)
    m_exp = proto_masks.unsqueeze(0)

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Computing Distances"):
            x_batch = x_batch.to(device)

            instance_embeddings = model.embeddings(x_batch)
            dists = model.prototypes.compute_distance_matrix(instance_embeddings)

            min_dists, _ = torch.min(dists, dim=1)
            all_min_dists.append(min_dists.cpu())

            x_exp_batch = x_batch.unsqueeze(1)
            diff_sq = (x_exp_batch - p_exp) ** 2
            masked_diff_sq = diff_sq * m_exp

            fs_dists = torch.sqrt(torch.sum(masked_diff_sq, dim=-1) + 1e-8)
            all_fs_dists.append(fs_dists.cpu())

            all_labels.append(y_batch)

    min_dists = torch.cat(all_min_dists)
    fs_dists = torch.cat(all_fs_dists)
    y_eval = torch.cat(all_labels).numpy()

    n_protos = min_dists.shape[1]
    proto_indices = list(range(n_protos))

    emb_cols = ["Prototype_Idx"]
    emb_rows = {pid: [pid] for pid in proto_indices}

    for c_idx in unique_class_indices:
        c_name = str(class_names[c_idx])
        emb_cols.extend([f"{c_name}_Mean", f"{c_name}_Std"])

        mask = (y_eval == c_idx)
        if not np.any(mask):
            for pid in proto_indices:
                emb_rows[pid].extend([np.nan, np.nan])
            continue

        class_dists = min_dists[mask].numpy()

        means = np.mean(class_dists, axis=0)
        stds_ = np.std(class_dists, axis=0)

        for pid in proto_indices:
            emb_rows[pid].extend([means[pid], stds_[pid]])

    wandb.log({
        f"{dataset_name}/embedding_space_stats": wandb.Table(
            data=list(emb_rows.values()),
            columns=emb_cols
        )
    })

    fs_cols = ["Prototype_Idx"]
    fs_rows = {pid: [pid] for pid in proto_indices}

    for c_idx in unique_class_indices:
        c_name = str(class_names[c_idx])
        fs_cols.extend([f"{c_name}_Mean", f"{c_name}_Std"])

        mask = (y_eval == c_idx)
        if not np.any(mask):
            for pid in proto_indices:
                fs_rows[pid].extend([np.nan, np.nan])
            continue

        class_fs_dists = fs_dists[mask].numpy()
        means = np.mean(class_fs_dists, axis=0)
        stds_ = np.std(class_fs_dists, axis=0)

        for pid in proto_indices:
            fs_rows[pid].extend([means[pid], stds_[pid]])

    wandb.log({
        f"{dataset_name}/feature_space_stats": wandb.Table(
            data=list(fs_rows.values()),
            columns=fs_cols
        )
    })

    ranks = torch.argsort(min_dists, dim=1)
    ranks_val = torch.argsort(ranks, dim=1).float()

    mean_ranks = ranks_val.mean(dim=0).numpy()
    std_ranks = ranks_val.std(dim=0).numpy()

    rank_data = [
        [pid, m, s] for pid, (m, s) in enumerate(zip(mean_ranks, std_ranks))
    ]
    wandb.log({
        f"{dataset_name}/proto_ranks": wandb.Table(
            data=rank_data,
            columns=["Prototype_Idx", "Mean_Rank", "Std_Rank"]
        )
    })

    raw_importance = proto_masks.sum(dim=0).cpu().numpy()
    if np.max(raw_importance) > 0:
        raw_importance = raw_importance / np.max(raw_importance)

    cls_weights = model.classifier.network[-1].weight.data
    proto_influence = cls_weights.abs().mean(dim=0)
    n_features_per_proto = torch.clamp(proto_masks.sum(dim=1), min=1.0)

    proto_influence = proto_influence / n_features_per_proto

    if torch.max(proto_influence) > 0:
        proto_influence = proto_influence / torch.max(proto_influence)

    weighted_importance = torch.matmul(proto_influence, proto_masks).cpu().numpy()

    feature_names = data_container.x_train.columns.tolist()
    fi_data = []
    for i, f_name in enumerate(feature_names):
        fi_data.append([f_name, raw_importance[i], weighted_importance[i]])

    wandb.log({
        f"{dataset_name}/feature_importance": wandb.Table(
            data=fi_data, columns=["Feature", "Raw_FI", "Weighted_FI"]
        )
    })


if __name__ == "__main__":
    main()
