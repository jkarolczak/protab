import glob
import json
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import wandb

from protab.data.named_data import TNamedBoolData


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

    # 1. Try recursive search for .table.json
    json_files = glob.glob(os.path.join(dir_path, "**", "*.table.json"), recursive=True)

    # 2. Fallback: Search for any .json file if the specific extension isn't found
    if not json_files:
        json_files = glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)

    if not json_files:
        raise FileNotFoundError(f"Could not find any JSON files for table '{table_name}' in downloaded artifact at {dir_path}")

    with open(json_files[0], "r") as f:
        table_dict = json.load(f)

    return pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])


def compute_separation_margin(df: pd.DataFrame) -> float:
    class_cols = [c for c in df.columns if str(c).endswith('_Mean')]
    if len(class_cols) < 2:
        return 0.0

    means = df[class_cols].values
    sorted_means = np.sort(means, axis=1)

    margins = 1.0 - (sorted_means[:, 0] / (sorted_means[:, 1] + 1e-8))

    return np.mean(margins)


def compute_gini(array: np.ndarray) -> float:
    array = np.clip(np.sort(array), 0, None)  # Ensure no negatives
    if np.sum(array) == 0:
        return 0.0
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


@click.command()
@click.option("--entity", type=str, default="jacek-karolczak", help="WandB entity name.")
@click.option("--project", type=str, default="ProTab", help="WandB project name.")
@click.option("--log-wandb", is_flag=True, help="Enable logging results to WandB.")
def main(entity: str, project: str, log_wandb: bool) -> None:
    api = wandb.Api(timeout=60)
    all_records = []

    print(f"Fetching prototype evaluations from {entity}/{project}...")

    for dataset_name in TNamedBoolData.__args__:
        # Search for the evaluation run explicitly tagged with 'prototypes_evaluation'
        filters = {
            "tags": {"$in": ["prototypes_evaluation"]},
            "config.data.name": dataset_name
        }

        runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at", per_page=1)
        run = next(iter(runs), None)

        if not run:
            continue

        df_emb = fetch_logged_table(run, "embedding_space_stats")
        df_feat = fetch_logged_table(run, "feature_space_stats")
        df_rank = fetch_logged_table(run, "proto_ranks")
        df_fi = fetch_logged_table(run, "feature_importance")

        # 2. Compute Aggregated Metrics
        latent_margin = compute_separation_margin(df_emb)
        feature_margin = compute_separation_margin(df_feat)
        rank_volatility = np.mean(df_rank["Std_Rank"] / len(df_rank))
        feature_focus = compute_gini(df_fi["Weighted_FI"].values)

        all_records.append({
            "Dataset": dataset_name,
            "Latent Margin": latent_margin,
            "Feature Margin": feature_margin,
            "Rank Volatility (Std Dev)": rank_volatility,
            "Feature Focus (Gini)": feature_focus
        })

    df = pd.DataFrame(all_records)

    df_formatted = df.copy()
    df_formatted["Latent Margin"] = df_formatted["Latent Margin"].apply(lambda x: f"{x:.4f}")
    df_formatted["Feature Margin"] = df_formatted["Feature Margin"].apply(lambda x: f"{x:.4f}")
    df_formatted["Rank Volatility (Std Dev)"] = df_formatted["Rank Volatility (Std Dev)"].apply(lambda x: f"{x:.4f}")
    df_formatted["Feature Focus (Gini)"] = df_formatted["Feature Focus (Gini)"].apply(lambda x: f"{x:.4f}")

    df_latex = df_formatted.copy()
    df_latex["Dataset"] = df_latex["Dataset"].apply(lambda x: str(x).replace("_", "\\_"))

    caption_text = (
        "Aggregated prototype learning metrics demonstrating class specialization and interpretability. "
        "Latent and Feature Margins denote how much closer prototypes are to their primary class than the secondary class. "
        "Rank Volatility indicates local specialization (dynamic prototype usage across instances), and "
        "Feature Focus (Gini coefficient) demonstrates reliance on a sparse subset of features."
    )

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        caption=caption_text,
        label="tab:prototype_analysis",
        column_format="lcccc"
    )

    output_path = Path("results") / "prototype_analysis_table.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_table)

    if log_wandb:
        wandb.init(
            project=project,
            entity=entity,
            name="prototype_aggregation_summary",
            tags=["prototypes_evaluation", "summary"]
        )

        wandb.log({
            "prototype_analysis_table": wandb.Table(dataframe=df)
        })

        wandb.save(output_path, policy="now")
        wandb.finish()


if __name__ == "__main__":
    main()
