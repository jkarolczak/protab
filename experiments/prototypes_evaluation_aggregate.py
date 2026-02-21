import glob
import json
import os
from pathlib import Path

import click
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D

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

    json_files = glob.glob(os.path.join(dir_path, "**", "*.table.json"), recursive=True)

    if not json_files:
        json_files = glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)

    with open(json_files[0], "r") as f:
        table_dict = json.load(f)

    return pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])


@click.command()
@click.option("--entity", type=str, default="jacek-karolczak")
@click.option("--project", type=str, default="ProTab")
@click.option("--log-wandb", is_flag=True)
def main(entity: str, project: str, log_wandb: bool) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 800,
        "savefig.dpi": 800,
    })

    api = wandb.Api(timeout=60)
    datasets_found = []
    purity_records = []
    rank_dfs = {}
    tsne_dfs = {}

    for dataset_name in TNamedBoolData.__args__:
        filters = {
            "tags": {"$in": ["prototypes_evaluation"]},
            "config.data.name": dataset_name
        }

        runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at", per_page=1)
        run = next(iter(runs), None)

        if not run:
            continue

        df_ranks = fetch_logged_table(run, "best_rank_distribution")
        df_purity = fetch_logged_table(run, "top_k_purity")
        df_tsne = fetch_logged_table(run, "tsne_embeddings")

        datasets_found.append(dataset_name)
        rank_dfs[dataset_name] = df_ranks
        tsne_dfs[dataset_name] = df_tsne

        avg_purity = df_purity[[c for c in df_purity.columns if "purity_" in c]].mean().to_dict()
        row = {"Dataset": dataset_name}
        for k_label, val in avg_purity.items():
            k_val = k_label.split("_")[1]
            row[f"$k={k_val}$"] = val
        purity_records.append(row)

    if not datasets_found:
        return

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    df_purity_agg = pd.DataFrame(purity_records)
    for col in df_purity_agg.columns:
        if col != "Dataset":
            df_purity_agg[col] = df_purity_agg[col].apply(lambda x: f"{x:.4f}")

    df_latex = df_purity_agg.copy()
    df_latex["Dataset"] = df_latex["Dataset"].apply(lambda x: str(x).replace("bng_", "").replace("_", "\\_"))

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        label="tab:top_k_purity",
        column_format="l" + "c" * (len(df_latex.columns) - 1)
    )

    with open(output_dir / "top_k_purity.tex", "w") as f:
        f.write(latex_table)

    fig_hist, axes_hist = plt.subplots(1, len(datasets_found), figsize=(4.8, 1.5), dpi=800)
    if len(datasets_found) == 1:
        axes_hist = [axes_hist]

    for ax, ds in zip(axes_hist, datasets_found):
        df = rank_dfs[ds]
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.bar(df["rank"], df["prototypes_count"], color="#666666", edgecolor="black")
        ax.set_title(ds.replace("bng", "").replace("_", " ").title(), pad=3, fontsize=8)
        if ax == axes_hist[0]:
            ax.set_ylabel("Number Of Prototypes", labelpad=2)
        ax.tick_params(axis="both", which="major", pad=2)

    fig_hist.supxlabel("Best Rank Achieved", fontsize=8)

    plt.tight_layout(pad=0.1, w_pad=0.025)
    hist_path = output_dir / "rank_histograms.pdf"
    fig_hist.savefig(hist_path, bbox_inches="tight")

    fig_tsne, axes_tsne = plt.subplots(1, len(datasets_found), figsize=(4.8, 1.5), dpi=800)
    if len(datasets_found) == 1:
        axes_tsne = [axes_tsne]

    cmap = cmc.batlowS

    for ax, ds in zip(axes_tsne, datasets_found):
        df = tsne_dfs[ds]
        classes = sorted(df["class"].unique())
        colors = cmap(np.linspace(0, 1, len(classes)))

        patches = df[df["type"] == "patch"]
        for idx, cls in enumerate(classes):
            subset = patches[patches["class"] == cls]
            ax.scatter(subset["x"], subset["y"], s=2, alpha=0.5, label=f"Class {int(cls)}", color=colors[idx])

        protos = df[df["type"] == "prototype"]
        for idx, cls in enumerate(classes):
            subset = protos[protos["class"] == cls]
            ax.scatter(subset["x"], subset["y"], s=30, marker="*", edgecolor="black", linewidth=0.5, color=colors[idx])

        ax.set_title(ds.replace("_", " ").title(), pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax == axes_tsne[0]:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))

            shape_handles = [
                Line2D([0], [0], marker="o", color="w", label="Patch", markerfacecolor="gray", markersize=3, alpha=0.5),
                Line2D([0], [0], marker="*", color="w", label="Prototype", markerfacecolor="gray", markeredgecolor="black",
                       markersize=6)
            ]

            all_handles = shape_handles + list(by_label.values())
            all_labels = [h.get_label() for h in shape_handles] + list(by_label.keys())

            ax.legend(all_handles, all_labels, loc="best", prop={"size": 4})

    plt.tight_layout(pad=0.2)
    tsne_path = output_dir / "tsne_embeddings.pdf"
    fig_tsne.savefig(tsne_path, bbox_inches="tight")

    if log_wandb:
        wandb.init(
            project=project,
            entity=entity,
            name="prototype_aggregation_summary",
            tags=["prototypes_evaluation", "summary"]
        )

        wandb.log({
            "top_k_purity_table": wandb.Table(dataframe=df_purity_agg),
            "rank_histograms_plot": wandb.Image(fig_hist),
            "tsne_plot": wandb.Image(fig_tsne)
        })

        wandb.save(str(output_dir / "top_k_purity.tex"), policy="now")
        wandb.save(str(hist_path), policy="now")
        wandb.save(str(tsne_path), policy="now")
        wandb.finish()

    plt.close(fig_hist)
    plt.close(fig_tsne)


if __name__ == "__main__":
    main()
