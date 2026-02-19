from pathlib import Path

import click
import pandas as pd
import wandb

from protab.data.named_data import TNamedData

# The ablation tags exactly as they appear in your ablation scripts
ABLATIONS = [
    "no_append_mask",
    "no_compound_loss",
    "no_prototypes",
    "no_trainable_token"
]

# Pretty names for the table columns
ABLATION_NAMES = {
    "no_append_mask": "No Append Mask",
    "no_compound_loss": "No Compound Loss",
    "no_prototypes": "No Prototypes",
    "no_trainable_token": "No Trainable Token"
}


@click.command()
@click.option("--metric", type=str, default="eval_balanced_accuracy_diff",
              help="Metric difference to aggregate (e.g., eval_balanced_accuracy_diff, eval_f1_score_diff).")
@click.option("--log-wandb", is_flag=True, help="Enable logging results to WandB.")
def main(metric: str, log_wandb: bool) -> None:
    api = wandb.Api()
    all_records = []
    for dataset_name in TNamedData.__args__:
        row = {"Dataset": dataset_name}

        has_data = False
        for ablation_tag in ABLATIONS:
            filters = {
                "config.data.name": dataset_name,
                "tags": {"$in": [ablation_tag]}
            }

            runs = api.runs(
                f"jacek-karolczak/ProTab",
                filters=filters,
                order="-created_at",
                per_page=1
            )

            run = next(iter(runs), None)

            if run and metric in run.summary:
                row[ABLATION_NAMES[ablation_tag]] = run.summary[metric]
                has_data = True
            else:
                row[ABLATION_NAMES[ablation_tag]] = None

        if has_data:
            all_records.append(row)

    if not all_records:
        print("No ablation runs found. Make sure your ablation experiments have finished running and logging.")
        return

    df = pd.DataFrame(all_records)

    float_cols = [ABLATION_NAMES[tag] for tag in ABLATIONS]
    for col in float_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    df_latex = df.copy()
    df_latex["Dataset"] = df_latex["Dataset"].apply(lambda x: x.replace("_", "\\_"))

    caption_text = (
        f"Ablation study results across datasets. Values represent the difference in "
        f"{metric.replace('_', ' ')} compared to the full ProTab model (Negative values indicate a performance drop)."
    )

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        caption=caption_text,
        label="tab:ablation_results",
        column_format="l" + "c" * len(ABLATIONS)
    )

    output_path = Path("results") / "ablation_results_table.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_table)

    if log_wandb:
        wandb.init(
            entity="jacek-karolczak",
            project="ProTab",
            name="ablation_aggregation",
            tags=["ablation", "summary"]
        )

        wandb.log({
            "ablation_table": wandb.Table(dataframe=df)
        })

        wandb.save(output_path, policy="now")
        wandb.finish()


if __name__ == "__main__":
    main()
