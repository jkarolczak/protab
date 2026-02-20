from pathlib import Path

import pandas as pd
import wandb

from protab.data.named_data import TNamedData

HPARAM_SYMBOLS = {
    "batch_size": "$B$",
    "n_patches": "$P$",
    "n_prototypes": "$R$",
    "prototype_dim": "$E$",
    "patch_len": "$k$",
    "learning_rate": "$lr$",
    "weight_decay": "$\\lambda$"
}

COLUMN_ORDER = ["Dataset", "$B$", "$P$", "$R$", "$E$", "$k$", "$lr$", "$\\lambda$"]


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_scientific(value):
    if isinstance(value, float) and (value < 1e-3 or value > 1e4):
        base, exponent = f"{value:.1e}".split('e')
        exponent = int(exponent)
        if base == "1.0":
            return f"$10^{{{exponent}}}$"
        return f"${base} \\times 10^{{{exponent}}}$"
    elif isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> None:
    api = wandb.Api()
    all_records = []

    for dataset_name in TNamedData.__args__:
        filters = {
            "config.data.name": dataset_name,
            "config.architecture": "ProTab",
            "tags": {"$in": ["hyperparameter_tuning"]}
        }

        runs = api.runs(
            f"jacek-karolczak/ProTab",
            filters=filters,
            order="-summary_metrics.eval_balanced_accuracy",
            per_page=1
        )

        run = next(iter(runs), None)
        if not run:
            continue

        flat_config = flatten_dict(run.config)

        row = {"Dataset": dataset_name.replace("_", "\\_")}

        for key, value in flat_config.items():
            key_ending = key.split('.')[-1]
            if key_ending in HPARAM_SYMBOLS:
                symbol = HPARAM_SYMBOLS[key_ending]

                if symbol in ["$lr$", "$\\lambda$"]:
                    row[symbol] = format_scientific(value)
                else:
                    row[symbol] = str(value)

        for col in COLUMN_ORDER:
            if col not in row:
                row[col] = "-"

        all_records.append(row)

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    df = df[[col for col in COLUMN_ORDER if col in df.columns]]

    caption_text = (
        "Best hyperparameters discovered for the ProTab architecture across all datasets. "
        "Symbols correspond to: $B$ (batch size), $P$ (number of patches), $R$ (number of prototypes), "
        "$E$ (embedding dimensionality), $k$ (features selected per patch), $lr$ (learning rate), "
        "and $\\lambda$ (weight decay)."
    )

    latex_table = df.to_latex(
        index=False,
        escape=False,
        caption=caption_text,
        label="tab:best_hyperparameters",
        column_format="l" + "c" * (len(df.columns) - 1)
    )

    output_path = Path("results") / "best_hyperparameters_table.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
