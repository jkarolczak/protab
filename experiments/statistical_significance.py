"""
Bootstrapped test set evaluation with statistical significance testing.

Produces a single LaTeX table (matching manuscript Table 2 format) with all datasets
and all models including MEDIC.

Usage:
    python experiments/statistical_significance.py [--device cpu] [--n-bootstrap 1000] [--log-wandb]
    python experiments/statistical_significance.py --datasets diabetes heloc
"""

import os
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             precision_score)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Add project root and MEDIC submodule to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vendor", "medic"))

from medic.classifiers.medic_classifier import Medic
from medic.preprocessing import StandardScaler as MedicScaler

from p2tab.data.dataset import (DataContainer,
                                DataContainerConfig)
from p2tab.data.named_data import TNamedData
from p2tab.training.config import fetch_best_run
from p2tab.training.reproducibility import set_seed

from experiments.hyperparameter_tuning_medic import (
    generate_definitions,
    load_raw_data,
)

# Model display order and names
MODELS = ["tree", "random_forest", "xgboost", "mlp", "medic", "P2Tab"]
MODEL_DISPLAY_NAMES = {
    "tree": "Tree",
    "random_forest": "RF",
    "xgboost": "XGB",
    "mlp": "MLP",
    "medic": "MEDIC",
    "P2Tab": "\\name",
}

# Dataset order matching the manuscript
DATASET_ORDER = ["codrna", "credit_card", "diabetes", "heloc", "bng_ionosphere" "statlog_shuttle"]
# "bng_pendigits",

DATASET_DISPLAY_NAMES = {
    "codrna": "CodRNA",
    "credit_card": "Credit Card",
    "diabetes": "Diabetes",
    "heloc": "HELOC",
    "bng_ionosphere": "Ionosphere",
    "bng_pendigits": "Pendigits",
    "statlog_shuttle": "Statlog Shuttle",
    "yeast": "Yeast",
    "covertype": "Covertype",
}

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_predictions_for_dataset(
        dataset_name: TNamedData,
        device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Get test predictions for all models on a given dataset.

    Returns y_test and a dict of model_name -> predictions.
    """
    api = wandb.Api()
    best_runs = {}

    for model in MODELS:
        if model == "medic":
            filters = {
                "config.data.name": dataset_name,
                "config.architecture": "medic",
                "tags": {"$in": ["hyperparameter_tuning", "medic"]},
                "summary_metrics.eval_balanced_accuracy": {"$ne": None},
            }
        else:
            filters = {
                "config.data.name": dataset_name,
                "config.architecture": model,
                "tags": {"$in": ["hyperparameter_tuning"]},
                "summary_metrics.eval_balanced_accuracy": {"$ne": None},
            }

        runs = api.runs(
            "jacek-karolczak/P2Tab",
            filters=filters,
            order="-summary_metrics.eval_balanced_accuracy",
            per_page=1,
        )

        run = next(iter(runs), None)
        if run:
            best_runs[model] = run

    # Load data
    data_container = DataContainer(DataContainerConfig(name=dataset_name))
    x_train = data_container.x_train
    y_train = np.argmax(data_container.y_train.values, axis=1)
    x_test = data_container.x_test
    y_test = np.argmax(data_container.y_test.values, axis=1)

    predictions = {}

    for model_name, run in best_runs.items():
        set_seed()

        if model_name == "P2Tab":
            try:
                _, _, _, _, model = fetch_best_run(
                    dataset_name, ["hyperparameter_tuning"], load_model=True
                )
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
                    logits = model(x_test_tensor)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions[model_name] = preds
            except Exception as e:
                print(f"    [WARN] P2Tab failed on {dataset_name}: {e}")
                continue

        elif model_name == "medic":
            try:
                config = run.config
                model_cfg = config.get("model", {})
                n_bins = model_cfg.get("n_bins", 5)
                n_prototypes = model_cfg.get("n_prototypes", 32)
                n_patches = model_cfg.get("n_patches", 64)
                hidden_dim = model_cfg.get("hidden_dim", 8)
                definitions = model_cfg.get("definitions", None)

                x_train_raw, y_train_raw, _, _, x_test_raw, y_test_raw, feat_names = load_raw_data(dataset_name)
                n_classes = len(np.unique(y_train_raw))

                if definitions is None:
                    definitions = generate_definitions(x_train_raw, feat_names, n_bins=n_bins)

                x_train_t = torch.tensor(x_train_raw, dtype=torch.float32)
                x_test_t = torch.tensor(x_test_raw, dtype=torch.float32)

                scaler = MedicScaler(definitions=definitions)
                x_train_scaled = scaler.fit_transform(x_train_t.clone())
                x_test_scaled = scaler.transform(x_test_t.clone())

                medic_model = Medic(
                    definitions=definitions,
                    n_classes=n_classes,
                    n_patches=n_patches,
                    n_prototypes=n_prototypes,
                    hidden_dim=hidden_dim,
                ).to(device)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        f = run.file("files/model_state_dict.pt")
                        f.download(root=tmp_dir, replace=True)
                        state_dict_path = Path(tmp_dir) / "files" / "model_state_dict.pt"
                    except Exception:
                        f = run.file("model_state_dict.pt")
                        f.download(root=tmp_dir, replace=True)
                        state_dict_path = Path(tmp_dir) / "model_state_dict.pt"

                    state_dict = torch.load(state_dict_path, map_location=device)
                    medic_model.load_state_dict(state_dict)

                medic_model.hard_binning = True
                medic_model.set_real_prototypes(x_train_scaled.to(device))
                medic_model.hard_parts = True
                medic_model.eval()

                with torch.no_grad():
                    logits = medic_model(x_test_scaled.to(device))
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions[model_name] = preds
            except Exception as e:
                print(f"    [WARN] MEDIC failed on {dataset_name}: {e}")
                continue

        else:
            try:
                params = run.config.get("model", {})
                if model_name == "xgboost":
                    clf = XGBClassifier(**params)
                elif model_name == "mlp":
                    clf = MLPClassifier(**params)
                elif model_name == "random_forest":
                    clf = RandomForestClassifier(**params)
                elif model_name == "tree":
                    clf = DecisionTreeClassifier(**params)
                else:
                    continue

                clf.fit(x_train, y_train)
                preds = clf.predict(x_test)
                predictions[model_name] = preds
            except Exception as e:
                print(f"    [WARN] {model_name} failed on {dataset_name}: {e}")
                continue

    return y_test, predictions


def bootstrap_and_format(
        y_test: np.ndarray,
        predictions: dict[str, np.ndarray],
        n_bootstrap: int = 1000,
) -> list[dict]:
    """Run bootstrap test and return formatted results for one dataset.

    Returns list of dicts with model name and formatted metric values.
    """
    metrics_keys = ["accuracy", "balanced_accuracy", "f1_score", "precision"]
    available_models = [m for m in MODELS if m in predictions]

    if not available_models:
        return []

    # Bootstrap
    bootstrapped_metrics = {model: {m: [] for m in metrics_keys} for model in available_models}

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_boot = y_test[indices]

        for model_name in available_models:
            preds_boot = predictions[model_name][indices]
            mets = compute_metrics(y_test_boot, preds_boot)
            for k, v in mets.items():
                bootstrapped_metrics[model_name][k].append(v)

    # Format with significance testing
    results = []
    for model_name in available_models:
        row = {"model": model_name, "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name)}

        for metric in metrics_keys:
            best_model = max(available_models, key=lambda m: np.mean(bootstrapped_metrics[m][metric]))
            best_vals = np.array(bootstrapped_metrics[best_model][metric])
            model_vals = np.array(bootstrapped_metrics[model_name][metric])
            mean_val = np.mean(model_vals)

            if model_name == best_model:
                is_bold = True
            else:
                diffs = best_vals - model_vals
                p_val = np.mean(diffs <= 0)
                is_bold = p_val >= 0.05

            if is_bold:
                row[metric] = f"\\textbf{{{mean_val:.4f}}}"
            else:
                row[metric] = f"{mean_val:.4f}"

            row[f"{metric}_raw"] = mean_val

        results.append(row)

    return results


def generate_latex_table(all_results: dict[str, list[dict]]) -> str:
    """Generate the full LaTeX table matching manuscript format."""
    n_models = len(MODELS)

    lines = []
    lines.append(r"\begin{table}[tbp]")
    lines.append(
        r"\caption{Bootstrapped test set metric. For each metric, the best result and those not significantly different from it ($p \ge 0.05$) are highlighted in bold.}")
    lines.append(r"\label{tab:results_predictive_performance}")
    lines.append(r"\centering")
    lines.append(r"    \begin{tabular}{llcccc}")
    lines.append(r"        \toprule")
    lines.append(r"        Dataset & Model & Acc. & Bal. Acc. & F1$_{\text{Macro}}$ & Precision$_{\text{Macro}}$ \\")
    lines.append(r"        \midrule")

    datasets_in_order = [d for d in DATASET_ORDER if d in all_results]
    print(f"  [DEBUG] DATASET_ORDER: {DATASET_ORDER}")
    print(f"  [DEBUG] all_results keys: {list(all_results.keys())}")
    print(f"  [DEBUG] datasets_in_order: {datasets_in_order}")

    # Fallback: if no overlap with DATASET_ORDER, use whatever is in all_results
    if not datasets_in_order:
        datasets_in_order = list(all_results.keys())

    for ds_idx, dataset_name in enumerate(datasets_in_order):
        results = all_results[dataset_name]
        if not results:
            continue
        display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        n_rows = len(results)

        for row_idx, row in enumerate(results):
            if row_idx == 0:
                dataset_col = f"\\multirow{{{n_rows}}}{{*}}{{{display_name}}}"
            else:
                dataset_col = ""

            model_col = row["display_name"]
            acc = row["accuracy"]
            bal_acc = row["balanced_accuracy"]
            f1 = row["f1_score"]
            prec = row["precision"]

            lines.append(f"        {dataset_col} & {model_col} & {acc} & {bal_acc} & {f1} & {prec} \\\\")

        # Add midrule between datasets (not after last)
        if ds_idx < len(datasets_in_order) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


@click.command()
@click.option("--device", type=str, default="cpu")
@click.option("--n-bootstrap", type=int, default=1000)
@click.option("--datasets", type=str, multiple=True, default=None,
              help="Specific datasets to run. If empty, runs all in DATASET_ORDER.")
@click.option("--log-wandb", is_flag=True)
def main(device: str, n_bootstrap: int, datasets: tuple, log_wandb: bool) -> None:
    set_seed()

    if datasets:
        datasets_to_run = list(datasets)
    else:
        datasets_to_run = DATASET_ORDER

    # Filter to datasets that actually exist in TNamedData
    valid_datasets = [d for d in datasets_to_run if d in TNamedData.__args__]

    all_results = {}

    for dataset_name in valid_datasets:
        print(f"\n{'=' * 60}")
        print(f"  Processing: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            y_test, predictions = get_predictions_for_dataset(dataset_name, device)

            if not predictions:
                print(f"  [SKIP] No models found for {dataset_name}")
                continue

            print(f"  Models loaded: {list(predictions.keys())}")
            results = bootstrap_and_format(y_test, predictions, n_bootstrap)
            all_results[dataset_name] = results

            # Print quick summary
            print(f"  {'Model':<12} {'Bal.Acc':>10}")
            print(f"  {'-' * 22}")
            for row in results:
                print(f"  {row['display_name']:<12} {row['balanced_accuracy_raw']:>10.4f}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {dataset_name}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        print("\n[ERROR] No datasets completed successfully.")
        return

    # Generate LaTeX table
    print(f"\n  Datasets in results: {list(all_results.keys())}")
    for k, v in all_results.items():
        print(f"    {k}: {len(v)} rows")
    latex_table = generate_latex_table(all_results)

    # Save
    output_path = Path("results") / "results_table_all.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_table)

    print(f"\n{'=' * 60}")
    print(f"LaTeX table saved to: {output_path}")
    print(f"{'=' * 60}")
    print(latex_table)

    if log_wandb:
        wandb.init(
            project="P2Tab",
            entity="jacek-karolczak",
            name="all_datasets_significance_test",
            tags=["statistical_significance", "all_datasets"],
        )
        wandb.save(str(output_path), policy="now")
        wandb.finish()


if __name__ == "__main__":
    main()
