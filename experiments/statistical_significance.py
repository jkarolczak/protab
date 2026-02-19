import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
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

from protab.data.dataset import (DataContainer,
                                 DataContainerConfig)
from protab.data.named_data import TNamedData
from protab.training.config import fetch_best_run
from protab.training.reproducibility import set_seed

MODELS = ["tree", "random_forest", "xgboost", "mlp", "ProTab"]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes metrics taking into account class imbalances."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0)
    }


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--device", type=str, default="cpu")
@click.option("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations for the test set.")
@click.option("--log-wandb", is_flag=True, help="Enable logging results to WandB.")
def main(
        dataset_name: TNamedData,
        device: str,
        n_bootstrap: int,
        log_wandb: bool
) -> None:
    api = wandb.Api()
    best_runs = {}

    metrics_keys = ["accuracy", "balanced_accuracy", "f1_score", "precision"]

    for model in MODELS:
        filters = {
            "config.data.name": dataset_name,
            "config.architecture": model,
            "tags": {"$in": ["hyperparameter_tuning"]}
        }

        runs = api.runs(
            "jacek-karolczak/ProTab",
            filters=filters,
            order="-summary_metrics.eval_balanced_accuracy",
            per_page=1
        )

        run = next(iter(runs), None)
        if run:
            best_runs[model] = run

    if not best_runs:
        return

    data_container = DataContainer(DataContainerConfig(name=dataset_name))
    x_train, y_train_ohe = data_container.x_train, data_container.y_train.values
    x_test, y_test_ohe = data_container.x_test, data_container.y_test.values

    y_train = np.argmax(y_train_ohe, axis=1)
    y_test = np.argmax(y_test_ohe, axis=1)

    predictions = {}

    for model_name, run in best_runs.items():
        set_seed()
        if model_name == "ProTab":
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
            except Exception:
                continue
        else:
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

    if not predictions:
        return

    bootstrapped_metrics = {model: {m: [] for m in metrics_keys} for model in predictions.keys()}

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        y_test_boot = y_test[indices]

        for model_name, preds in predictions.items():
            preds_boot = preds[indices]
            mets = compute_metrics(y_test_boot, preds_boot)
            for k, v in mets.items():
                bootstrapped_metrics[model_name][k].append(v)

    formatted_results = {model: {"Model": model.replace("_", "\\_")} for model in predictions.keys()}
    available_models = [m for m in MODELS if m in bootstrapped_metrics]

    if not available_models:
        return

    for metric in metrics_keys:
        best_model_for_metric = max(available_models, key=lambda m: np.mean(bootstrapped_metrics[m][metric]))
        best_vals = np.array(bootstrapped_metrics[best_model_for_metric][metric])

        for model_name in available_models:
            model_vals = np.array(bootstrapped_metrics[model_name][metric])
            mean_val = np.mean(model_vals)

            if model_name == best_model_for_metric:
                is_bold = True
            else:
                diffs = best_vals - model_vals
                p_val = np.mean(diffs <= 0)
                is_bold = p_val >= 0.05

            if is_bold:
                formatted_results[model_name][metric] = f"\\textbf{{{mean_val:.4f}}}"
            else:
                formatted_results[model_name][metric] = f"{mean_val:.4f}"

    all_records = []
    for m in MODELS:
        if m in formatted_results:
            all_records.append(formatted_results[m])

    df = pd.DataFrame(all_records)
    df = df.rename(columns={
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1_score": "Macro F1",
        "precision": "Macro Precision"
    })

    caption_text = (
        f"Bootstrapped Test Set Metrics for {dataset_name.replace('_', '\\_')}. "
        "For each metric, the best result and those not significantly different from it "
        "($p \\ge 0.05$) are highlighted in bold."
    )

    latex_table = df.to_latex(
        index=False,
        escape=False,
        caption=caption_text,
        label=f"tab:results_{dataset_name}",
        column_format="lcccc"
    )

    output_path = Path("results") / f"results_table_{dataset_name}.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_table)

    if log_wandb:
        wandb.init(
            project="ProTab",
            entity="jacek-karolczak",
            name=f"{dataset_name}_significance_test",
            tags=["statistical_significance", dataset_name]
        )

        wandb.log({
            "dataset": dataset_name,
            "results_table": wandb.Table(dataframe=df)
        })

        wandb.save(output_path, policy="now")
        wandb.finish()


if __name__ == "__main__":
    main()
