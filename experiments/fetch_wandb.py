from pathlib import Path

import pandas as pd
import tqdm
import wandb

from protab.data.named_data import TNamedData

MODELS = [
    "mlp", "xgboost", "ProTab", "random_forest",
    "tree", "logistic_regression"
]


def fetch_best_runs_targeted(datasets, models):
    api = wandb.Api()
    run_data = []

    total_combinations = len(datasets) * len(models)

    with tqdm.tqdm(total=total_combinations) as pbar:
        for d_name in datasets:
            for model in models:
                filters = {
                    "config.data.name": d_name,
                    "config.architecture": model
                }

                runs = api.runs(
                    f"jacek-karolczak/ProTab",
                    filters=filters,
                    order="-summary_metrics.eval_balanced_accuracy",
                    per_page=1
                )

                best_run = next(iter(runs), None)

                if best_run:
                    summary = best_run.summary
                    run_data.append({
                        "dataset": d_name,
                        "model": model,
                        "eval_balanced_accuracy": summary.get("eval_balanced_accuracy"),
                        "eval_accuracy": summary.get("eval_accuracy"),
                        "eval_cohen_kappa": summary.get("eval_cohen_kappa"),
                        "eval_f1_score": summary.get("eval_f1_score"),
                        "eval_precision": summary.get("eval_precision"),
                        "run_id": best_run.id
                    })

                pbar.update(1)

    df = pd.DataFrame(run_data)
    return df


df_results = fetch_best_runs_targeted(TNamedData.__args__, MODELS)

if not df_results.empty:
    df_results = df_results.sort_values(["dataset", "eval_balanced_accuracy"], ascending=[True, False])
    output_path = Path("results") / "best_metrics_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
