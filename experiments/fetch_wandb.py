import pandas as pd
import tqdm
import wandb

ENTITY = "jacek-karolczak"
PROJECT = "ProTab"
DATASETS = [
    "bng_ionosphere", "bng_pendigits", "cardiotocography", "codrna", "covertype", "credit_card", "diabetes", "heloc",
    "statlog_shuttle", "yeast"
]

MODELS = [
    "mlp", "xgboost", "ProTab", "random_forest",
    "tree", "logistic_regression"
]


def fetch_best_runs_targeted(entity, project, datasets, models):
    api = wandb.Api()
    run_data = []

    # Calculate total for progress bar
    total_combinations = len(datasets) * len(models)
    print(f"Fetching best runs for {total_combinations} combinations...")

    # Iterate through every combination
    with tqdm.tqdm(total=total_combinations) as pbar:
        for d_name in datasets:
            for model in models:

                # --- THE OPTIMIZATION ---
                # 1. filter: match specific dataset AND specific model
                # 2. order: sort by balanced_accuracy descending (-)
                # 3. limit: we only process the first result
                filters = {
                    "config.data.name": d_name,
                    "config.architecture": model
                }

                runs = api.runs(
                    f"{entity}/{project}",
                    filters=filters,
                    order="-summary_metrics.eval_balanced_accuracy",
                    per_page=1
                )

                # Grab the first run (if it exists)
                # This executes the query but only pulls the top result
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

    # Convert to DataFrame
    df = pd.DataFrame(run_data)
    return df


# --- Execution ---
df_results = fetch_best_runs_targeted(ENTITY, PROJECT, DATASETS, MODELS)

if not df_results.empty:
    print("\nExtraction Complete. Preview:")
    # Sort for clean viewing
    df_results = df_results.sort_values(['dataset', 'eval_balanced_accuracy'], ascending=[True, False])
    print(df_results.head(10).to_string())

    # Save to CSV for easy use
    df_results.to_csv("best_metrics_summary.csv", index=False)
    print("\nSaved full results to 'best_metrics_summary.csv'")
else:
    print("No runs found. Please check your ENTITY and PROJECT names.")
