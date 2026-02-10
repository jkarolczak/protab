import os
import platform
import uuid

import click
import numpy as np
import optuna
import torch
import torchmetrics.functional as tmf
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from protab.data.named_data import TNamedData
from protab.training.config import read_data_and_configs

MODELS = ["xgboost", "mlp", "random_forest", "tree", "logistic_regression"]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_optimizer_params(trial: optuna.Trial, model_name: str):
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [50, 75, 100, 150]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_jobs": 1, "verbosity": 0
        }
    elif model_name == "mlp":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes",
                                                            ["50", "100", "100,50", "100,100", "200"]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 100
        }
        params["hidden_layer_sizes"] = [int(d.strip()) for d in params["hidden_layer_sizes"].split(",")]

        return params
    elif model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "n_jobs": 1
        }
    elif model_name == "tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "max_iter": 1000, "n_jobs": 1
        }


def objective(trial: optuna.Trial, dataset_name: TNamedData, model_name: str, log_wandb: bool) -> float:
    data_container, _, trainer_config = read_data_and_configs(dataset_name)

    y_train = np.argmax(data_container.y_train.values, axis=1)
    y_eval = np.argmax(data_container.y_eval.values, axis=1)
    x_train, x_eval = data_container.x_train, data_container.x_eval

    params = get_optimizer_params(trial, model_name)

    if model_name == "xgboost":
        model = XGBClassifier(**params)
    elif model_name == "mlp":
        model = MLPClassifier(**params)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_name == "tree":
        model = DecisionTreeClassifier(**params)
    elif model_name == "logistic_regression":
        model = LogisticRegression(**params)

    platform_name = platform.node()

    if log_wandb:
        wandb.init(
            project=trainer_config.wandb_config.project,
            entity=trainer_config.wandb_config.entity,
            name=f"{dataset_name}_{model_name}_{uuid.uuid4()}",
            mode="online",
            tags=["hyperparameter_tuning", "baseline", model_name],
            config={
                "architecture": model_name,
                "model": params,
                "model_type": model_name,
                "platform": platform_name},
            reinit=True
        )

    model.fit(x_train, y_train)
    logits_all = torch.tensor(model.predict_proba(x_eval), dtype=torch.float32)
    labels_all = torch.tensor(y_eval, dtype=torch.long)

    num_classes = logits_all.shape[-1]
    task = "multiclass"

    metrics = {
        "accuracy": tmf.classification.accuracy(logits_all, labels_all, task=task, average="micro",
                                                num_classes=num_classes).item(),
        "balanced_accuracy": tmf.classification.recall(logits_all, labels_all, average="macro", task=task,
                                                       num_classes=num_classes).item(),
        "precision": tmf.classification.precision(logits_all, labels_all, average="macro", task=task,
                                                  num_classes=num_classes).item(),
        "f1_score": tmf.classification.f1_score(logits_all, labels_all, average="macro", task=task,
                                                num_classes=num_classes).item(),
        "cohen_kappa": tmf.cohen_kappa(logits_all, labels_all, task=task, num_classes=num_classes).item()
    }

    if log_wandb:
        wandb.log({f"eval_{k}": v for k, v in metrics.items()})
        wandb.finish()

    return metrics["balanced_accuracy"]


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--model", type=click.Choice(MODELS + ["all"]), default="all")
@click.option("--n-trials", type=int, default=100)
@click.option("--log-wandb", is_flag=True)
def main(dataset_name: TNamedData, model: str, n_trials: int, log_wandb: bool) -> None:
    models_to_run = MODELS if model == "all" else [model]
    for m in models_to_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, dataset_name, m, log_wandb), n_trials=n_trials, timeout=3 * 24 * 60 * 60)
        print(f"Best {m}: {study.best_value} with {study.best_params}")


if __name__ == "__main__":
    main()
