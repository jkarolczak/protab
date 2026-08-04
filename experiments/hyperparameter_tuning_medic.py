"""
Hyperparameter tuning for MEDIC baseline on P2Tab datasets.

This script runs MEDIC (Model for Explainable Diagnosis using Interpretable Concepts)
on P2Tab's datasets using the same train/eval/test splits and Optuna-based tuning protocol.

MEDIC definitions are auto-generated from the data: binary features get one-hot encoding,
continuous features get fuzzy binning with a tunable number of bins.

Usage:
    python experiments/hyperparameter_tuning_medic.py <dataset_name> [--n-trials 100] [--log-wandb]
"""

import os
import platform
import sys
import uuid

# Add project root and MEDIC submodule to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vendor", "medic"))

import click
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as tmf
import wandb
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from medic.classifiers.medic_classifier import Medic
from medic.preprocessing import StandardScaler as MedicScaler

from p2tab.data.named_data import TNamedData
from p2tab.training.reproducibility import set_seed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_definitions(x_train_raw: np.ndarray, feature_names: list[str], n_bins: int = 3) -> list[dict]:
    """Auto-generate MEDIC-compatible feature definitions from raw data.

    Binary features (exactly 2 unique values) -> one-hot encoding.
    Continuous/multi-valued features -> fuzzy binning with n_bins.
    """
    definitions = []
    for i, name in enumerate(feature_names):
        col = x_train_raw[:, i]
        unique_vals = np.unique(col[~np.isnan(col)])
        n_unique = len(unique_vals)

        if n_unique <= 2:
            definitions.append({"name": name, "binning": False, "n_values": max(n_unique, 2)})
        else:
            definitions.append({"name": name, "binning": True, "n_bins": n_bins})

    return definitions


def load_raw_data(dataset_name: TNamedData) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load raw (unscaled) data from P2Tab's CSV splits.

    Returns x_train, y_train, x_eval, y_eval, x_test, y_test as numpy arrays,
    plus the list of feature names.
    """
    import pandas as pd
    from pathlib import Path

    base = os.environ.get("PROTAB_DATASETS", "./data/")
    base_path = Path(base)

    def read_split(split: str):
        x = pd.read_csv(base_path / dataset_name / f"{split}_x.csv")
        y = pd.read_csv(base_path / dataset_name / f"{split}_y.csv")
        return x, y

    x_train_df, y_train_df = read_split("train")
    x_eval_df, y_eval_df = read_split("eval")
    x_test_df, y_test_df = read_split("test")

    feature_names = x_train_df.columns.tolist()

    # Encode categorical features as integers (same logic as DataContainer: binary -> 0/1)
    for col in feature_names:
        unique_vals = x_train_df[col].dropna().unique()
        if len(unique_vals) == 2:
            unique_vals_sorted = sorted(unique_vals)
            mapping = {val: i for i, val in enumerate(unique_vals_sorted)}
            x_train_df[col] = x_train_df[col].map(mapping).fillna(0).astype(float)
            x_eval_df[col] = x_eval_df[col].map(mapping).fillna(0).astype(float)
            x_test_df[col] = x_test_df[col].map(mapping).fillna(0).astype(float)

    # Fill remaining NaNs with column mean from train
    for col in feature_names:
        mean_val = x_train_df[col].mean()
        x_train_df[col] = x_train_df[col].fillna(mean_val)
        x_eval_df[col] = x_eval_df[col].fillna(mean_val)
        x_test_df[col] = x_test_df[col].fillna(mean_val)

    x_train = x_train_df.to_numpy(dtype=np.float32)
    x_eval = x_eval_df.to_numpy(dtype=np.float32)
    x_test = x_test_df.to_numpy(dtype=np.float32)

    # Encode y as integer class labels
    y_col = y_train_df.columns[0]
    all_classes = sorted(y_train_df[y_col].unique())
    class_map = {c: i for i, c in enumerate(all_classes)}

    y_train = y_train_df[y_col].map(class_map).to_numpy(dtype=np.int64)
    y_eval = y_eval_df[y_col].map(class_map).to_numpy(dtype=np.int64)
    y_test = y_test_df[y_col].map(class_map).to_numpy(dtype=np.int64)

    return x_train, y_train, x_eval, y_eval, x_test, y_test, feature_names


def train_medic(
        model: Medic,
        train_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        learning_rate: float,
        epochs_stage_1: int,
        epochs_stage_2: int,
        epochs_stage_3: int,
        penalty_l1: float,
        penalty_diversity: float,
        x_train_scaled: torch.Tensor,
) -> None:
    """Train MEDIC using the 3-stage protocol from the original paper."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Stage 1: Soft binning + soft parts
    model.train()
    for epoch in range(epochs_stage_1):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch) + penalty_l1 * model.l1_factor + penalty_diversity * model.diversity_factor
            if torch.isnan(loss):
                return
            loss.backward()
            optimizer.step()

    # Stage 2: Hard binning
    if epochs_stage_2 > 0:
        model.hard_binning = True
        for epoch in range(epochs_stage_2):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss):
                    return
                loss.backward()
                optimizer.step()

    # Stage 3: Real prototypes + hard parts
    if epochs_stage_3 > 0:
        model.set_real_prototypes(x_train_scaled.to(device))
        model.hard_parts = True
        for epoch in range(epochs_stage_3):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss):
                    return
                loss.backward()
                optimizer.step()


def evaluate_medic(
        model: Medic,
        eval_loader: DataLoader,
        device: torch.device,
        n_classes: int,
) -> dict[str, float]:
    """Evaluate MEDIC and return metrics matching P2Tab's evaluation protocol."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in eval_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            logits_list.append(outputs.cpu())
            labels_list.append(y_batch)

    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    task = "multiclass"

    metrics = {
        "accuracy": tmf.classification.accuracy(
            logits_all, labels_all, task=task, average="micro", num_classes=n_classes
        ).item(),
        "balanced_accuracy": tmf.classification.recall(
            logits_all, labels_all, average="macro", task=task, num_classes=n_classes
        ).item(),
        "precision": tmf.classification.precision(
            logits_all, labels_all, average="macro", task=task, num_classes=n_classes
        ).item(),
        "f1_score": tmf.classification.f1_score(
            logits_all, labels_all, average="macro", task=task, num_classes=n_classes
        ).item(),
        "cohen_kappa": tmf.cohen_kappa(
            logits_all, labels_all, task=task, num_classes=n_classes
        ).item(),
    }

    return metrics


def objective(trial: optuna.Trial, dataset_name: TNamedData, device_str: str, log_wandb: bool) -> float:
    set_seed()
    device = torch.device(device_str)

    # Load raw data
    x_train, y_train, x_eval, y_eval, x_test, y_test, feature_names = load_raw_data(dataset_name)

    n_classes = len(np.unique(y_train))

    # Hyperparameters
    n_bins = trial.suggest_categorical("n_bins", [3, 5, 7])
    n_prototypes = trial.suggest_categorical("n_prototypes", [16, 32, 48, 64, 96])
    n_patches_multiplier = trial.suggest_categorical("n_patches_multiplier", [1, 2, 3])
    n_patches = n_prototypes * n_patches_multiplier
    hidden_dim = trial.suggest_categorical("hidden_dim", [3, 5, 8, 12, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    penalty_l1 = trial.suggest_float("penalty_l1", 1e-4, 0.1, log=True)
    penalty_diversity = trial.suggest_float("penalty_diversity", 1e-4, 0.1, log=True)
    epochs_stage_1 = trial.suggest_categorical("epochs_stage_1", [30, 40, 50])
    epochs_stage_2 = trial.suggest_categorical("epochs_stage_2", [20, 30])
    epochs_stage_3 = trial.suggest_categorical("epochs_stage_3", [20, 30])

    # Generate definitions
    definitions = generate_definitions(x_train, feature_names, n_bins=n_bins)

    # Convert to tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
    y_eval_t = torch.tensor(y_eval, dtype=torch.long)

    # Scale using MEDIC's scaler (only scales continuous features)
    scaler = MedicScaler(definitions=definitions)
    x_train_scaled = scaler.fit_transform(x_train_t.clone())
    x_eval_scaled = scaler.transform(x_eval_t.clone())

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(x_train_scaled, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        TensorDataset(x_eval_scaled, y_eval_t),
        batch_size=batch_size, shuffle=False
    )

    # Build model
    model = Medic(
        definitions=definitions,
        n_classes=n_classes,
        n_patches=n_patches,
        n_prototypes=n_prototypes,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # W&B logging
    platform_name = platform.node()
    if log_wandb:
        wandb.init(
            project="P2Tab",
            entity="jacek-karolczak",
            name=f"{dataset_name}_medic_{uuid.uuid4()}",
            mode="online",
            tags=["hyperparameter_tuning", "baseline", "medic"],
            config={
                "architecture": "medic",
                "data": {"name": dataset_name},
                "model": {
                    "n_bins": n_bins,
                    "n_prototypes": n_prototypes,
                    "n_patches": n_patches,
                    "hidden_dim": hidden_dim,
                    "definitions": definitions,
                    "feature_names": feature_names,
                },
                "trainer": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "penalty_l1": penalty_l1,
                    "penalty_diversity": penalty_diversity,
                    "epochs_stage_1": epochs_stage_1,
                    "epochs_stage_2": epochs_stage_2,
                    "epochs_stage_3": epochs_stage_3,
                },
                "n_classes": n_classes,
                "platform": platform_name,
            },
            reinit=True,
        )

    # Train
    train_medic(
        model=model,
        train_loader=train_loader,
        device=device,
        criterion=criterion,
        learning_rate=learning_rate,
        epochs_stage_1=epochs_stage_1,
        epochs_stage_2=epochs_stage_2,
        epochs_stage_3=epochs_stage_3,
        penalty_l1=penalty_l1,
        penalty_diversity=penalty_diversity,
        x_train_scaled=x_train_scaled,
    )

    # Evaluate
    metrics = evaluate_medic(model, eval_loader, device, n_classes)

    if log_wandb:
        wandb.log({f"eval_{k}": v for k, v in metrics.items()})

        # Save model checkpoint and metadata for later explanation analysis
        import json
        model_path = os.path.join(wandb.run.dir, "model_state_dict.pt")
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path, policy="now")

        meta_path = os.path.join(wandb.run.dir, "medic_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "definitions": definitions,
                "n_classes": n_classes,
                "n_patches": n_patches,
                "n_prototypes": n_prototypes,
                "hidden_dim": hidden_dim,
                "n_bins": n_bins,
                "feature_names": feature_names,
                "scaler_mean": scaler.mean.tolist() if scaler.mean is not None else None,
                "scaler_std": scaler.std.tolist() if scaler.std is not None else None,
                "scaler_continuous_indices": scaler.continuous_indices,
            }, f)
        wandb.save(meta_path, policy="now")

        wandb.finish()

    return metrics["balanced_accuracy"]


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--n-trials", type=int, default=100)
@click.option("--device", type=str, default="cpu")
@click.option("--log-wandb", is_flag=True)
def main(dataset_name: TNamedData, n_trials: int, device: str, log_wandb: bool) -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, dataset_name, device, log_wandb),
        n_trials=n_trials,
        timeout=3 * 24 * 60 * 60,
    )

    print(f"\nBest trial for {dataset_name}:")
    print(f"  Balanced Accuracy: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
