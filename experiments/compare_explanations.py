"""
Compare explanations between P2Tab and MEDIC on a given dataset.

Metrics computed:
1. Prototype sparsity: average number of active features per prototypical part
2. Feature overlap (Jaccard): pairwise Jaccard similarity of feature masks across prototypes
3. Top-k purity: for each prototype, fraction of k-nearest patches in embedding space
   that belong to the same class as the prototype's assigned class

Both models are trained from scratch using dataset-specific configs (P2Tab) or
hyperparameters (MEDIC) and then compared side-by-side.

Usage:
    python experiments/compare_explanations.py diabetes [--device cpu] [--p2tab-tag hyperparameter_tuning]
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vendor", "medic"))

from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from medic.classifiers.medic_classifier import Medic
from medic.preprocessing import StandardScaler as MedicScaler

from p2tab.data.dataset import DataContainer, SimpleDataset
from p2tab.data.named_data import TNamedData
from p2tab.models.p2tab import P2Tab
from p2tab.training.config import fetch_best_run, read_data_and_configs
from p2tab.training.reproducibility import set_seed
from p2tab.training.trainer import P2TabTrainer

from experiments.hyperparameter_tuning_medic import (
    generate_definitions,
    load_raw_data,
    train_medic,
)


# =============================================================================
# P2Tab: extract feature masks and embeddings
# =============================================================================

def get_p2tab_masks(model: P2Tab) -> torch.Tensor:
    """Extract binary feature masks from P2Tab's patching layer.

    Returns:
        masks: (n_patches, n_features) binary tensor
    """
    weights = model.patching.weights.data  # (n_patches, n_features)
    patch_len = model.config.patching.patch_len
    _, topk_indices = torch.topk(weights, patch_len, dim=-1)

    masks = torch.zeros_like(weights)
    masks.scatter_(-1, topk_indices, 1.0)
    return masks


def get_p2tab_prototype_masks(model: P2Tab, data_container: DataContainer, device: torch.device) -> torch.Tensor:
    """Get per-prototype feature masks by finding which patch is closest to each prototype.

    Returns:
        proto_masks: (n_prototypes, n_features) binary tensor
    """
    # The patch masks are shared — each prototype is matched to one patch
    # We use the classifier weights to determine which patches matter per prototype,
    # but more directly: after set_real_prototypes, each prototype is grounded in a
    # specific patch. The patch index is stored in global_nearest_idcs[:, 1].
    # However the simplest approach: each prototype selects the nearest patch from the
    # patching layer's masks.

    # Since prototypes live in embedding space and patches are mapped to prototypes
    # via minimum distance, the relevant mask for each prototype is the mask of the
    # patch that was selected during set_real_prototypes.

    # The patch masks are identical for all inputs (they're learned parameters).
    # So we just need patching masks (n_patches x n_features) and the patch-to-prototype
    # assignment from set_real_prototypes.

    # After set_real_prototypes, best_patches already contains the masked feature values.
    # Features with NaN are inactive. So we can derive masks from that directly.

    # Let's just use the patch-level masks since those define the explanation.
    return get_p2tab_masks(model)


def compute_p2tab_purity(
        model: P2Tab,
        dataloader: DataLoader,
        device: torch.device,
) -> dict:
    """Compute top-k purity for P2Tab prototypes."""
    model.eval()

    patch_embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for x, y, *_ in dataloader:
            x = x.to(device).float()
            y = y.to(device)

            patches_embeddings = model.embeddings(x)  # (B, P, E)

            if y.dim() > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)

            # Get patch-to-prototype assignment
            _, patches_idcs = model.prototypes(patches_embeddings)  # patches_idcs: (B, n_protos)

            B, P, E = patches_embeddings.shape
            # Use only the active (selected) patches
            active_mask = torch.zeros((B, P), dtype=torch.bool, device=device)
            active_mask.scatter_(1, patches_idcs, True)

            active_patches = patches_embeddings[active_mask]
            active_labels = y.unsqueeze(1).expand(B, P)[active_mask]

            patch_embeddings_list.append(active_patches.cpu())
            labels_list.append(active_labels.cpu())

    all_patches = torch.cat(patch_embeddings_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Get prototype embeddings
    proto_emb = F.normalize(model.prototypes.prototypes.data, p=2, dim=-1).cpu()

    # Determine prototype-to-class assignment from classifier weights
    w_cls = model.classifier.network[0].weight.data.cpu()
    if w_cls.shape[0] == 1:
        assigned_classes = (w_cls[0] > 0).long()
    else:
        assigned_classes = torch.argmax(w_cls, dim=0)

    # Compute distances from prototypes to all patches
    dists = torch.cdist(proto_emb, all_patches)  # (n_protos, n_patches_total)

    k_vals = [3, 5, 7, 9, 11]
    max_k = min(max(k_vals), all_patches.shape[0])
    valid_k_vals = [k for k in k_vals if k <= max_k]

    purity_records = []
    for j in range(proto_emb.shape[0]):
        assigned_c = assigned_classes[j].item()
        top_indices = torch.topk(dists[j], k=max_k, largest=False).indices
        top_labels = all_labels[top_indices]

        record = {"prototype_idx": j, "assigned_class": assigned_c}
        for k in valid_k_vals:
            purity = (top_labels[:k] == assigned_c).float().mean().item()
            record[f"purity_{k}"] = purity
        purity_records.append(record)

    return purity_records


# =============================================================================
# MEDIC: extract feature masks and embeddings
# =============================================================================

def get_medic_feature_masks(model: Medic, definitions: list[dict], top_k: int | None = None) -> torch.Tensor:
    """Extract per-prototype feature masks from MEDIC.

    MEDIC's prototypical_parts are (n_prototypes, embedding_dim) where embedding_dim
    is the sum of all bin/value dimensions. We map back to original features by computing
    the aggregate activation magnitude per feature, then apply top-k selection for fair
    comparison with P2Tab (which uses a fixed patch_len).

    Args:
        model: Trained MEDIC model with real prototypes set.
        definitions: Feature definitions (binning/n_bins/n_values per feature).
        top_k: If provided, keep only the top-k features per prototype (by magnitude).
                If None, use a relative threshold (> mean activation for that prototype).

    Returns:
        masks: (n_prototypes, n_original_features) binary tensor
    """
    if model.prototypical_parts is None:
        raise ValueError("MEDIC model has no real prototypes set. Call set_real_prototypes first.")

    proto_parts = model.prototypical_parts.detach().cpu()  # (n_prototypes, embedding_dim)
    n_prototypes = proto_parts.shape[0]
    n_features = len(definitions)

    # Compute per-feature importance as the L2 norm of that feature's bin dimensions
    feature_importance = torch.zeros(n_prototypes, n_features)

    offset = 0
    for feat_idx, defn in enumerate(definitions):
        if defn["binning"]:
            dim = defn["n_bins"]
        else:
            dim = defn["n_values"]

        feat_slice = proto_parts[:, offset:offset + dim]
        # Use L2 norm across bin dimensions as feature importance
        feature_importance[:, feat_idx] = feat_slice.norm(dim=-1)
        offset += dim

    # Apply top-k or relative threshold
    if top_k is not None:
        # Select top-k features per prototype (like P2Tab's fixed patch_len)
        _, topk_indices = torch.topk(feature_importance, min(top_k, n_features), dim=-1)
        masks = torch.zeros_like(feature_importance)
        masks.scatter_(-1, topk_indices, 1.0)
    else:
        # Relative threshold: feature is active if importance > mean importance for that prototype
        mean_importance = feature_importance.mean(dim=-1, keepdim=True)
        masks = (feature_importance > mean_importance).float()

    return masks


def compute_medic_purity(
        model: Medic,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        device: torch.device,
        batch_size: int = 256,
) -> dict:
    """Compute top-k purity for MEDIC prototypes."""
    model.eval()
    model = model.to(device)

    # Get all part embeddings (features in MEDIC's terminology)
    feature_list = []
    label_list = []

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            features = model.forward_features(x_batch)  # (B, n_patches, hidden_dim)
            B, P, H = features.shape

            # Expand labels to match patches
            y_expanded = y_batch.unsqueeze(1).expand(B, P)

            feature_list.append(features.view(-1, H).cpu())
            label_list.append(y_expanded.reshape(-1).cpu())

    all_features = torch.cat(feature_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)

    # Get prototype embeddings
    proto_emb = model._prototypical_parts_embeddings.data.cpu()

    # Determine prototype-to-class assignment from classifier weights
    w_cls = model._classification_head.weight.data.cpu()
    if w_cls.shape[0] == 1:
        assigned_classes = (w_cls[0] > 0).long()
    else:
        assigned_classes = torch.argmax(w_cls, dim=0)

    # Compute distances
    dists = torch.cdist(proto_emb, all_features)  # (n_protos, total_patches)

    k_vals = [3, 5, 7, 9, 11]
    max_k = min(max(k_vals), all_features.shape[0])
    valid_k_vals = [k for k in k_vals if k <= max_k]

    purity_records = []
    for j in range(proto_emb.shape[0]):
        assigned_c = assigned_classes[j].item()
        top_indices = torch.topk(dists[j], k=max_k, largest=False).indices
        top_labels = all_labels[top_indices]

        record = {"prototype_idx": j, "assigned_class": assigned_c}
        for k in valid_k_vals:
            purity = (top_labels[:k] == assigned_c).float().mean().item()
            record[f"purity_{k}"] = purity
        purity_records.append(record)

    return purity_records


# =============================================================================
# Shared metrics
# =============================================================================

def compute_sparsity(masks: torch.Tensor) -> dict:
    """Compute sparsity statistics from binary masks.

    Args:
        masks: (n_prototypes, n_features) binary tensor

    Returns:
        dict with mean, std, min, max of active features per prototype
    """
    active_per_proto = masks.sum(dim=-1)
    n_features = masks.shape[1]
    return {
        "mean_active_features": active_per_proto.mean().item(),
        "std_active_features": active_per_proto.std().item(),
        "min_active_features": active_per_proto.min().item(),
        "max_active_features": active_per_proto.max().item(),
        "n_features_total": n_features,
        "mean_sparsity_ratio": 1 - (active_per_proto.mean().item() / n_features),
    }


def compute_feature_focus_gini(masks: torch.Tensor) -> float:
    """Compute Feature Focus as the Gini coefficient of aggregated feature frequencies.

    This matches the metric from the manuscript: aggregate how frequently each feature
    appears across all prototypical parts, then compute the Gini coefficient.

    Args:
        masks: (n_prototypes, n_features) binary tensor

    Returns:
        Gini coefficient (0 = uniform usage, 1 = single feature dominates)
    """
    # Aggregate feature frequencies across prototypes
    feature_freq = masks.sum(dim=0).float()  # (n_features,)
    sorted_freq, _ = torch.sort(feature_freq)

    D = len(sorted_freq)
    if sorted_freq.sum() == 0:
        return 0.0

    indices = torch.arange(1, D + 1, dtype=torch.float32)
    gini = ((2 * indices - D - 1) * sorted_freq).sum() / (D * sorted_freq.sum())
    return gini.item()


def compute_rank_diversity(
        model,
        dataloader: DataLoader,
        device: torch.device,
        model_type: str = "p2tab",
) -> float:
    """Compute Rank Diversity: mean std of relative prototype ranks across instances.

    For each instance, rank all prototypes by their distance to the nearest patch.
    Then compute the std of relative ranks for each prototype across all instances,
    and average across prototypes.

    Args:
        model: trained model (P2Tab or Medic)
        dataloader: evaluation data loader
        device: torch device
        model_type: "p2tab" or "medic"

    Returns:
        Rank Diversity score (higher = more instance-specific prototype usage)
    """
    model.eval()
    all_ranks = []

    with torch.no_grad():
        if model_type == "p2tab":
            for x, y, *_ in dataloader:
                x = x.to(device).float()
                patches_embeddings = model.embeddings(x)  # (B, P, E)
                prototype_dist, _ = model.prototypes(patches_embeddings)  # (B, R)
                # Rank: 1 = closest, R = farthest
                ranks = torch.argsort(torch.argsort(prototype_dist, dim=1), dim=1) + 1
                all_ranks.append(ranks.cpu())
        elif model_type == "medic":
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                # MEDIC's forward_distances returns (min_distances, indices)
                min_distances, _ = model.forward_distances(x_batch)  # (B, n_prototypes)
                ranks = torch.argsort(torch.argsort(min_distances, dim=1), dim=1) + 1
                all_ranks.append(ranks.cpu())

    all_ranks = torch.cat(all_ranks, dim=0).float()  # (N, R)
    R = all_ranks.shape[1]

    # Normalize to relative ranks
    relative_ranks = all_ranks / R

    # Compute std per prototype, then average
    rank_std_per_proto = relative_ranks.std(dim=0)  # (R,)
    rank_diversity = rank_std_per_proto.mean().item()

    return rank_diversity


def compute_jaccard_overlap(masks: torch.Tensor) -> dict:
    """Compute pairwise Jaccard similarity across prototype masks.

    Args:
        masks: (n_prototypes, n_features) binary tensor

    Returns:
        dict with mean, std, min, max Jaccard similarity
    """
    n = masks.shape[0]
    if n < 2:
        return {"mean_jaccard": 0.0, "std_jaccard": 0.0, "min_jaccard": 0.0, "max_jaccard": 0.0}

    masks_bool = masks.bool()
    jaccards = []

    for i in range(n):
        for j in range(i + 1, n):
            intersection = (masks_bool[i] & masks_bool[j]).sum().float()
            union = (masks_bool[i] | masks_bool[j]).sum().float()
            if union > 0:
                jaccards.append((intersection / union).item())
            else:
                jaccards.append(0.0)

    jaccards = np.array(jaccards)
    return {
        "mean_jaccard": jaccards.mean(),
        "std_jaccard": jaccards.std(),
        "min_jaccard": jaccards.min(),
        "max_jaccard": jaccards.max(),
    }


def summarize_purity(purity_records: list[dict]) -> dict:
    """Summarize purity records into mean values per k."""
    df = pd.DataFrame(purity_records)
    purity_cols = [c for c in df.columns if c.startswith("purity_")]
    summary = {}
    for col in purity_cols:
        summary[f"mean_{col}"] = df[col].mean()
        summary[f"std_{col}"] = df[col].std()
    return summary


# =============================================================================
# Train MEDIC fresh for comparison
# =============================================================================

def fetch_best_medic_config(dataset_name: str) -> dict:
    """Fetch the best MEDIC hyperparameters from W&B."""
    import wandb

    api = wandb.Api()

    runs = api.runs(
        path="jacek-karolczak/P2Tab",
        filters={
            "config.architecture": "medic",
            "config.data.name": dataset_name,
            "tags": {"$in": ["hyperparameter_tuning", "medic"]},
            "summary_metrics.eval_balanced_accuracy": {"$ne": None},
        },
        order="-summary_metrics.eval_balanced_accuracy",
        per_page=1,
    )

    if len(runs) == 0:
        print(f"  [WARN] No MEDIC runs found in W&B for '{dataset_name}'. Using defaults.")
        return {}

    best_run = runs[0]
    config = best_run.config
    print(f"  Best MEDIC run: {best_run.name}")
    print(f"  Balanced accuracy: {best_run.summary.get('eval_balanced_accuracy', 'N/A'):.4f}")

    return {
        "n_bins": config.get("model", {}).get("n_bins", 5),
        "n_prototypes": config.get("model", {}).get("n_prototypes", 32),
        "n_patches": config.get("model", {}).get("n_patches", 64),
        "hidden_dim": config.get("model", {}).get("hidden_dim", 8),
        "learning_rate": config.get("trainer", {}).get("learning_rate", 0.005),
        "batch_size": config.get("trainer", {}).get("batch_size", 64),
        "penalty_l1": config.get("trainer", {}).get("penalty_l1", 0.01),
        "penalty_diversity": config.get("trainer", {}).get("penalty_diversity", 0.02),
        "epochs_stage_1": config.get("trainer", {}).get("epochs_stage_1", 40),
        "epochs_stage_2": config.get("trainer", {}).get("epochs_stage_2", 30),
        "epochs_stage_3": config.get("trainer", {}).get("epochs_stage_3", 30),
    }


def fetch_best_medic_run(
        dataset_name: str,
        device: torch.device,
) -> tuple[Medic, list[dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch the best MEDIC model from W&B, load state dict, and return model + data.

    Downloads model_state_dict.pt from the best run.
    Reconstructs the model architecture from run config, loads weights,
    and re-runs set_real_prototypes on the training data.
    """
    import tempfile
    import wandb
    from pathlib import Path

    api = wandb.Api()

    runs = api.runs(
        path="jacek-karolczak/P2Tab",
        filters={
            "config.architecture": "medic",
            "config.data.name": dataset_name,
            "tags": {"$in": ["hyperparameter_tuning", "medic"]},
            "summary_metrics.eval_balanced_accuracy": {"$ne": None},
        },
        order="-summary_metrics.eval_balanced_accuracy",
        per_page=1,
    )

    if len(runs) == 0:
        raise ValueError(f"No MEDIC runs found in W&B for '{dataset_name}'.")

    best_run = runs[0]
    config = best_run.config
    print(f"  Best MEDIC run: {best_run.name}")
    print(f"  Balanced accuracy: {best_run.summary.get('eval_balanced_accuracy', 'N/A'):.4f}")

    # Extract model config
    model_cfg = config.get("model", {})
    n_bins = model_cfg.get("n_bins", 5)
    n_prototypes = model_cfg.get("n_prototypes", 32)
    n_patches = model_cfg.get("n_patches", 64)
    hidden_dim = model_cfg.get("hidden_dim", 8)
    definitions = model_cfg.get("definitions", None)

    # Load raw data to get eval set and regenerate definitions if needed
    x_train, y_train, x_eval, y_eval, x_test, y_test, feat_names = load_raw_data(dataset_name)
    n_classes = len(np.unique(y_train))

    if definitions is None:
        definitions = generate_definitions(x_train, feat_names, n_bins=n_bins)

    # Scale data
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
    y_eval_t = torch.tensor(y_eval, dtype=torch.long)

    scaler = MedicScaler(definitions=definitions)
    x_train_scaled = scaler.fit_transform(x_train_t.clone())
    x_eval_scaled = scaler.transform(x_eval_t.clone())

    # Build model architecture
    model = Medic(
        definitions=definitions,
        n_classes=n_classes,
        n_patches=n_patches,
        n_prototypes=n_prototypes,
        hidden_dim=hidden_dim,
    ).to(device)

    # Download and load state dict
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            f = best_run.file("files/model_state_dict.pt")
            f.download(root=tmp_dir, replace=True)
            state_dict_path = Path(tmp_dir) / "files" / "model_state_dict.pt"
        except Exception:
            f = best_run.file("model_state_dict.pt")
            f.download(root=tmp_dir, replace=True)
            state_dict_path = Path(tmp_dir) / "model_state_dict.pt"

        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()

    # Re-run set_real_prototypes to populate model.prototypical_parts
    model.hard_binning = True
    model.set_real_prototypes(x_train_scaled.to(device))
    model.hard_parts = True

    return model, definitions, x_train_scaled, y_train_t, x_eval_scaled, y_eval_t


def train_medic_for_comparison(
        dataset_name: TNamedData,
        device: torch.device,
        n_bins: int = 5,
        n_prototypes: int = 32,
        n_patches: int | None = None,
        hidden_dim: int = 8,
        learning_rate: float = 0.005,
        batch_size: int = 64,
        penalty_l1: float = 0.01,
        penalty_diversity: float = 0.02,
        epochs_stage_1: int = 40,
        epochs_stage_2: int = 30,
        epochs_stage_3: int = 30,
) -> tuple[Medic, list[dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train MEDIC from scratch and return model + data."""
    set_seed()

    x_train, y_train, x_eval, y_eval, x_test, y_test, feature_names = load_raw_data(dataset_name)
    n_classes = len(np.unique(y_train))

    definitions = generate_definitions(x_train, feature_names, n_bins=n_bins)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
    y_eval_t = torch.tensor(y_eval, dtype=torch.long)

    scaler = MedicScaler(definitions=definitions)
    x_train_scaled = scaler.fit_transform(x_train_t.clone())
    x_eval_scaled = scaler.transform(x_eval_t.clone())

    class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if n_patches is None:
        n_patches = n_prototypes * 2

    train_loader = DataLoader(
        TensorDataset(x_train_scaled, y_train_t),
        batch_size=batch_size, shuffle=True
    )

    model = Medic(
        definitions=definitions,
        n_classes=n_classes,
        n_patches=n_patches,
        n_prototypes=n_prototypes,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

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

    return model, definitions, x_train_scaled, y_train_t, x_eval_scaled, y_eval_t


# =============================================================================
# Single-dataset evaluation
# =============================================================================

def evaluate_dataset(
        dataset_name: TNamedData,
        device: torch.device,
        p2tab_tag: str,
) -> dict | None:
    """Run explanation comparison for a single dataset. Returns results dict or None on failure."""
    print(f"\n{'=' * 60}")
    print(f"  Processing: {dataset_name}")
    print(f"{'=' * 60}")

    try:
        # =====================================================================
        # Load / train P2Tab
        # =====================================================================
        print("\n[P2Tab] Loading model...")
        if p2tab_tag == "train":
            data_container, p2tab_config, trainer_config = read_data_and_configs(dataset_name)
            trainer_config.device = str(device)
            trainer_config.wandb_config.active = False
            trainer_config.verbose = False

            p2tab = P2Tab(p2tab_config)
            trainer = P2TabTrainer(data_container, p2tab, trainer_config)
            trainer.train(wandb_finish=True)
        else:
            print(f"  Fetching best run with tag '{p2tab_tag}' from W&B...")
            _, data_container, p2tab_config, trainer_config, p2tab = fetch_best_run(
                dataset_name, [p2tab_tag], load_model=True
            )

        p2tab = p2tab.to(device)
        p2tab.eval()

        if p2tab_tag == "train":
            train_dataset = SimpleDataset(data_container.x_train, data_container.y_train)
            train_loader = DataLoader(train_dataset, batch_size=trainer_config.batch_size, shuffle=False)
            p2tab.set_real_prototypes(train_loader)

        # P2Tab metrics
        p2tab_masks = get_p2tab_masks(p2tab)
        p2tab_sparsity = compute_sparsity(p2tab_masks)
        p2tab_jaccard = compute_jaccard_overlap(p2tab_masks)
        p2tab_gini = compute_feature_focus_gini(p2tab_masks)

        _, _, test_dataset = data_container.to_simple_datasets()
        test_loader = DataLoader(test_dataset, batch_size=trainer_config.batch_size, shuffle=False)
        p2tab_purity = compute_p2tab_purity(p2tab, test_loader, device)
        p2tab_purity_summary = summarize_purity(p2tab_purity)
        p2tab_rank_diversity = compute_rank_diversity(p2tab, test_loader, device, model_type="p2tab")

        # =====================================================================
        # Load MEDIC from W&B
        # =====================================================================
        print("\n[MEDIC] Fetching best model from W&B...")
        medic_model, definitions, x_train_scaled, y_train_t, x_eval_scaled, y_eval_t = \
            fetch_best_medic_run(dataset_name, device)

        # MEDIC metrics
        p2tab_patch_len = p2tab.config.patching.patch_len
        medic_masks = get_medic_feature_masks(medic_model, definitions, top_k=p2tab_patch_len)
        medic_masks_relative = get_medic_feature_masks(medic_model, definitions, top_k=None)

        medic_sparsity = compute_sparsity(medic_masks)
        medic_sparsity_rel = compute_sparsity(medic_masks_relative)
        medic_jaccard = compute_jaccard_overlap(medic_masks)
        medic_jaccard_rel = compute_jaccard_overlap(medic_masks_relative)
        medic_gini = compute_feature_focus_gini(medic_masks_relative)

        medic_purity = compute_medic_purity(medic_model, x_eval_scaled, y_eval_t, device)
        medic_purity_summary = summarize_purity(medic_purity)

        # Rank diversity for MEDIC
        medic_eval_loader = DataLoader(
            TensorDataset(x_eval_scaled, y_eval_t),
            batch_size=256, shuffle=False
        )
        medic_rank_diversity = compute_rank_diversity(medic_model, medic_eval_loader, device, model_type="medic")

        return {
            "dataset": dataset_name,
            "p2tab_patch_len": p2tab_patch_len,
            "p2tab_sparsity": p2tab_sparsity,
            "p2tab_jaccard": p2tab_jaccard,
            "p2tab_purity": p2tab_purity_summary,
            "p2tab_gini": p2tab_gini,
            "p2tab_rank_diversity": p2tab_rank_diversity,
            "medic_sparsity": medic_sparsity,
            "medic_sparsity_rel": medic_sparsity_rel,
            "medic_jaccard": medic_jaccard,
            "medic_jaccard_rel": medic_jaccard_rel,
            "medic_purity": medic_purity_summary,
            "medic_gini": medic_gini,
            "medic_rank_diversity": medic_rank_diversity,
        }

    except Exception as e:
        print(f"  [ERROR] Failed on {dataset_name}: {e}")
        return None


def print_aggregate_table(all_results: list[dict]) -> None:
    """Print a summary table aggregating results across datasets."""
    print(f"\n{'#' * 70}")
    print(f"  AGGREGATE RESULTS ACROSS {len(all_results)} DATASETS")
    print(f"{'#' * 70}")

    # --- Sparsity table ---
    print(f"\n{'=' * 70}")
    print("Table 1: Prototype Sparsity (mean active features / sparsity ratio)")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<18} {'N feat':>6} {'P2Tab':>10} {'MEDIC(top-k)':>12} {'MEDIC(rel)':>12}")
    print(f"{'-' * 58}")
    for r in all_results:
        print(f"{r['dataset']:<18} {r['p2tab_sparsity']['n_features_total']:>6} "
              f"{r['p2tab_sparsity']['mean_active_features']:>10.1f} "
              f"{r['medic_sparsity']['mean_active_features']:>12.1f} "
              f"{r['medic_sparsity_rel']['mean_active_features']:>12.1f}")

    # Averages
    avg_p2tab_spar = np.mean([r["p2tab_sparsity"]["mean_sparsity_ratio"] for r in all_results])
    avg_medic_spar = np.mean([r["medic_sparsity"]["mean_sparsity_ratio"] for r in all_results])
    avg_medic_spar_rel = np.mean([r["medic_sparsity_rel"]["mean_sparsity_ratio"] for r in all_results])
    print(f"{'-' * 58}")
    print(f"{'Avg sparsity ratio':<18} {'':>6} {avg_p2tab_spar:>10.3f} {avg_medic_spar:>12.3f} {avg_medic_spar_rel:>12.3f}")

    # --- Jaccard table ---
    print(f"\n{'=' * 70}")
    print("Table 2: Feature Overlap (mean pairwise Jaccard similarity)")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<18} {'P2Tab':>10} {'MEDIC(top-k)':>12} {'MEDIC(rel)':>12}")
    print(f"{'-' * 52}")
    for r in all_results:
        print(f"{r['dataset']:<18} "
              f"{r['p2tab_jaccard']['mean_jaccard']:>10.3f} "
              f"{r['medic_jaccard']['mean_jaccard']:>12.3f} "
              f"{r['medic_jaccard_rel']['mean_jaccard']:>12.3f}")

    avg_p2tab_jacc = np.mean([r["p2tab_jaccard"]["mean_jaccard"] for r in all_results])
    avg_medic_jacc = np.mean([r["medic_jaccard"]["mean_jaccard"] for r in all_results])
    avg_medic_jacc_rel = np.mean([r["medic_jaccard_rel"]["mean_jaccard"] for r in all_results])
    print(f"{'-' * 52}")
    print(f"{'Average':<18} {avg_p2tab_jacc:>10.3f} {avg_medic_jacc:>12.3f} {avg_medic_jacc_rel:>12.3f}")

    # --- Purity table ---
    print(f"\n{'=' * 70}")
    print("Table 3: Top-k Purity (mean across prototypes)")
    print(f"{'=' * 70}")

    # Find common k values
    k_vals = set()
    for r in all_results:
        for key in r["p2tab_purity"]:
            if key.startswith("mean_purity_"):
                k_vals.add(key.replace("mean_purity_", ""))
    k_vals = sorted(k_vals, key=lambda x: int(x))

    for k_str in k_vals:
        print(f"\n  k = {k_str}:")
        print(f"  {'Dataset':<18} {'P2Tab':>10} {'MEDIC':>10}")
        print(f"  {'-' * 38}")
        p2tab_vals = []
        medic_vals = []
        for r in all_results:
            p_val = r["p2tab_purity"].get(f"mean_purity_{k_str}", float("nan"))
            m_val = r["medic_purity"].get(f"mean_purity_{k_str}", float("nan"))
            p2tab_vals.append(p_val)
            medic_vals.append(m_val)
            print(f"  {r['dataset']:<18} {p_val:>10.3f} {m_val:>10.3f}")
        print(f"  {'-' * 38}")
        print(f"  {'Average':<18} {np.nanmean(p2tab_vals):>10.3f} {np.nanmean(medic_vals):>10.3f}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY: P2Tab vs MEDIC (wins / ties / losses)")
    print(f"{'=' * 70}")

    jaccard_wins = sum(1 for r in all_results if r["p2tab_jaccard"]["mean_jaccard"] < r["medic_jaccard"]["mean_jaccard"])
    jaccard_ties = sum(
        1 for r in all_results if abs(r["p2tab_jaccard"]["mean_jaccard"] - r["medic_jaccard"]["mean_jaccard"]) < 0.01)
    jaccard_losses = len(all_results) - jaccard_wins - jaccard_ties

    print(f"  Feature diversity (lower Jaccard = better): {jaccard_wins}W / {jaccard_ties}T / {jaccard_losses}L")

    if k_vals:
        k_mid = k_vals[len(k_vals) // 2]
        purity_wins = sum(1 for r in all_results
                          if
                          r["p2tab_purity"].get(f"mean_purity_{k_mid}", 0) > r["medic_purity"].get(f"mean_purity_{k_mid}", 0))
        purity_losses = len(all_results) - purity_wins
        print(f"  Purity @ k={k_mid} (higher = better):        {purity_wins}W / 0T / {purity_losses}L")


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


def generate_latex_tables(all_results: list[dict]) -> str:
    """Generate LaTeX tables for the explanation comparison."""

    lines = []

    # --- Table 1: Rank Diversity + Sparsity Ratio + Jaccard ---
    lines.append(r"\begin{table}[tbp]")
    lines.append(r"\caption{Explanation quality comparison between \name and MEDIC. "
                 r"Rank Diversity measures instance-specific prototype usage (higher = better). "
                 r"Sparsity Ratio indicates the fraction of inactive features per prototypical part (higher = more concise explanations). "
                 r"Mean Jaccard measures pairwise feature overlap across prototypes (lower = more diverse). "
                 r"Best values per dataset are in bold.}")
    lines.append(r"\label{tab:explanation_quality}")
    lines.append(r"\centering")
    lines.append(r"    \begin{tabular}{lcccccc}")
    lines.append(r"        \toprule")
    lines.append(
        r"        & \multicolumn{2}{c}{Rank Diversity $\uparrow$} & \multicolumn{2}{c}{Sparsity Ratio $\uparrow$} & \multicolumn{2}{c}{Mean Jaccard $\downarrow$} \\")
    lines.append(r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"        Dataset & \name & MEDIC & \name & MEDIC & \name & MEDIC \\")
    lines.append(r"        \midrule")

    for r in all_results:
        ds = DATASET_DISPLAY_NAMES.get(r["dataset"], r["dataset"])
        p2_rd = r["p2tab_rank_diversity"]
        m_rd = r["medic_rank_diversity"]
        p2_spar = r["p2tab_sparsity"]["mean_sparsity_ratio"]
        m_spar = r["medic_sparsity_rel"]["mean_sparsity_ratio"]
        p2_jacc = r["p2tab_jaccard"]["mean_jaccard"]
        m_jacc = r["medic_jaccard"]["mean_jaccard"]

        lines.append(
            f"        {ds} & {p2_rd:.4f} & {m_rd:.4f} & {p2_spar:.4f} & {m_spar:.4f} & {p2_jacc:.4f} & {m_jacc:.4f} \\\\")

    # Average row
    avg_p2_rd = np.mean([r["p2tab_rank_diversity"] for r in all_results])
    avg_m_rd = np.mean([r["medic_rank_diversity"] for r in all_results])
    avg_p2_spar = np.mean([r["p2tab_sparsity"]["mean_sparsity_ratio"] for r in all_results])
    avg_m_spar = np.mean([r["medic_sparsity_rel"]["mean_sparsity_ratio"] for r in all_results])
    avg_p2_jacc = np.mean([r["p2tab_jaccard"]["mean_jaccard"] for r in all_results])
    avg_m_jacc = np.mean([r["medic_jaccard"]["mean_jaccard"] for r in all_results])

    lines.append(r"        \midrule")
    lines.append(
        f"        Average & {avg_p2_rd:.4f} & {avg_m_rd:.4f} & {avg_p2_spar:.4f} & {avg_m_spar:.4f} & {avg_p2_jacc:.4f} & {avg_m_jacc:.4f} \\\\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    lines.append("")

    # --- Table 2: Top-k Purity ---
    # Find common k values
    k_vals = set()
    for r in all_results:
        for key in r["p2tab_purity"]:
            if key.startswith("mean_purity_"):
                k_vals.add(key.replace("mean_purity_", ""))
    k_vals = sorted(k_vals, key=lambda x: int(x))

    # Pick representative k values (3, 7, 11) to keep table compact
    representative_k = [k for k in k_vals if k in ["3", "5", "7", "9", "11"]]
    if not representative_k:
        representative_k = k_vals[:3]

    n_k = len(representative_k)
    col_spec = "l" + "cc" * n_k

    lines.append(r"\begin{table}[tbp]")
    lines.append(r"\caption{Top-$k$ nearest neighbor purity comparison. "
                 r"Purity measures the fraction of a prototype's $k$-nearest patch embeddings "
                 r"that share its assigned class (higher = more class-coherent prototypes). Best values in bold.}")
    lines.append(r"\label{tab:purity_comparison}")
    lines.append(r"\centering")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")

    # Header
    header_cols = " & ".join([f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in representative_k])
    lines.append(f"        & {header_cols} \\\\")

    cmidrule_parts = " ".join([f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_k)])
    lines.append(f"        {cmidrule_parts}")

    subheader = " & ".join(["\\name & MEDIC"] * n_k)
    lines.append(f"        Dataset & {subheader} \\\\")
    lines.append(r"        \midrule")

    for r in all_results:
        ds = DATASET_DISPLAY_NAMES.get(r["dataset"], r["dataset"])
        cols = []
        for k_str in representative_k:
            p2_val = r["p2tab_purity"].get(f"mean_purity_{k_str}", float("nan"))
            m_val = r["medic_purity"].get(f"mean_purity_{k_str}", float("nan"))
            cols.append(f"{p2_val:.3f} & {m_val:.3f}")

        lines.append(f"        {ds} & {' & '.join(cols)} \\\\")

    # Average row
    lines.append(r"        \midrule")
    avg_cols = []
    for k_str in representative_k:
        avg_p2 = np.nanmean([r["p2tab_purity"].get(f"mean_purity_{k_str}", float("nan")) for r in all_results])
        avg_m = np.nanmean([r["medic_purity"].get(f"mean_purity_{k_str}", float("nan")) for r in all_results])
        avg_cols.append(f"{avg_p2:.3f} & {avg_m:.3f}")

    lines.append(f"        Average & {' & '.join(avg_cols)} \\\\")
    lines.append(r"        \bottomrule")
    lines.append(f"    \\end{{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

@click.command()
@click.argument("dataset_name", type=click.Choice(list(TNamedData.__args__) + ["all"]))
@click.option("--device", type=str, default="cpu")
@click.option("--p2tab-tag", type=str, default="hyperparameter_tuning",
              help="W&B tag to fetch best P2Tab run. Set to 'train' to train fresh.")
def main(dataset_name: str, device: str, p2tab_tag: str) -> None:
    set_seed()
    device = torch.device(device)

    if dataset_name == "all":
        datasets = list(TNamedData.__args__)
    else:
        datasets = [dataset_name]

    all_results = []
    for ds in datasets:
        result = evaluate_dataset(ds, device, p2tab_tag)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n[ERROR] No datasets completed successfully.")
        return

    # Print per-dataset results
    for r in all_results:
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {r['dataset']}")
        print(f"{'=' * 60}")

        print(f"\n--- Prototype Sparsity ---")
        print(f"{'Metric':<30} {'P2Tab':>12} {'MEDIC(top-k)':>12} {'MEDIC(rel)':>12}")
        print(f"{'-' * 66}")
        print(
            f"{'Total features':<30} {r['p2tab_sparsity']['n_features_total']:>12} {r['medic_sparsity']['n_features_total']:>12} {r['medic_sparsity_rel']['n_features_total']:>12}")
        print(
            f"{'Mean active features':<30} {r['p2tab_sparsity']['mean_active_features']:>12.2f} {r['medic_sparsity']['mean_active_features']:>12.2f} {r['medic_sparsity_rel']['mean_active_features']:>12.2f}")
        print(
            f"{'Sparsity ratio':<30} {r['p2tab_sparsity']['mean_sparsity_ratio']:>12.3f} {r['medic_sparsity']['mean_sparsity_ratio']:>12.3f} {r['medic_sparsity_rel']['mean_sparsity_ratio']:>12.3f}")

        print(f"\n--- Feature Overlap (Jaccard) ---")
        print(f"{'Metric':<30} {'P2Tab':>12} {'MEDIC(top-k)':>12} {'MEDIC(rel)':>12}")
        print(f"{'-' * 66}")
        print(
            f"{'Mean Jaccard':<30} {r['p2tab_jaccard']['mean_jaccard']:>12.3f} {r['medic_jaccard']['mean_jaccard']:>12.3f} {r['medic_jaccard_rel']['mean_jaccard']:>12.3f}")

        print(f"\n--- Top-k Purity ---")
        all_k = sorted(set(
            [k.replace("mean_purity_", "") for k in r["p2tab_purity"] if k.startswith("mean_")]
        ))
        print(f"{'k':<10} {'P2Tab':>12} {'MEDIC':>12}")
        print(f"{'-' * 34}")
        for k_str in all_k:
            p2tab_mean = r["p2tab_purity"].get(f"mean_purity_{k_str}", float("nan"))
            medic_mean = r["medic_purity"].get(f"mean_purity_{k_str}", float("nan"))
            print(f"{k_str:<10} {p2tab_mean:>12.3f} {medic_mean:>12.3f}")

    # Print aggregate if multiple datasets
    if len(all_results) > 1:
        print_aggregate_table(all_results)

    # Generate LaTeX tables
    latex_tables = generate_latex_tables(all_results)
    output_path = Path("results") / "results_explanation_comparison.tex"
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w") as f:
        f.write(latex_tables)

    print(f"\n{'=' * 60}")
    print(f"LaTeX tables saved to: {output_path}")
    print(f"{'=' * 60}")
    print(latex_tables)


if __name__ == "__main__":
    main()
