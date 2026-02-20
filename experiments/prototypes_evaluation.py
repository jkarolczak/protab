import platform

import click
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from protab.data.named_data import TNamedData
from protab.training.config import fetch_best_run

MAX_PATCHES = 5000


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--device", type=str, default="cpu")
def main(dataset_name: TNamedData, device: str) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 800,
        "savefig.dpi": 800,
    })

    best_run, data_container, model_config, trainer_config, protab = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=True
    )

    protab = protab.to(device)
    protab.eval()

    _, _, test_dataset = data_container.to_simple_datasets()
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False
    )

    patch_embeddings_list = []
    labels_list = []
    ranks_list = []

    with torch.no_grad():
        for x, y, *_ in dataloader:
            x = x.to(device).to(torch.float32)
            y = y.to(device)

            logits, patches_embeddings = protab(x, return_embeddings=True)
            prototype_dist, patches_idcs = protab.prototypes(patches_embeddings)

            ranks = torch.argsort(torch.argsort(prototype_dist, dim=1), dim=1) + 1
            ranks_list.append(ranks.cpu())

            if y.dim() > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)

            B, P_cnt, E = patches_embeddings.shape
            active_mask = torch.zeros((B, P_cnt), dtype=torch.bool, device=device)
            active_mask.scatter_(1, patches_idcs, True)

            active_patches = patches_embeddings[active_mask]
            active_labels = y.unsqueeze(1).expand(B, P_cnt)[active_mask]

            patch_embeddings_list.append(active_patches.cpu())
            labels_list.append(active_labels.cpu())

    all_ranks = torch.cat(ranks_list, dim=0)

    best_ranks_per_proto = torch.min(all_ranks, dim=0).values

    best_rank_distribution = []
    for r in range(1, all_ranks.shape[1] + 1):
        count = (best_ranks_per_proto == r).sum().item()
        best_rank_distribution.append({"rank": r, "prototypes_count": count})

    w_cls = protab.classifier.network[0].weight.data.cpu()
    if w_cls.shape[0] == 1:
        assigned_classes = (w_cls[0] > 0).long()
    else:
        assigned_classes = torch.argmax(w_cls, dim=0)

    all_patches_flat = torch.cat(patch_embeddings_list, dim=0)
    all_labels_flat = torch.cat(labels_list, dim=0)

    proto_emb = F.normalize(protab.prototypes.prototypes.data, p=2, dim=-1).cpu()

    dists = torch.cdist(proto_emb, all_patches_flat)

    k_vals = [3, 5, 7, 9, 11, 13, 15]
    purity_records = []

    valid_k_vals = [k for k in k_vals if k <= all_patches_flat.shape[0]]

    for j in range(all_ranks.shape[1]):
        proto_dists = dists[j]
        top_indices = torch.topk(proto_dists, k=max(valid_k_vals), largest=False).indices
        top_labels = all_labels_flat[top_indices]

        assigned_c = assigned_classes[j].item()
        record = {"prototype_idx": j, "assigned_class": assigned_c}
        for k in valid_k_vals:
            purity = (top_labels[:k] == assigned_c).float().mean().item()
            record[f"purity_{k}"] = purity
        purity_records.append(record)

    if all_patches_flat.shape[0] > MAX_PATCHES:
        indices = torch.randperm(all_patches_flat.shape[0])[:MAX_PATCHES]
        sub_patches = all_patches_flat[indices]
        sub_labels = all_labels_flat[indices]
    else:
        sub_patches = all_patches_flat
        sub_labels = all_labels_flat

    combined_emb = torch.cat([sub_patches, proto_emb], dim=0).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(combined_emb)

    patch_tsne = tsne_emb[:sub_patches.shape[0]]
    proto_tsne = tsne_emb[sub_patches.shape[0]:]

    tsne_records = []
    for i in range(patch_tsne.shape[0]):
        tsne_records.append(
            {"type": "patch", "x": float(patch_tsne[i, 0]), "y": float(patch_tsne[i, 1]), "class": int(sub_labels[i].item())})
    for i in range(proto_tsne.shape[0]):
        tsne_records.append({"type": "prototype", "x": float(proto_tsne[i, 0]), "y": float(proto_tsne[i, 1]),
                             "class": int(assigned_classes[i].item())})

    df_rank = pd.DataFrame(best_rank_distribution)
    fig_hist, ax_hist = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    ax_hist.bar(df_rank["rank"], df_rank["prototypes_count"], color="#666666", edgecolor="black")
    ax_hist.set_xlabel("Best Rank Achieved")
    ax_hist.set_ylabel("Number of Prototypes")
    ax_hist.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    df_tsne = pd.DataFrame(tsne_records)
    fig_tsne, ax_tsne = plt.subplots(figsize=(4, 4), dpi=800)

    cmap = cmc.batlowS

    classes = sorted(df_tsne["class"].unique())

    # Sample evenly across the colormap to maximize contrast for categorical data,
    # rather than picking the first adjacent (and therefore visually similar) colors.
    colors = cmap(np.linspace(0, 1, len(classes)))

    patches = df_tsne[df_tsne["type"] == "patch"]
    for idx, cls in enumerate(classes):
        subset = patches[patches["class"] == cls]
        ax_tsne.scatter(subset["x"], subset["y"], s=10, alpha=0.5, label=f"class {int(cls)}", color=colors[idx])

    protos = df_tsne[df_tsne["type"] == "prototype"]
    for idx, cls in enumerate(classes):
        subset = protos[protos["class"] == cls]
        ax_tsne.scatter(subset["x"], subset["y"], s=150, marker="*", edgecolor="black", linewidth=1.0, color=colors[idx])

    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])

    handles, labels = ax_tsne.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    shape_handles = [
        Line2D([0], [0], marker="o", color="w", label="Patch", markerfacecolor="gray", markersize=4, alpha=0.5),
        Line2D([0], [0], marker="*", color="w", label="Prototype", markerfacecolor="gray", markeredgecolor="black",
               markersize=8)
    ]

    all_handles = shape_handles + list(by_label.values())
    all_labels = [h.get_label() for h in shape_handles] + list(by_label.keys())

    ax_tsne.legend(all_handles, all_labels, loc="best")
    plt.tight_layout()

    platform_name = platform.node()

    wandb.init(
        project="ProTab",
        entity="jacek-karolczak",
        name=f"{dataset_name}_prototypes_evaluation",
        tags=["prototypes_evaluation"],
        config={
            "architecture": "ProTab",
            "model": model_config.__dict__,
            "trainer": trainer_config.__dict__,
            "data": data_container.config.__dict__,
            "platform": platform_name
        },
    )

    wandb.log({
        "best_rank_distribution": wandb.Table(dataframe=df_rank),
        "top_k_purity": wandb.Table(dataframe=pd.DataFrame(purity_records)),
        "tsne_embeddings": wandb.Table(dataframe=df_tsne),
        "rank_histogram_plot": wandb.Image(fig_hist),
        "tsne_plot": wandb.Image(fig_tsne)
    })

    plt.close(fig_hist)
    plt.close(fig_tsne)
    wandb.finish()


if __name__ == "__main__":
    main()
