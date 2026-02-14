from dataclasses import dataclass
from typing import (Literal,
                    TypeAlias)

import torch
import torch.nn as nn
import torch.nn.functional as F

TClassificationLoss: TypeAlias = Literal["cross_entropy", "focal"]


@dataclass
class CompoundLossConfig:
    classification_loss: TClassificationLoss = "focal"
    triplet_margin: float = 1.0
    triplet_p: int = 2
    w_cls: float = 1.0
    w_triplet: float = 1.0
    w_patch_diversity: float = 1.0
    w_proto_diversity: float = 1.0
    ce_pos_weight: float | list[float] | None = None


def diversity(
        weights: torch.Tensor
) -> torch.Tensor:
    n = weights.shape[0]
    p_norm = F.normalize(weights, p=2, dim=1)
    gram_matrix = torch.matmul(p_norm, p_norm.t())
    identity = torch.eye(n, device=weights.device)
    loss = torch.norm(gram_matrix - identity, p='fro') / n
    return loss


class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0,
                 reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        match self.reduction:
            case "mean":
                return focal_loss.mean()
            case "sum":
                return focal_loss.sum()
            case _:
                return focal_loss


class CompoundLoss(nn.Module):
    def __init__(
            self,
            config: CompoundLossConfig
    ) -> None:
        super().__init__()
        self.config = config

        if self.config.ce_pos_weight is None:
            weight = None
        else:
            weight = torch.tensor(self.config.ce_pos_weight)

        if self.config.classification_loss == "cross_entropy":
            self.classification_loss = nn.CrossEntropyLoss(weight=weight)
        elif self.config.classification_loss == "focal":
            self.classification_loss = MultiClassFocalLoss(weight=weight)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.config.triplet_margin, p=self.config.triplet_p)

    def to(
            self,
            device: str | torch.device
    ) -> 'CompoundLoss':
        self.classification_loss.to(device)
        self.triplet_margin_loss.to(device)

        return self

    def forward_partial(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            anchor_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor,
            negative_embeddings: torch.Tensor,
            patching_weights: torch.Tensor,
            prototypes_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if targets.ndim > 1 and targets.size(1) > 1:
            target_indices = targets.argmax(dim=1)
        else:
            target_indices = targets.view(-1).long()

        cls_loss = self.classification_loss(logits, target_indices)
        triplet = self.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        patch_diversity = diversity(patching_weights)
        proto_diversity = diversity(prototypes_weights)

        w_cls = self.config.w_cls * cls_loss
        w_triplet = self.config.w_triplet * triplet
        w_patch_diversity = self.config.w_patch_diversity * patch_diversity
        w_proto_diversity = self.config.w_proto_diversity * proto_diversity

        total_loss = w_cls + w_triplet + w_patch_diversity + w_proto_diversity
        return total_loss, cls_loss, triplet, patch_diversity, proto_diversity

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            anchor_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor,
            negative_embeddings: torch.Tensor,
            patching_weights: torch.Tensor,
            prototypes_weights: torch.Tensor
    ) -> torch.Tensor:
        total_loss, *_ = self.forward_partial(logits, targets, anchor_embeddings, positive_embeddings, negative_embeddings,
                                              patching_weights, prototypes_weights)
        return total_loss
