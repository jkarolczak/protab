from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompoundLossConfig:
    triplet_margin: float = 1.0
    triplet_p: int = 2
    w_ce: float = 1.0
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


class CompoundLoss(nn.Module):
    def __init__(
            self,
            config: CompoundLossConfig
    ) -> None:
        super().__init__()
        self.config = config

        if self.config.ce_pos_weight is not None:
            ce_pos_weight = torch.tensor(self.config.ce_pos_weight)

        self.cross_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=ce_pos_weight)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.config.triplet_margin, p=self.config.triplet_p)

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
        if logits.max() > 1.0 or logits.min() < 0.0:
            logits = F.softmax(logits, dim=-1)
        ce = self.cross_entropy_loss(logits, targets)
        triplet = self.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        patch_diversity = diversity(patching_weights)
        proto_diversity = diversity(prototypes_weights)

        w_ce = self.config.w_ce * ce
        w_triplet = self.config.w_triplet * triplet
        w_patch_diversity = self.config.w_patch_diversity * patch_diversity
        w_proto_diversity = self.config.w_proto_diversity * proto_diversity

        total_loss = w_ce + w_triplet + w_patch_diversity + w_proto_diversity
        return total_loss, ce, triplet, patch_diversity, proto_diversity

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
