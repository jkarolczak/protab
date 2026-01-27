from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

TDistanceMetric: TypeAlias = Literal["cosine", "dot", "l1", "l2"]


@dataclass
class PrototypeConfig:
    n_prototypes: int
    prototype_dim: int
    distance_metric: TDistanceMetric = "l2"


class Prototypes(nn.Module):
    def __init__(
            self,
            config: PrototypeConfig
    ) -> None:
        super().__init__()
        self.config = config

        self.prototypes = nn.Parameter(
            torch.randn(self.config.n_prototypes, self.config.prototype_dim)
        )

    def compute_distance_matrix(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        match self.config.distance_metric:
            case "cosine":
                x_norm = F.normalize(x, dim=-1)
                p_norm = F.normalize(self.prototypes, dim=-1)
                distances = 1 - torch.matmul(x_norm, p_norm.t())

            case "dot":
                distances = -torch.matmul(x, self.prototypes.t())

            case "l1" | "l2":
                x_expanded = x.unsqueeze(2)
                p_expanded = self.prototypes.unsqueeze(0).unsqueeze(0)

                if self.config.distance_metric == "l1":
                    distances = torch.sum(torch.abs(x_expanded - p_expanded), dim=-1)
                else:
                    distances = torch.sqrt(torch.sum((x_expanded - p_expanded) ** 2, dim=-1) + 1e-8)

            case _:
                raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")

        return distances

    def forward(
            self,
            x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self.compute_distance_matrix(x)

        prototype_dist, patches_idcs = torch.topk(distances, dim=-2, k=1, largest=False)
        prototype_dist = prototype_dist.squeeze(-1)
        patches_idcs = patches_idcs.squeeze(-1)

        return prototype_dist, patches_idcs
