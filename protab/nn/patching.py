from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchingConfig:
    n_features: int
    patch_len: int
    n_patches: int
    append_masks: bool = True
    probabilistic: bool = True


class ProbabilisticPatching(nn.Module):
    def __init__(
            self,
            config: PatchingConfig
    ) -> None:
        super().__init__()
        self.config = config

        self.weights = nn.Parameter(
            torch.randn(self.config.n_patches, self.config.n_features, dtype=torch.float32)
        )

    def _probabilistic_masks(
            self,
            batch_size: int
    ) -> torch.Tensor:
        """Generates stochastic binary masks by sampling from learned feature weights.
        """

        probs = F.softmax(self.weights, dim=-1)

        probs_expanded = probs.unsqueeze(0).expand(batch_size, -1, -1)
        probs_flat = probs_expanded.reshape(-1, self.config.n_features)
        indices = torch.multinomial(probs_flat, self.config.patch_len, replacement=False)
        masks_flat = torch.zeros_like(probs_flat)
        masks_flat.scatter_(dim=1, index=indices, src=torch.ones_like(masks_flat))
        masks = masks_flat.view(batch_size, self.config.n_patches, self.config.n_features)

        return masks

    def _deterministic_masks(
            self,
            batch_size: int
    ) -> torch.Tensor:
        """Generates deterministic binary masks by selecting top-k features from learned weights.
        """
        topk_indices = self.weights.topk(self.config.patch_len, dim=-1).indices
        mask_template = torch.zeros_like(self.weights)
        mask_template.scatter_(dim=1, index=topk_indices, value=1.0)
        masks = mask_template.unsqueeze(0).expand(batch_size, -1, -1)

        return masks

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_features = x.shape
        assert n_features == self.config.n_features, \
            f"Input features ({n_features}) do not match config ({self.config.n_features})"

        if self.config.probabilistic or self.training:
            masks = self._probabilistic_masks(batch_size)
        else:
            masks = self._deterministic_masks(batch_size)
        masks = masks.to(x.device)

        patches = x.unsqueeze(1) * masks

        if self.config.append_masks:
            patches = torch.cat([patches, masks], dim=-1)

        return patches
