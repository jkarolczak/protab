from dataclasses import dataclass

import torch
import torch.nn as nn

from protab.training.reproducibility import set_seed


@dataclass
class PatchingConfig:
    n_features: int
    patch_len: int
    n_patches: int
    append_masks: bool = True
    probabilistic: bool = True
    use_learnable_mask_token: bool = True


class ProbabilisticPatching(nn.Module):
    def __init__(
            self,
            config: PatchingConfig
    ) -> None:
        super().__init__()
        set_seed()
        self.config = config

        self.weights = nn.Parameter(
            torch.randn(self.config.n_patches, self.config.n_features, dtype=torch.float32) * 0.01
        )

        usage_counts = torch.zeros(self.config.n_features, dtype=torch.float32)
        with torch.no_grad():
            for i in range(self.config.n_patches):
                probs = 1.0 / (usage_counts + 0.1)
                selected_indices = torch.multinomial(probs, num_samples=self.config.patch_len, replacement=False)

                self.weights[i, selected_indices] = 1.0
                usage_counts[selected_indices] += 1.0

        if self.config.use_learnable_mask_token:
            signs = torch.randint(0, 2, (self.config.n_features,), dtype=torch.float32) * 2 - 1
            magnitudes = torch.rand(self.config.n_features, dtype=torch.float32) + 2.5
            self.mask_token = nn.Parameter(signs * magnitudes)
        else:
            self.mask_token = None

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_features = x.shape
        assert n_features == self.config.n_features, \
            f"Input features ({n_features}) do not match config ({self.config.n_features})"

        logits = self.weights

        _, topk_indices = torch.topk(logits, self.config.patch_len, dim=-1)
        mask_hard = torch.zeros_like(logits)
        mask_hard.scatter_(-1, topk_indices, 1.0)

        mask_soft = torch.sigmoid(logits)

        masks = mask_hard.detach() - mask_soft.detach() + mask_soft

        masks = masks.unsqueeze(0).expand(batch_size, -1, -1)

        patches = x.unsqueeze(1) * masks

        if self.config.use_learnable_mask_token and self.mask_token is not None:
            token_part = self.mask_token.view(1, 1, -1) * (1 - masks)
            patches = patches + token_part

        if self.config.append_masks:
            patches = torch.cat([patches, masks], dim=-1)

        return patches
