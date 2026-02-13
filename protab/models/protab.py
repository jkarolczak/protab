from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from protab.models.mlp import MLPConfig, MLP
from protab.nn.patching import PatchingConfig, ProbabilisticPatching
from protab.nn.prototypes import (PrototypeConfig,
                                  TDistanceMetric, Prototypes)


@dataclass
class ProTabConfig:
    patching: PatchingConfig
    encoder: MLPConfig
    prototypes: PrototypeConfig
    classifier: MLPConfig


class ProTabConfigFactory:
    @staticmethod
    def build(
            n_features: int,
            n_patches: int,
            patch_len: int,
            n_prototypes: int,
            n_classes: int,
            encoder_hidden_dims: list[int],
            prototype_dim: int,
            classifier_hidden_dims: list[int],
            append_masks: bool = True,
            probabilistic: bool = True,
            prototype_distance_metric: TDistanceMetric = "l2"
    ) -> ProTabConfig:
        patching_config = PatchingConfig(
            n_features=n_features,
            patch_len=patch_len,
            n_patches=n_patches,
            append_masks=append_masks,
            probabilistic=probabilistic
        )

        encoder_input_dim = n_features * (1 + append_masks)
        encoder_config = MLPConfig(
            input_dim=encoder_input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=prototype_dim
        )

        prototypes_config = PrototypeConfig(
            n_prototypes=n_prototypes,
            prototype_dim=prototype_dim,
            distance_metric=prototype_distance_metric
        )

        classifier_config = MLPConfig(
            input_dim=n_prototypes,
            hidden_dims=classifier_hidden_dims,
            output_dim=n_classes
        )

        return ProTabConfig(
            patching=patching_config,
            encoder=encoder_config,
            prototypes=prototypes_config,
            classifier=classifier_config
        )

    @staticmethod
    def from_yaml(config_path: Path) -> ProTabConfig:
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        patching_config = PatchingConfig(**config_dict["patching"])
        encoder_config = MLPConfig(**config_dict["encoder"])
        prototypes_config = PrototypeConfig(**config_dict["prototypes"])
        classifier_config = MLPConfig(**config_dict["classifier"])

        return ProTabConfig(
            patching=patching_config,
            encoder=encoder_config,
            prototypes=prototypes_config,
            classifier=classifier_config
        )


class ProTab(nn.Module):
    def __init__(
            self,
            config: ProTabConfig
    ) -> None:
        super().__init__()
        self.config = config

        self.patching = ProbabilisticPatching(self.config.patching)
        self.encoder = MLP(self.config.encoder)
        self.prototypes = Prototypes(self.config.prototypes)
        self.classifier = MLP(self.config.classifier)

    def embeddings(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        patches = self.patching(x)
        patches_embeddings = self.encoder(patches)
        patches_embeddings = F.normalize(patches_embeddings, p=2, dim=-1)
        return patches_embeddings

    def forward(
            self,
            x: torch.Tensor,
            return_embeddings: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        patches_embeddings = self.embeddings(x)
        prototype_dist, patches_idcs = self.prototypes(patches_embeddings)

        similarity = torch.exp(-prototype_dist)

        logits = self.classifier(similarity)
        logits = logits.squeeze(1)
        if return_embeddings:
            return logits, patches_embeddings
        return logits

    def set_real_prototypes(
            self,
            dataloader: torch.utils.data.DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.prototypes.prototypes.device
        self.eval()

        worker_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=int(dataloader.batch_size),
            shuffle=False,
            num_workers=dataloader.num_workers,
            drop_last=False
        )

        n_protos = self.config.prototypes.n_prototypes
        patch_dim = self.config.patching.n_features * (1 + self.config.patching.append_masks)
        n_patches_per_sample = self.config.patching.n_patches

        global_min_dists = torch.full((n_protos,), float("inf"), device=device)
        global_nearest_idcs = torch.full((n_protos, 2), -1, device=device, dtype=torch.long)
        best_patches = torch.zeros((n_protos, patch_dim), device=device)

        p = F.normalize(self.prototypes.prototypes.data, p=2, dim=-1)
        p_sq = torch.sum(p ** 2, dim=-1)

        best_prototype_vectors = p.clone()

        with torch.no_grad():
            for batch_idx, (x_batch, *_) in enumerate(worker_dataloader):
                x_batch = x_batch.to(device).to(torch.float32)

                patches = self.patching(x_batch)
                z = self.encoder(patches)
                z = F.normalize(z, p=2, dim=-1)

                B, P, E = z.shape
                z_flat = z.view(-1, E)
                p_flat = patches.view(-1, patch_dim)

                z_sq = torch.sum(z_flat ** 2, dim=-1, keepdim=True)
                distances_sq = z_sq + p_sq - 2 * torch.matmul(z_flat, p.t())

                distances = torch.sqrt(torch.clamp(distances_sq, min=1e-8))  # (B * P, n_protos)

                # Find batch-wise nearest neighbors
                batch_min_dists, batch_argmin = torch.min(distances, dim=0)  # (n_protos,)

                # Update global best
                update_mask = batch_min_dists < global_min_dists

                if update_mask.any():
                    global_min_dists[update_mask] = batch_min_dists[update_mask]

                    indices_to_update = batch_argmin[update_mask]
                    best_prototype_vectors[update_mask] = z_flat[indices_to_update]
                    best_patches[update_mask] = p_flat[indices_to_update]

                    # Index math
                    sample_idx_in_batch = indices_to_update // n_patches_per_sample
                    patch_idx_within_sample = indices_to_update % n_patches_per_sample

                    batch_start_idx = batch_idx * worker_dataloader.batch_size
                    absolute_sample_idx = batch_start_idx + sample_idx_in_batch

                    global_nearest_idcs[update_mask] = torch.stack(
                        [absolute_sample_idx, patch_idx_within_sample], dim=1
                    )

        # Sync the model's learned prototypes with the real projected ones
        self.prototypes.prototypes.data.copy_(best_prototype_vectors)

        if self.config.patching.append_masks:
            mask = best_patches[:, self.config.patching.n_features:].bool()
            best_patches = best_patches[:, :self.config.patching.n_features]

            best_patches[~mask] = torch.nan

        return global_min_dists, global_nearest_idcs, best_patches
