import os
import platform
import uuid
from dataclasses import dataclass
from typing import Sequence

import torch
import torchmetrics.functional as tmf
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protab.data.dataset import (DataContainer,
                                 SimpleDataset)
from protab.models.protab import ProTab
from protab.training.log import WandbConfig
from protab.training.loss import (CompoundLoss,
                                  CompoundLossConfig)
from protab.training.reproducibility import set_seed
from protab.training.utils import SimpleCounter


@dataclass
class ProTabTrainerConfig:
    batch_size: int
    epochs_stage_1: int
    epochs_stage_2: int
    epochs_stage_3: int
    criterion_config: CompoundLossConfig
    wandb_config: WandbConfig
    device: str = "cpu"
    learning_rate: float = 1e-3
    run_validation: bool = True
    verbose: bool = True


class ProTabTrainer:
    def __init__(
            self,
            data_container: DataContainer,
            model: ProTab,
            config: ProTabTrainerConfig
    ) -> None:
        set_seed()
        self.config = config
        self.device = torch.device(self.config.device)
        self.data_container = data_container
        train_set, eval_set, _ = self.data_container.to_triplet_datasets()

        self.train_dataloader = self._build_dataloader(train_set)
        self.eval_dataloader = self._build_dataloader(eval_set)

        self.model = model.to(self.device)
        self.criterion = CompoundLoss(self.config.criterion_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self._iter_counter = SimpleCounter()

    def _build_dataloader(
            self,
            dataset: SimpleDataset,
            shuffle: bool = True,
            num_workers: int = 1
    ) -> DataLoader:

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if "cuda" in self.config.device else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def _initialize_prototypes_kmeans(self, max_samples: int = 100000) -> None:
        self.model.eval()
        all_embeddings = []
        total_collected = 0

        with torch.no_grad():
            for batch in self.train_dataloader:
                anchor = batch[0].to(self.device)

                patches = self.model.patching(anchor)
                embeddings = self.model.encoder(patches)

                flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])  # (B * P, E)

                all_embeddings.append(flat_embeddings)
                total_collected += flat_embeddings.shape[0]

                if total_collected >= max_samples:
                    break

        full_embeddings = torch.cat(all_embeddings, dim=0)

        if full_embeddings.shape[0] > max_samples:
            indices = torch.randperm(full_embeddings.shape[0])[:max_samples]
            full_embeddings = full_embeddings[indices]

        self.model.prototypes.init_with_kmeans(full_embeddings)

    def _train_epoch(self) -> None:
        self.model.train()
        total_loss = 0.0

        for anchor, positive, negative, labels in tqdm(self.train_dataloader, leave=False, desc="Epoch batches",
                                                       disable=not self.config.verbose):
            self.optimizer.zero_grad()

            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            labels = labels.to(self.device)

            logits, anchor_embeddings = self.model.forward(anchor, return_embeddings=True)  # (B, C), (B, P, E)
            positive_embeddings = self.model.embeddings(positive)  # (B, P, E)
            negative_embeddings = self.model.embeddings(negative)  # (B, P, E)

            loss = self.criterion(
                logits=logits,
                targets=labels,
                anchor_embeddings=anchor_embeddings,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
                patching_weights=self.model.patching.weights,
                prototypes_weights=self.model.prototypes.prototypes
            )
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.train_dataloader)
        wandb.log({"train_loss": avg_loss}, step=self._iter_counter())

    def _train_stage(
            self,
            n_epochs: int,
            idx: int | None = None
    ) -> None:
        desc = f"Training epochs (stage {idx})" if idx is not None else "Evaluation batches"

        for _ in tqdm(range(n_epochs), desc=desc, disable=not self.config.verbose):
            self._train_epoch()

    def _validation(
            self,
            idx: int | None = None
    ) -> dict[str, float]:

        self.model.eval()
        metrics = {}
        logits_list = []
        labels_list = []

        desc = f"Evaluation batches (stage {idx})" if idx is not None else "Evaluation batches"

        with torch.no_grad():
            for anchor, positive, negative, labels in tqdm(self.eval_dataloader, leave=False, desc=desc,
                                                           disable=not self.config.verbose):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                labels = labels.to(self.device)

                logits, anchor_embeddings = self.model.forward(anchor, return_embeddings=True)  # (B, C), (B, P, E)

                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

                positive_embeddings = self.model.embeddings(positive)  # (B, P, E)
                negative_embeddings = self.model.embeddings(negative)  # (B, P, E)

                total, ce, triplet, patch_diversity, proto_diversity = self.criterion.forward_partial(
                    logits=logits,
                    targets=labels,
                    anchor_embeddings=anchor_embeddings,
                    positive_embeddings=positive_embeddings,
                    negative_embeddings=negative_embeddings,
                    patching_weights=self.model.patching.weights,
                    prototypes_weights=self.model.prototypes.prototypes
                )

                for m in ["total", "ce", "triplet", "patch_diversity", "proto_diversity"]:
                    if m not in metrics:
                        metrics[m] = 0.0
                    metrics[m] += locals()[m].item() * len(labels)

        for m in metrics:
            metrics[m] /= len(self.eval_dataloader.dataset)

        logits_all = torch.cat(logits_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        labels_all = torch.argmax(labels_all, dim=1)

        num_classes = logits_all.shape[-1]
        task = "multiclass"

        metrics["accuracy"] = tmf.classification.accuracy(logits_all, labels_all, task=task, average="micro",
                                                          num_classes=num_classes).item()
        metrics["balanced_accuracy"] = tmf.classification.recall(logits_all, labels_all, average="macro", task=task,
                                                                 num_classes=num_classes).item()
        metrics["precision"] = tmf.classification.precision(logits_all, labels_all, average="macro", task=task,
                                                            num_classes=num_classes).item()
        metrics["f1_score"] = tmf.classification.f1_score(logits_all, labels_all, average="macro", task=task,
                                                          num_classes=num_classes).item()
        metrics["cohen_kappa"] = tmf.cohen_kappa(logits_all, labels_all, task=task, num_classes=num_classes).item()

        wandb_log_dict = {f"eval_{k}": v for k, v in metrics.items()}
        wandb.log(wandb_log_dict, step=int(self._iter_counter))

        if self.config.verbose:
            print("Metrics:", metrics)

        return metrics

    def train(
            self,
            return_score: bool = False,
            wandb_tags: Sequence[str] | None = None,
            wandb_finish: bool = True
    ) -> None | float:
        import wandb

        platform_name = platform.node()

        wandb.init(
            project=self.config.wandb_config.project,
            entity=self.config.wandb_config.entity,
            name=f"{self.data_container.config.name}_{platform_name}_{uuid.uuid4()}",
            mode="online" if self.config.wandb_config.active else "disabled",
            tags=wandb_tags,
            config={
                "architecture": "ProTab",
                "model": self.model.config.__dict__,
                "trainer": self.config.__dict__,
                "data": self.data_container.config.__dict__,
                "platform": platform_name
            }
        )

        self._initialize_prototypes_kmeans()

        # Stage 1: probabilistic patching
        self._train_stage(self.config.epochs_stage_1, idx=1)
        if self.config.run_validation:
            self._validation(idx=1)

        # Stage 2: deterministic patching, artificial prototypes
        if self.config.epochs_stage_2 > 0:
            self.model.patching.config.probabilistic = False
            self._train_stage(self.config.epochs_stage_2, idx=2)
            if self.config.run_validation:
                self._validation(idx=2)

        # Stage 3: deterministic patching, real-world prototypes, classification fine-tuning
        if self.config.epochs_stage_3 > 0:
            train_dataset = SimpleDataset(self.data_container.x_train, self.data_container.y_train)
            dataloader = self._build_dataloader(train_dataset, shuffle=False)
            dist, idcs, patches = self.model.set_real_prototypes(dataloader)
            self._train_stage(self.config.epochs_stage_3, idx=3)

            readable_patches = self.data_container.descale(patches)

            if self.config.run_validation:
                metrics = self._validation(idx=3)

            if self.config.verbose:
                print("Distances:")
                print(dist)
                print("Indices:")
                print(idcs)
                print("Prototypical parts:")
                print(readable_patches)
                print("Classification matrix:")
                print(self.model.classifier.network[-1].weight.data)

            cls_matrix_cols = [f"Proto_{i}" for i in range(self.model.classifier.network[-1].weight.shape[1])]
            wandb.log({
                "distances": dist.tolist(),
                "indices": idcs.tolist(),
                "prototypical_parts": wandb.Table(data=readable_patches.to_numpy().tolist(),
                                                  columns=readable_patches.columns.map(str).tolist()),
                "classification_matrix": wandb.Table(data=self.model.classifier.network[-1].weight.data.cpu().tolist(),
                                                     columns=cls_matrix_cols)
            }, step=int(self._iter_counter))

        if self.config.wandb_config.active:
            model_filename = "model_state_dict.pt"
            model_path = os.path.join(wandb.run.dir, model_filename)
            torch.save(self.model.state_dict(), model_path)
            wandb.save(model_path, policy="now")

        if wandb_finish:
            wandb.finish()

        if return_score and self.config.run_validation:
            return metrics["balanced_accuracy"]
        return None
