import os
from dataclasses import (dataclass,
                         field)
from pathlib import Path
from typing import (Literal,
                    TypeAlias)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from protab.data.named_data import TNamedData

TDatasetPurpose: TypeAlias = Literal["train", "eval", "test"]
TDatasetType: TypeAlias = Literal["x", "y"]


class SimpleDataset(Dataset):
    def __init__(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame
    ) -> None:
        self.x = x.to_numpy()
        self.y = y.to_numpy()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(
            self,
            idx: int
    ) -> tuple:
        return (torch.tensor(self.x[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))


class TripletDataset(SimpleDataset):
    def __init__(
            self,
            x: pd.DataFrame,
            y: pd.DataFrame
    ) -> None:
        super().__init__(x, y)
        self.labels = np.argmax(self.y, axis=1)

        sorted_idcs = np.argsort(self.labels)
        sorted_labels = self.labels[sorted_idcs]

        unique_labels, split_indices = np.unique(sorted_labels, return_index=True)
        grouped_indices = np.split(sorted_idcs, split_indices[1:])

        self.label_to_indices = dict(zip(unique_labels, grouped_indices))
        self.unique_labels = unique_labels

    def find_pos_neg_idcs(
            self,
            idx: int
    ) -> tuple[int, int]:
        anchor_label = self.labels[idx]

        # --- Positive Sampling ---
        pos_indices = self.label_to_indices[anchor_label]

        positive_idx = idx
        if len(pos_indices) > 1:
            while positive_idx == idx:
                positive_idx = np.random.choice(pos_indices)
        else:
            positive_idx = idx

        # --- Negative Sampling ---
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.unique_labels)

        negative_idx = np.random.choice(self.label_to_indices[neg_label])

        return positive_idx, negative_idx

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_idx, negative_idx = self.find_pos_neg_idcs(idx)

        anchor_x = torch.tensor(self.x[idx], dtype=torch.float32)
        positive_x = torch.tensor(self.x[positive_idx], dtype=torch.float32)
        negative_x = torch.tensor(self.x[negative_idx], dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return (
            anchor_x,
            positive_x,
            negative_x,
            y
        )


@dataclass
class DataContainerConfig:
    name: TNamedData | str
    read_csv_kwargs: dict = field(default_factory=dict)


class DataContainer:
    def __init__(
            self,
            config: DataContainerConfig
    ) -> None:
        self.config = config

        self.x_train, self.y_train = self.load_data("train")
        self.x_eval, self.y_eval = self.load_data("eval")
        self.x_test, self.y_test = self.load_data("test")

        self._encode_y_one_hot()
        self._scale_features()

    def build_path(
            self,
            dataset_purpose: TDatasetPurpose,
            data_type: TDatasetType
    ) -> Path:
        base = os.environ.get("PROTAB_DATASETS", "./data/")
        base_path = Path(base)
        full_path = base_path / self.config.name / f"{dataset_purpose}_{data_type}.csv"
        return full_path

    def read_csv(
            self,
            dataset_purpose: TDatasetPurpose,
            data_type: TDatasetType
    ) -> pd.DataFrame:
        path = self.build_path(dataset_purpose, data_type)
        df = pd.read_csv(path, **self.config.read_csv_kwargs)
        return df

    def load_data(
            self,
            dataset_type: TDatasetPurpose
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_df = self.read_csv(dataset_type, "x")
        y_df = self.read_csv(dataset_type, "y")
        return x_df, y_df

    def _encode_y_one_hot(self) -> None:
        col = self.y_train.columns[0]
        self.y_train = pd.get_dummies(self.y_train[col]).astype(int)
        self.y_eval = pd.get_dummies(self.y_eval[col]).astype(int)
        self.y_test = pd.get_dummies(self.y_test[col]).astype(int)

        train_columns = self.y_train.columns

        self.y_eval = self.y_eval.reindex(columns=train_columns, fill_value=0)
        self.y_test = self.y_test.reindex(columns=train_columns, fill_value=0)

    def _scale_features(self) -> None:
        self._mappings = {}
        self._norm_stats = {}

        for col in self.x_train.columns:
            unique_vals = self.x_train[col].dropna().unique()

            if len(unique_vals) == 2:
                unique_vals.sort()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                self._mappings[col] = mapping

                self.x_train[col] = self.x_train[col].map(mapping)
                self.x_eval[col] = self.x_eval[col].map(mapping)
                self.x_test[col] = self.x_test[col].map(mapping)

                self.x_eval[col] = self.x_eval[col].fillna(0).astype(float)
                self.x_test[col] = self.x_test[col].fillna(0).astype(float)

            else:
                mean = self.x_train[col].mean()
                std = self.x_train[col].std()

                if std <= 0.1:
                    std = 1.0

                self._norm_stats[col] = {"mean": float(mean), "std": float(std)}

                self.x_train[col] = (self.x_train[col] - mean) / std
                self.x_eval[col] = (self.x_eval[col] - mean) / std
                self.x_test[col] = (self.x_test[col] - mean) / std

    def __repr__(self) -> str:
        return rf"""DataContainer(
    name={self.config.name}
    x_train shape={self.x_train.shape}, y_train shape={self.y_train.shape}
    x_eval shape={self.x_eval.shape}, y_eval shape={self.y_eval.shape}
    x_test shape={self.x_test.shape}, y_test shape={self.y_test.shape}
)"""

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    @property
    def n_classes(self) -> int:
        return self.y_train.shape[1]

    @property
    def pos_weight(self) -> list[float]:
        num_positives = self.y_train.sum(axis=0)
        num_negatives = len(self.y_train) - num_positives

        pos_weights = (num_negatives / num_positives.replace(0, 1)).values

        return pos_weights.tolist()

    def _create_dataset_bundle(
            self,
            dataset_class: type[SimpleDataset]
    ) -> tuple[SimpleDataset, SimpleDataset, SimpleDataset]:
        train = dataset_class(self.x_train, self.y_train)
        eval = dataset_class(self.x_eval, self.y_eval)
        test = dataset_class(self.x_test, self.y_test)
        return train, eval, test

    def to_simple_datasets(self) -> tuple[SimpleDataset, SimpleDataset, SimpleDataset]:
        return self._create_dataset_bundle(SimpleDataset)

    def to_triplet_datasets(self) -> tuple[TripletDataset, TripletDataset, TripletDataset]:
        return self._create_dataset_bundle(TripletDataset)

    def descale(self, tensor: torch.Tensor) -> pd.DataFrame:
        """
        Converts a tensor (e.g., prototypes/patches) back to original feature scales.
        Handles NaNs by preserving them in the output DataFrame.
        """
        arr = tensor.detach().cpu().numpy()
        df = pd.DataFrame(arr, columns=self.x_train.columns)

        for col, stats in self._norm_stats.items():
            if col in df.columns:
                df[col] = (df[col] * stats["std"]) + stats["mean"]

        for col, mapping in self._mappings.items():
            if col in df.columns:
                inv_mapping = {v: k for k, v in mapping.items()}
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].round().map(inv_mapping)

        return df
