import enum
import os
from pathlib import Path
from typing import (Literal,
                    TypeAlias)

import pandas as pd

from protab.training.reproducibility import set_seed

TNamedData: TypeAlias = Literal[
    "bng_ionosphere", "bng_pendigits", "codrna", "covertype", "credit_card", "diabetes", "heloc",
    "statlog_shuttle", "yeast"
]

TNamedBoolData: TypeAlias = Literal["bng_ionosphere", "codrna", "credit_card", "diabetes", "heloc", "yeast"]


class DataSource(enum.Enum):
    UCI = enum.auto()
    OPENML = enum.auto()


DATASET_SOURCE_ID = {
    "bng_ionosphere": (DataSource.OPENML, 59),
    "bng_pendigits": (DataSource.OPENML, 261),
    "codrna": (DataSource.OPENML, 351),
    "covertype": (DataSource.UCI, 31),
    "credit_card": (DataSource.UCI, 350),
    "diabetes": (DataSource.UCI, 329),
    "heloc": (DataSource.OPENML, 45023),
    "statlog_shuttle": (DataSource.UCI, 148),
    "yeast": (DataSource.UCI, 110),
}


def get_uci(id_: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=id_)
    features, targets = dataset.data.features, dataset.data.targets

    if targets.shape[1] > 1:
        targets = targets.loc[:, "CLASS"]
        targets = pd.DataFrame(targets).reset_index(drop=True)

    return features, targets


def get_openml(id_: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import openml

    dataset = openml.datasets.get_dataset(id_)
    features, targets, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    if isinstance(targets, pd.Series):
        targets = targets.to_frame()

    return features, targets


def stratified_train_eval_test_split(
        features: pd.DataFrame,
        targets: pd.DataFrame,
        train_size: float = 0.7,
        eval_size: float = 0.15,
        test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    random_state = set_seed()

    if isinstance(targets.dtypes.iloc[0], pd.SparseDtype):
        targets = pd.DataFrame(targets.to_numpy(), columns=targets.columns)

    x_temp, x_test, y_temp, y_test = train_test_split(
        features, targets, test_size=test_size, stratify=targets, random_state=random_state
    )
    relative_eval_size = eval_size / (train_size + eval_size)
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_temp, y_temp, test_size=relative_eval_size, stratify=y_temp, random_state=random_state
    )

    return x_train, y_train, x_eval, y_eval, x_test, y_test


def save_to_csvs(
        name: TNamedData,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_eval: pd.DataFrame,
        y_eval: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> None:
    base = os.environ.get("PROTAB_DATASETS", "./data/")
    base_path = Path(base)
    dataset_path = base_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)

    x_train.to_csv(dataset_path / "train_x.csv", index=False)
    y_train.to_csv(dataset_path / "train_y.csv", index=False)
    x_eval.to_csv(dataset_path / "eval_x.csv", index=False)
    y_eval.to_csv(dataset_path / "eval_y.csv", index=False)
    x_test.to_csv(dataset_path / "test_x.csv", index=False)
    y_test.to_csv(dataset_path / "test_y.csv", index=False)


def download(name: TNamedData) -> None:
    if name not in TNamedData.__args__:
        raise ValueError(f"Dataset '{name}' is not recognized. Available datasets: {TNamedData.__args__}")

    source, id_ = DATASET_SOURCE_ID[name]
    match source:
        case DataSource.UCI:
            features, targets = get_uci(id_)
        case DataSource.OPENML:
            features, targets = get_openml(id_)

    data_split = stratified_train_eval_test_split(features, targets)
    save_to_csvs(name, *data_split)
