import os
from pathlib import Path

import yaml

from protab.data.dataset import (DataContainerConfig,
                                 DataContainer)
from protab.data.named_data import TNamedData
from protab.models.protab import (ProTabConfig,
                                  ProTabConfigFactory)
from protab.training.log import WandbConfig
from protab.training.loss import CompoundLossConfig
from protab.training.trainer import ProTabTrainerConfig


def read_data_and_configs(
        dataset_name: TNamedData
) -> tuple[DataContainer, ProTabConfig, ProTabTrainerConfig]:
    base = os.environ.get("PROTAB_CONFIGS", "./configs/")
    base_path = Path(base)
    config_path = base_path / f"{dataset_name}.yml"

    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)

    data_container = DataContainer(DataContainerConfig(name=config_dict["data"]["name"]))
    config_dict["model"]["n_features"] = data_container.n_features
    config_dict["model"]["n_classes"] = data_container.n_classes

    protab_config = ProTabConfigFactory.build(**config_dict["model"])
    config_dict["trainer"]["wandb_config"] = WandbConfig(**config_dict["trainer"]["wandb_config"])

    config_dict["trainer"]["criterion_config"]["ce_pos_weight"] = data_container.pos_weight
    config_dict["trainer"]["criterion_config"] = CompoundLossConfig(**config_dict["trainer"]["criterion_config"])
    trainer_config = ProTabTrainerConfig(**config_dict["trainer"])

    return data_container, protab_config, trainer_config
