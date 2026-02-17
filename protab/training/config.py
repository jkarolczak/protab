import os
import tempfile
from pathlib import Path

import torch
import wandb
import yaml

from protab.data.dataset import DataContainer
from protab.data.dataset import DataContainerConfig
from protab.data.named_data import TNamedData
from protab.models.mlp import MLPConfig
from protab.models.protab import (ProTab,
                                  ProTabConfigFactory)
from protab.models.protab import ProTabConfig
from protab.nn.patching import PatchingConfig
from protab.nn.prototypes import PrototypeConfig
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


def fetch_best_run(dataset_name: str, tags: list[str], load_model: bool = False):
    api = wandb.Api()

    runs = api.runs(
        path="jacek-karolczak/ProTab",
        filters={
            "config.architecture": "ProTab",
            "config.data.name": dataset_name,
            "tags": {"$in": tags},
            "summary_metrics.eval_balanced_accuracy": {"$ne": None}
        },
        order="-summary_metrics.eval_balanced_accuracy",
        per_page=1
    )

    if len(runs) == 0:
        raise ValueError(f"No runs found for dataset '{dataset_name}' with tags {tags}")

    best_run = runs[0]
    config = best_run.config

    data_config = DataContainerConfig(**config["data"])
    data_container = DataContainer(data_config)

    protab_config_dict = config["model"]

    protab_config_dict["encoder"]["activation"] = eval(protab_config_dict["encoder"]["activation"])
    protab_config_dict["classifier"]["activation"] = eval(protab_config_dict["classifier"]["activation"])

    protab_config_dict["patching"] = PatchingConfig(**protab_config_dict["patching"])
    protab_config_dict["encoder"] = MLPConfig(**protab_config_dict["encoder"])
    protab_config_dict["prototypes"] = PrototypeConfig(**protab_config_dict["prototypes"])
    protab_config_dict["classifier"] = MLPConfig(**protab_config_dict["classifier"])

    protab_config = ProTabConfig(**protab_config_dict)

    trainer_dict = config["trainer"]

    wandb_cfg = WandbConfig(**trainer_dict.get("wandb_config", {}))

    crit_dict = trainer_dict.get("criterion_config", {})
    crit_dict["ce_pos_weight"] = data_container.pos_weight
    criterion_cfg = CompoundLossConfig(**crit_dict)

    trainer_dict["wandb_config"] = wandb_cfg
    trainer_dict["criterion_config"] = criterion_cfg

    trainer_config = ProTabTrainerConfig(**trainer_dict)

    if "cuda" in trainer_config.device and not torch.cuda.is_available():
        trainer_config.device = "cpu"

    model = None
    if load_model:
        model = ProTab(protab_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            f = best_run.file("files/model_state_dict.pt")
            f.download(root=tmp_dir, replace=True)

            state_dict_path = Path(tmp_dir) / "files" / "model_state_dict.pt"

            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)

        model.to(trainer_config.device)
        model.eval()

    return best_run, data_container, protab_config, trainer_config, model
