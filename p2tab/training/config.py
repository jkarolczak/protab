import os
import tempfile
from pathlib import Path

import torch
import wandb
import yaml

from p2tab.data.dataset import DataContainer
from p2tab.data.dataset import DataContainerConfig
from p2tab.data.named_data import TNamedData
from p2tab.models.mlp import MLPConfig
from p2tab.models.p2tab import (P2Tab,
                                P2TabConfig,
                                P2TabConfigFactory)
from p2tab.nn.patching import PatchingConfig
from p2tab.nn.prototypes import PrototypeConfig
from p2tab.training.log import WandbConfig
from p2tab.training.loss import CompoundLossConfig
from p2tab.training.trainer import P2TabTrainerConfig


def read_data_and_configs(
        dataset_name: TNamedData
) -> tuple[DataContainer, P2TabConfig, P2TabTrainerConfig]:
    base = os.environ.get("P2TAB_CONFIGS", "./configs/")
    base_path = Path(base)
    config_path = base_path / f"{dataset_name}.yml"

    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)

    data_container = DataContainer(DataContainerConfig(name=config_dict["data"]["name"]))
    config_dict["model"]["n_features"] = data_container.n_features
    config_dict["model"]["n_classes"] = data_container.n_classes

    p2tab_config = P2TabConfigFactory.build(**config_dict["model"])
    config_dict["trainer"]["wandb_config"] = WandbConfig(**config_dict["trainer"]["wandb_config"])

    config_dict["trainer"]["criterion_config"]["ce_pos_weight"] = data_container.pos_weight
    config_dict["trainer"]["criterion_config"] = CompoundLossConfig(**config_dict["trainer"]["criterion_config"])
    trainer_config = P2TabTrainerConfig(**config_dict["trainer"])

    return data_container, p2tab_config, trainer_config


def fetch_best_run(dataset_name: str, tags: list[str], load_model: bool = False):
    api = wandb.Api()

    runs = api.runs(
        path="jacek-karolczak/P2Tab",
        filters={
            "config.architecture": "P2Tab",
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

    p2tab = config["model"]

    p2tab["encoder"]["activation"] = eval(p2tab["encoder"]["activation"])
    p2tab["classifier"]["activation"] = eval(p2tab["classifier"]["activation"])

    p2tab["patching"] = PatchingConfig(**p2tab["patching"])
    p2tab["encoder"] = MLPConfig(**p2tab["encoder"])
    p2tab["prototypes"] = PrototypeConfig(**p2tab["prototypes"])
    p2tab["classifier"] = MLPConfig(**p2tab["classifier"])

    p2tab_config = P2TabConfig(**p2tab)

    trainer_dict = config["trainer"]

    wandb_cfg = WandbConfig(**trainer_dict.get("wandb_config", {}))

    crit_dict = trainer_dict.get("criterion_config", {})
    crit_dict["ce_pos_weight"] = data_container.pos_weight
    criterion_cfg = CompoundLossConfig(**crit_dict)

    trainer_dict["wandb_config"] = wandb_cfg
    trainer_dict["criterion_config"] = criterion_cfg

    trainer_config = P2TabTrainerConfig(**trainer_dict)

    if "cuda" in trainer_config.device and not torch.cuda.is_available():
        trainer_config.device = "cpu"

    model = None
    if load_model:
        model = P2Tab(p2tab_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            f = best_run.file("files/model_state_dict.pt")
            f.download(root=tmp_dir, replace=True)

            state_dict_path = Path(tmp_dir) / "files" / "model_state_dict.pt"

            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)

        model.to(trainer_config.device)
        model.eval()

    return best_run, data_container, p2tab_config, trainer_config, model
