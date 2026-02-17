import click

from protab.data.dataset import DataContainer, DataContainerConfig
from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import fetch_best_run
from protab.training.reproducibility import set_seed
from protab.training.trainer import ProTabTrainer


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__), default=None)
@click.option("--device", type=str, default="cpu")
@click.option("--n-trials", type=int, default=10)
def main(dataset_name: TNamedData, device: str, n_trials: int) -> None:
    _, _, protab_config, trainer_config, _ = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=False
    )

    trainer_config.device = device
    trainer_config.wandb_config.active = True

    for i in range(n_trials):
        seed = 42 + i
        set_seed(seed)

        data_container = DataContainer(DataContainerConfig(name=dataset_name))
        protab = ProTab(protab_config)
        trainer = ProTabTrainer(data_container, protab, trainer_config)
        trainer.train(wandb_tags=["statistical_significance"])


if __name__ == "__main__":
    main()
