import click

from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import read_data_and_configs
from protab.training.trainer import ProTabTrainer


@click.command()
@click.argument("dataset-name", type=click.Choice(TNamedData.__args__))
def main(dataset_name: TNamedData) -> None:
    data_container, protab_config, trainer_config = read_data_and_configs(dataset_name)
    protab = ProTab(protab_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    trainer.train(wandb_tags=["simple_experiment"])


if __name__ == "__main__":
    main()
