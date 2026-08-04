import click

from p2tab.data.named_data import TNamedData
from p2tab.models.p2tab import P2Tab
from p2tab.training.config import read_data_and_configs
from p2tab.training.trainer import P2TabTrainer


@click.command()
@click.argument("dataset-name", type=click.Choice(TNamedData.__args__))
def main(dataset_name: TNamedData) -> None:
    data_container, p2tab_config, trainer_config = read_data_and_configs(dataset_name)
    p2tab = P2Tab(p2tab_config)
    trainer = P2TabTrainer(data_container, p2tab, trainer_config)

    trainer.train(wandb_tags=["simple_experiment"])


if __name__ == "__main__":
    main()
