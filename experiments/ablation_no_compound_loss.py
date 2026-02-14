import click

from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import read_data_and_configs
from protab.training.trainer import ProTabTrainer


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
def main(dataset_name: TNamedData) -> None:
    data_container, protab_config, trainer_config = read_data_and_configs(dataset_name)

    trainer_config.criterion_config.w_cls = 1.0
    trainer_config.criterion_config.w_triplet = 0.0
    trainer_config.criterion_config.w_patch_diversity = 0.0
    trainer_config.criterion_config.w_proto_diversity = 0.0

    protab = ProTab(protab_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    trainer.train(wandb_tags=["ablation", "no_compound_loss"])


if __name__ == "__main__":
    main()
