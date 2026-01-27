import click
import optuna

from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import read_data_and_configs
from protab.training.trainer import ProTabTrainer


def objective(
        trial: optuna.Trial,
        dataset_name: TNamedData
) -> float:
    data_container, protab_config, trainer_config = read_data_and_configs(dataset_name)

    # model architecture hyperparameters
    # noinspection PyTypeChecker
    encoder_dims = trial.suggest_categorical("protab_config.encoder_hidden_dims", [
        "128", "384",  # shallow, wide
        "64,64", "192,192",  # two layers, medium
        "32,32,32", "96,96,96",  # three layers, narrow
        "64,128,64",  # diamond shaped
        "128,64,16",  # funnel shaped
        "64,96,128"  # bell shaped
    ])
    protab_config.patching.append_masks = trial.suggest_categorical("protab_config.patching.append_masks", [True, False])
    protab_config.encoder.input_dim = data_container.n_features * (2 if protab_config.patching.append_masks else 1)

    protab_config.encoder_hidden_dims = [int(dim_str) for dim_str in encoder_dims.split(",")]
    prototype_dim = trial.suggest_categorical("protab_config.prototypes.prototype_dim", [2, 3, 5, 8])
    protab_config.prototypes.prototype_dim = prototype_dim
    protab_config.encoder.output_dim = prototype_dim

    # training hyperparameters
    trainer_config.learning_rate = trial.suggest_float("trainer_config.learning_rate", 1e-5, 1e-2)
    trainer_config.batch_size = trial.suggest_categorical("trainer_config.batch_size", [128, 256, 512, 1024])

    trainer_config.verbose = False

    protab = ProTab(protab_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    final_balanced_accuracy = trainer.train(return_score=True, wandb_tags=["hyperparameter_tuning"])
    return final_balanced_accuracy


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
def main(dataset_name: TNamedData) -> None:
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(lambda trial: objective(trial, dataset_name), n_trials=100, timeout=10 * 60)
    print(f"Best params is {study.best_params} with value {study.best_value}")


if __name__ == "__main__":
    main()
