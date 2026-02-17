import click
import optuna

from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import read_data_and_configs
from protab.training.reproducibility import set_seed
from protab.training.trainer import ProTabTrainer


def objective(
        trial: optuna.Trial,
        dataset_name: TNamedData,
        device: str,
        log_wandb: bool
) -> float:
    set_seed()
    
    data_container, protab_config, trainer_config = read_data_and_configs(dataset_name)
    trainer_config.device = device
    trainer_config.wandb_config.active = log_wandb

    # model architecture hyperparameters
    # noinspection PyTypeChecker
    encoder_dims = trial.suggest_categorical("protab_config.encoder.hidden_dims", [
        "32,32", "64,64", "32,32,32", "64,64,64",  # simple architectures
        "32,16", "64,32", "128,64", "128,64,32", "256,96,32", "128,64,32,16",  # gradual compression
        "64,128,64"  # diamond shape for non-linear feature interactions
    ])
    # protab_config.patching.append_masks = trial.suggest_categorical("protab_config.patching.append_masks", [True, False])
    protab_config.patching.append_masks = True
    protab_config.encoder.input_dim = data_container.n_features * (2 if protab_config.patching.append_masks else 1)

    # model architecture hyperparameters
    protab_config.encoder.hidden_dims = [int(dim_str.strip()) for dim_str in encoder_dims.split(",")]
    prototype_dim = trial.suggest_categorical("protab_config.prototypes.prototype_dim", [3, 5, 8, 12, 16])
    protab_config.prototypes.prototype_dim = prototype_dim
    protab_config.encoder.output_dim = prototype_dim

    n_prototypes = trial.suggest_categorical("protab_config.prototypes.n_prototypes", [32, 96, 128, 192, 256])
    protab_config.prototypes.n_prototypes = n_prototypes
    protab_config.classifier.input_dim = n_prototypes

    n_patches = trial.suggest_categorical("protab_config.patching.n_patches", [8, 16, 24, 32])
    protab_config.patching.n_patches = n_patches

    protab_config.patching.patch_len = trial.suggest_int("protab_config.patching.patch_len",
                                                         3, min(data_container.n_features // 2, 16))

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
@click.option("--device", type=str, default="cpu")
@click.option("--n-trials", type=int, default=100)
@click.option("--log-wandb", is_flag=True)
def main(dataset_name: TNamedData, device: str, n_trials: int, log_wandb) -> None:
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(lambda trial: objective(trial, dataset_name, device, log_wandb), n_trials=n_trials,
                   timeout=3 * 24 * 60 * 60)
    print(f"Best params is {study.best_params} with value {study.best_value}")


if __name__ == "__main__":
    main()
