import click
import wandb

from protab.data.named_data import TNamedData
from protab.models.protab import ProTab
from protab.training.config import fetch_best_run
from protab.training.trainer import ProTabTrainer


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedData.__args__))
@click.option("--device", type=str, default="cpu")
def main(dataset_name: TNamedData, device: str) -> None:
    best_run, data_container, model_config, trainer_config, _ = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=False
    )
    trainer_config.device = device

    model_config.patching.use_learnable_mask_token = False

    protab = ProTab(model_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    trainer.train(wandb_tags=["ablation", "no_trainable_token"], wandb_finish=False)

    best_run_summary = best_run.summary
    current_run_summary = wandb.run.summary
    metric_dict = {}
    for metric in ["eval_accuracy", "eval_balanced_accuracy", "eval_cohen_kappa", "eval_f1_score", "eval_precision"]:
        best_metric = best_run_summary[metric]
        current_metric = current_run_summary[metric]
        diff = current_metric - best_metric
        metric_dict[f"{metric}_diff"] = diff

    wandb.finish()


if __name__ == "__main__":
    main()
