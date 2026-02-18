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

    trainer_config.criterion_config.w_cls = 1.0
    trainer_config.criterion_config.w_triplet = 0.0
    trainer_config.criterion_config.w_patch_diversity = 0.0
    trainer_config.criterion_config.w_proto_diversity = 0.0

    protab = ProTab(model_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    current_run_summary = trainer.train(wandb_tags=["ablation", "no_compound_loss"], wandb_finish=False, return_score=True)

    best_run_summary = best_run.summary
    metric_dict = {}
    for metric in ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_score", "precision"]:
        best_metric = best_run_summary[f"eval_{metric}"]
        current_metric = current_run_summary[metric]
        diff = current_metric - best_metric
        metric_dict[f"eval_{metric}_diff"] = diff

    wandb.log(metric_dict)

    wandb.finish()


if __name__ == "__main__":
    main()
