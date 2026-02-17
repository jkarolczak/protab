import click
import wandb

from protab.data.named_data import TNamedBoolData
from protab.models.protab import ProTab
from protab.training.config import fetch_best_run
from protab.training.trainer import ProTabTrainer


@click.command()
@click.argument("dataset_name", type=click.Choice(TNamedBoolData.__args__))
@click.option("--device", type=str, default="cpu")
def main(dataset_name: TNamedBoolData, device: str) -> None:
    best_run, data_container, model_config, trainer_config, _ = fetch_best_run(
        dataset_name, ["hyperparameter_tuning"], load_model=False
    )

    trainer_config.device = device

    trainer_config.epochs_stage_2 = 0
    trainer_config.epochs_stage_3 = 0

    trainer_config.criterion_config.w_cls = 1.0
    trainer_config.criterion_config.w_triplet = 0.0
    trainer_config.criterion_config.w_patch_diversity = 0.0
    trainer_config.criterion_config.w_proto_diversity = 0.0

    model_config.patching.n_patches = 1
    model_config.patching.patch_len = model_config.patching.n_features

    model_config.prototypes.n_prototypes = model_config.classifier.output_dim
    model_config.classifier.input_dim = model_config.prototypes.n_prototypes

    protab = ProTab(model_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    trainer.train(wandb_tags=["ablation", "no_prototypes"], wandb_finish=False)

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
