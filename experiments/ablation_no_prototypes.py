from protab.models.protab import ProTab
from protab.training.config import read_data_and_configs
from protab.training.trainer import ProTabTrainer

if __name__ == "__main__":
    data_container, protab_config, trainer_config = read_data_and_configs("codrna")

    trainer_config.epochs_stage_2 = 0
    trainer_config.epochs_stage_3 = 0

    trainer_config.criterion_config.w_ce = 1.0
    trainer_config.criterion_config.w_triplet = 0.0
    trainer_config.criterion_config.w_patch_diversity = 0.0
    trainer_config.criterion_config.w_proto_diversity = 0.0

    protab_config.patching.n_patches = 1
    protab_config.patching.patch_len = protab_config.patching.n_features

    protab_config.prototypes.n_prototypes = protab_config.classifier.output_dim

    protab = ProTab(protab_config)
    trainer = ProTabTrainer(data_container, protab, trainer_config)

    trainer.train(wandb_tags=["ablation", "no_prototypes"])
