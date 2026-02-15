from dataclasses import (dataclass,
                         field)

import torch
from torch import nn


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    activation: type[nn.Module] = field(default=nn.ReLU)


class MLP(nn.Module):
    def __init__(
            self,
            config: MLPConfig
    ) -> None:
        super().__init__()
        self.config = config
        layers = []
        all_dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(self.config.activation())
        self.network = nn.Sequential(*layers)

        self.apply(MLP._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",
                nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
