"""Frame-recurrent UA-HSN network (60,289 trainable parameters)."""

import torch
from torch import nn


class UAHSN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(96, 64)
        self.activation = nn.PReLU()
        self.gru1 = nn.GRU(64, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 64, batch_first=True)
        self.gain_head = nn.Linear(64, 32)
        self.log_variance_head = nn.Linear(64, 32)

    def forward(
        self, features: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != 96:
            raise ValueError("Expected batch x frames x 96 features")
        x = self.activation(self.input(features))
        h1, h2 = (None, None) if state is None else state
        x, h1 = self.gru1(x, h1)
        x, h2 = self.gru2(x, h2)
        return torch.sigmoid(self.gain_head(x)), self.log_variance_head(x), (h1, h2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
