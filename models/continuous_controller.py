import torch
from torch import nn

from ltc.cell import LTCCell
from ltc.config import LTCConfig


class ContinuousController(nn.Module):
    """Produce comandi tra 0 e 1, conservando la memoria nello stato."""

    def __init__(self, config: LTCConfig, output_size: int = 1):
        super().__init__()

        if type(output_size) is not int:
            raise TypeError("output_size deve essere un numero intero.")

        if output_size <= 0:
            raise ValueError("output_size deve essere maggiore di zero.")

        self.cell = LTCCell(config)
        self.readout = nn.Linear(config.hidden_size, output_size)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.cell.initial_state(batch_size)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        *,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: [batch, ingressi]
        state:  [batch, neuroni]

        Restituisce:
        commands:   [batch, uscite]
        next_state: [batch, neuroni]
        """
        next_state = self.cell(inputs, state, dt=dt)
        raw_commands = self.readout(next_state)
        commands = torch.sigmoid(raw_commands)

        return commands, next_state