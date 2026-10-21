import torch
from torch import nn

from ltc.cell import LTCCell
from ltc.config import LTCConfig
from ltc.sequence import run_sequence


class ReturnForecaster(nn.Module):
    """Stima il prossimo valore della serie dallo stato finale della LTC."""

    def __init__(self, config: LTCConfig):
        super().__init__()

        self.cell = LTCCell(config)
        self.readout = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        """inputs: [batch, istanti, ingressi] -> previsioni [batch]."""
        _, final_state = run_sequence(
            self.cell,
            inputs,
            dt=dt,
            collect_history=False,
        )

        return self.readout(final_state).squeeze(-1)
