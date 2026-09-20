import torch
from torch import nn

from ltc.cell import LTCCell
from ltc.config import LTCConfig
from ltc.sequence import run_sequence


class SequenceClassifier(nn.Module):
    """Classifica una sequenza leggendo lo stato finale della LTC."""

    def __init__(self, config: LTCConfig, num_classes: int = 2):
        super().__init__()

        if type(num_classes) is not int:
            raise TypeError("num_classes deve essere un numero intero.")

        if num_classes < 2:
            raise ValueError("Servono almeno due classi.")

        self.cell = LTCCell(config)
        self.readout = nn.Linear(config.hidden_size, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        _, final_state = run_sequence(
            self.cell,
            inputs,
            dt=dt,
        )

        return self.readout(final_state)