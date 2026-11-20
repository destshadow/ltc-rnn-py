import torch
from torch import nn

from .config import CfcConfig


class LeCunTanh(nn.Module):
    """1.7159 · tanh(0.666 · x), l'attivazione predefinita del backbone negli autori."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh(0.666 * values)


class CfcCell(nn.Module):
    """Cella CfC in forma chiusa: un solo calcolo per campione, Δt come ingresso.

    Struttura e nomi dei pesi seguono la cella "default" (con gate) degli autori
    (libreria ncps), per poterle confrontare:
        z      = backbone([ingresso, stato])
        g, h   = tanh(ff1(z)), tanh(ff2(z))
        gate   = sigmoid(time_a(z) · Δt + time_b(z))
        stato' = g · (1 − gate) + gate · h
    Il nuovo stato è una media pesata di valori in [−1, 1], quindi resta in [−1, 1].
    """

    def __init__(self, config: CfcConfig):
        super().__init__()
        self.config = config

        layers = []
        width = config.input_size + config.hidden_size
        for _ in range(config.backbone_layers):
            layers += [nn.Linear(width, config.backbone_units), LeCunTanh()]
            width = config.backbone_units
        self.backbone = nn.Sequential(*layers)

        self.ff1 = nn.Linear(width, config.hidden_size)
        self.ff2 = nn.Linear(width, config.hidden_size)
        self.time_a = nn.Linear(width, config.hidden_size)
        self.time_b = nn.Linear(width, config.hidden_size)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        reference = self.ff1.weight
        return torch.zeros(batch_size, self.config.hidden_size,
                           device=reference.device, dtype=reference.dtype)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor, *, dt) -> torch.Tensor:
        """inputs [batch, ingressi], state [batch, neuroni]; dt numero o tensore [batch]."""
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_size:
            raise ValueError("inputs deve avere forma [batch, input_size].")
        batch = inputs.shape[0]
        if state.shape != (batch, self.config.hidden_size):
            raise ValueError("state deve avere forma [batch, hidden_size].")

        dt = torch.as_tensor(dt, dtype=inputs.dtype, device=inputs.device)
        if dt.ndim == 0:
            dt = dt.expand(batch)
        if dt.shape != (batch,) or not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError("dt deve essere finito, non negativo, scalare o [batch].")

        features = self.backbone(torch.cat([inputs, state], dim=1))
        g = torch.tanh(self.ff1(features))
        h = torch.tanh(self.ff2(features))
        gate = torch.sigmoid(self.time_a(features) * dt[:, None] + self.time_b(features))
        return g * (1.0 - gate) + gate * h
