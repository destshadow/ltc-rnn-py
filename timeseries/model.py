from dataclasses import dataclass

import torch
from torch import nn

from cfc.cell import CfcCell
from cfc.config import CfcConfig
from cfc.sequence import run_cfc_sequence
from ltc.cell import LTCCell
from ltc.config import LTCConfig
from ltc.sequence import run_sequence


@dataclass(frozen=True)
class LstmConfig:
    input_size: int
    hidden_size: int


class ReturnForecaster(nn.Module):
    """Stima il prossimo valore della serie dallo stato finale di una LTC o di una CfC.

    Con la CfC il Δt di ogni passo è dt moltiplicato per l'ingresso `elapsed_channel`
    (la durata dell'intervallo), invece di un passo fisso. Supporta intervalli variabili;
    la formula NON garantisce stato invariato a Δt = 0 né che un passo lungo equivalga
    a più passi brevi.
    """

    def __init__(self, config: LTCConfig | CfcConfig, *, elapsed_channel: int | None = None):
        super().__init__()

        if isinstance(config, CfcConfig):
            if elapsed_channel is None or not 0 <= elapsed_channel < config.input_size:
                raise ValueError("Con la CfC serve elapsed_channel, l'ingresso con la durata.")
            self.cell = CfcCell(config)
        elif isinstance(config, LTCConfig):
            self.cell = LTCCell(config)
        else:
            raise TypeError("config deve essere LTCConfig o CfcConfig.")

        self.elapsed_channel = elapsed_channel
        self.readout = nn.Linear(config.hidden_size, 1)

    def forward(self, inputs: torch.Tensor, *, dt: float) -> torch.Tensor:
        """inputs: [batch, istanti, ingressi] -> previsioni [batch]."""
        if isinstance(self.cell, CfcCell):
            _, final_state = run_cfc_sequence(
                self.cell, inputs, dt=dt * inputs[:, :, self.elapsed_channel],
                collect_history=False,
            )
        else:
            _, final_state = run_sequence(self.cell, inputs, dt=dt, collect_history=False)

        return self.readout(final_state).squeeze(-1)


class LstmForecaster(nn.Module):
    """Riferimento obbligatorio: una LSTM standard sulle stesse finestre.

    Non usa dt: vede la durata dell'intervallo solo come ingresso, come qualunque RNN
    a passo fisso. Se LTC/CfC non la battono, la loro complessità non è giustificata.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, *, dt: float) -> torch.Tensor:
        _, (final_hidden, _) = self.lstm(inputs)
        return self.readout(final_hidden[-1]).squeeze(-1)


def make_forecaster(cell: str, *, input_size: int, hidden_size: int, substeps: int,
                    backbone_units: int, elapsed_channel: int):
    """Configurazione e modello per --cell ltc|cfc|lstm; la configurazione va nel checkpoint."""
    if cell == "ltc":
        config = LTCConfig(input_size=input_size, hidden_size=hidden_size, substeps=substeps)
        return config, ReturnForecaster(config)
    if cell == "cfc":
        config = CfcConfig(input_size=input_size, hidden_size=hidden_size,
                           backbone_units=backbone_units)
        return config, ReturnForecaster(config, elapsed_channel=elapsed_channel)
    if cell == "lstm":
        return LstmConfig(input_size=input_size, hidden_size=hidden_size), LstmForecaster(
            input_size, hidden_size)
    raise ValueError("cell deve essere 'ltc', 'cfc' o 'lstm'.")
