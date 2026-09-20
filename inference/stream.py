import torch

from ltc.sequence import run_sequence
from ltc.validation import validate_duration
from models.sequence_classifier import SequenceClassifier


class LTCStream:
    """Esegue un singolo flusso, conservando lo stato tra i blocchi."""

    def __init__(self, model: SequenceClassifier, *, dt: float):
        validate_duration(dt)

        self.model = model
        self.model.eval()

        self.dt = dt
        self._state = None

    def reset(self) -> None:
        """Inizia un nuovo flusso indipendente."""
        self._state = None

    @torch.no_grad()
    def push_block(self, samples: torch.Tensor) -> torch.Tensor:
        """Riceve [istanti, ingressi] e restituisce [1, classi]."""
        if samples.ndim != 2:
            raise ValueError(
                "samples deve avere forma [istanti, ingressi]."
            )

        if samples.shape[0] == 0:
            raise ValueError("Il blocco non può essere vuoto.")

        if samples.shape[1] != self.model.cell.config.input_size:
            raise ValueError("Numero di ingressi non compatibile.")

        reference = self.model.cell.neurons.resting_potential
        samples = samples.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        _, final_state = run_sequence(
            self.model.cell,
            samples.unsqueeze(0),
            dt=self.dt,
            initial_state=self._state,
            collect_history=False,
        )

        self._state = final_state

        return self.model.readout(final_state)

    def push(self, sample: torch.Tensor) -> torch.Tensor:
        """Riceve un singolo campione con forma [ingressi]."""
        if sample.ndim != 1:
            raise ValueError("sample deve avere forma [ingressi].")

        return self.push_block(sample.unsqueeze(0))