import torch

from ltc.validation import validate_duration
from models.continuous_controller import ContinuousController


class ControllerStream:
    """Gestisce la memoria di un singolo controllore in inferenza."""

    def __init__(self, model: ContinuousController, *, dt: float):
        validate_duration(dt)

        self.model = model
        self.model.eval()
        self.dt = dt
        self._state = None

    def reset(self) -> None:
        """Prepara il controllore per un nuovo episodio indipendente."""
        self._state = None

    def state_snapshot(self) -> torch.Tensor | None:
        """Restituisce una copia dello stato, senza avanzare la rete."""
        if self._state is None:
            return None

        return self._state.detach().clone()

    @torch.no_grad()
    def push(self, sample: torch.Tensor) -> torch.Tensor:
        """Riceve [ingressi] e restituisce [uscite]."""
        expected = (self.model.cell.config.input_size,)

        if sample.shape != expected:
            raise ValueError(
                f"Forma del campione {tuple(sample.shape)}, "
                f"attesa {expected}."
            )

        reference = self.model.cell.neurons.resting_potential
        sample = sample.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if not torch.isfinite(sample).all().item():
            raise ValueError("Il campione contiene valori non finiti.")

        state = self._state
        if state is None:
            state = self.model.initial_state(batch_size=1)

        commands, next_state = self.model(
            sample.unsqueeze(0),
            state,
            dt=self.dt,
        )

        if not torch.isfinite(commands).all().item():
            raise RuntimeError("Il controllore ha prodotto comandi non finiti.")

        if not torch.isfinite(next_state).all().item():
            raise RuntimeError("Il controllore ha prodotto uno stato non finito.")

        self._state = next_state.detach().clone()

        return commands[0].detach().clone()