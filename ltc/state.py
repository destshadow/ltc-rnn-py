import torch

from .config import LTCConfig


#La funzione ha una sola responsabilità: creare lo stato iniziale con dimensioni, dispositivo e tipo numerico scelti.

def create_initial_state(
    config: LTCConfig,
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Crea uno stato iniziale indipendente per ogni sequenza."""
    if type(batch_size) is not int:
        raise TypeError("batch_size deve essere un numero intero.")

    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di zero.")

    if not dtype.is_floating_point:
        raise TypeError("Lo stato deve usare numeri in virgola mobile.")

    return torch.zeros(
        (batch_size, config.hidden_size),
        device=device,
        dtype=dtype,
    )
