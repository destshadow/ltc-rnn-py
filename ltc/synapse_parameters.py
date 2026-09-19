import torch
from torch import nn
from torch.nn import functional as F


class SynapseParameters(nn.Module):
    """Parametri di un gruppo di collegamenti sorgente-destinazione."""

    def __init__(self, source_size: int, target_size: int):
        super().__init__()

        for name, value in (
            ("source_size", source_size),
            ("target_size", target_size),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} deve essere un numero intero.")

            if value <= 0:
                raise ValueError(f"{name} deve essere maggiore di zero.")

        shape = (source_size, target_size)

        self.raw_strength = nn.Parameter(
            torch.empty(shape).uniform_(-3.0, -1.0)
        )

        self.raw_slope = nn.Parameter(
            torch.ones(shape)
        )

        self.threshold = nn.Parameter(
            torch.empty(shape).uniform_(-0.5, 0.5)
        )

        initial_reversal = torch.randint(0, 2, shape).float() * 2 - 1
        self.reversal_potential = nn.Parameter(initial_reversal)

    @property
    def strength(self) -> torch.Tensor:
        return F.softplus(self.raw_strength) + 1e-6

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope) + 1e-6