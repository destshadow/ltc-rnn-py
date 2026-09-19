import torch
from torch import nn
from torch.nn import functional as F

from .config import LTCConfig


class NeuronParameters(nn.Module):
    """Contiene i parametri apprendibili dei singoli neuroni."""

    def __init__(self, config: LTCConfig):
        super().__init__()

        size = config.hidden_size

        self.raw_capacitance = nn.Parameter(torch.zeros(size))
        self.raw_leak_conductance = nn.Parameter(torch.zeros(size))
        self.resting_potential = nn.Parameter(torch.zeros(size))

    @property
    def capacitance(self) -> torch.Tensor:
        return F.softplus(self.raw_capacitance) + 1e-6

    @property
    def leak_conductance(self) -> torch.Tensor:
        return F.softplus(self.raw_leak_conductance) + 1e-6