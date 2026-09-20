import torch
from torch import nn

from .config import LTCConfig
from .neuron_parameters import NeuronParameters
from .solver import advance_state
from .state import create_initial_state
from .synapse_parameters import SynapseParameters


class LTCCell(nn.Module):
    """Collega i componenti LTC per elaborare un singolo istante."""

    def __init__(self, config: LTCConfig):
        super().__init__()

        self.config = config

        self.neurons = NeuronParameters(config)

        self.sensory = SynapseParameters(
            source_size=config.input_size,
            target_size=config.hidden_size,
        )

        self.recurrent = SynapseParameters(
            source_size=config.hidden_size,
            target_size=config.hidden_size,
        )

    def initial_state(self, batch_size: int) -> torch.Tensor:
        reference = self.neurons.resting_potential

        return create_initial_state(
            self.config,
            batch_size=batch_size,
            device=reference.device,
            dtype=reference.dtype,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        return advance_state(
            inputs=inputs,
            state=state,
            neurons=self.neurons,
            sensory=self.sensory,
            recurrent=self.recurrent,
            dt=dt,
            substeps=self.config.substeps,
        )