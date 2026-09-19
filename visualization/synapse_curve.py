import math

import torch

from ltc.synapse_parameters import SynapseParameters
from ltc.synapses import compute_conductances


def positive_to_raw(value: float) -> float:
    """Converte un valore positivo nel parametro grezzo corrispondente."""
    if not math.isfinite(value) or value <= 1e-6:
        raise ValueError("Il valore deve essere finito e maggiore di 1e-6.")

    return math.log(math.expm1(value - 1e-6))


@torch.no_grad()
def sample_curve(
    sources: torch.Tensor,
    params: SynapseParameters,
    *,
    threshold: float,
    slope: float,
    strength: float,
):
    """Calcola la curva di un singolo collegamento sulla CPU."""
    params.threshold.fill_(threshold)
    params.raw_slope.fill_(positive_to_raw(slope))
    params.raw_strength.fill_(positive_to_raw(strength))

    conductances = compute_conductances(sources, params)

    return conductances[:, 0, 0].cpu().numpy()