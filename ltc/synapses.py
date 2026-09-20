import torch

from .synapse_parameters import SynapseParameters

#calcola l’effetto dei collegamenti; l’aggiornamento dello stato arriverà nel solver.

def compute_conductances(
    sources: torch.Tensor,
    params: SynapseParameters,
) -> torch.Tensor:
    """Restituisce le conduttanze: [batch, sorgenti, destinazioni]."""
    if sources.ndim != 2:
        raise ValueError("sources deve avere forma [batch, sorgenti].")

    if sources.shape[1] != params.threshold.shape[0]:
        raise ValueError("Il numero di sorgenti non coincide con i collegamenti.")

    if sources.device != params.threshold.device:
        raise ValueError("Sorgenti e parametri devono essere sullo stesso dispositivo.")

    if sources.dtype != params.threshold.dtype:
        raise TypeError("Sorgenti e parametri devono avere lo stesso tipo numerico.")

    source_values = sources.unsqueeze(-1)

    activation = torch.sigmoid(
        params.slope * (source_values - params.threshold) #parte liquida della funzione di attivazione, che determina quanto i neuroni sorgente influenzano i neuroni destinazione ?
    )

    return params.strength * activation


def compute_synaptic_effects(
    sources: torch.Tensor,
    params: SynapseParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restituisce drive e conduttanza totale, entrambi [batch, destinazioni]."""
    conductances = compute_conductances(sources, params)

    drive = (conductances * params.reversal_potential).sum(dim=1)
    total_conductance = conductances.sum(dim=1)

    return drive, total_conductance