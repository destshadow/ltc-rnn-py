import torch

from ltc.synapses import compute_conductances


@torch.no_grad()
def recurrent_conductances(recurrent, state):
    """Calcola le conduttanze sullo stato fornito, senza avanzarlo."""
    reference = recurrent.raw_strength
    hidden_size = reference.shape[0]

    if state.shape != (hidden_size,):
        raise ValueError("Lo stato deve contenere un valore per neurone.")

    if not torch.isfinite(state).all().item():
        raise ValueError("Lo stato deve contenere solo valori finiti.")

    sources = state.detach().to(
        device=reference.device,
        dtype=reference.dtype,
    )

    conductances = compute_conductances(
        sources.unsqueeze(0),
        recurrent,
    )

    # Da [batch, sorgente, destinazione]
    # a [sorgente, destinazione].
    return conductances[0].detach().cpu().clone()