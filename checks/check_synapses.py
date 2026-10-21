import torch

from ltc.synapse_parameters import SynapseParameters
from ltc.synapses import compute_conductances, compute_synaptic_effects


def main():
    torch.manual_seed(42)

    params = SynapseParameters(3, 8).to("cuda")

    # Impostiamo un caso controllato senza registrare queste modifiche
    # nel percorso dei gradienti.
    with torch.no_grad():
        params.threshold.zero_()
        params.reversal_potential.fill_(1.0)

    sources = torch.zeros(2, 3, device="cuda")

    conductances = compute_conductances(sources, params)
    drive, total = compute_synaptic_effects(sources, params)

    assert tuple(conductances.shape) == (2, 3, 8)
    assert tuple(drive.shape) == (2, 8)
    assert tuple(total.shape) == (2, 8)

    expected = (params.strength * 0.5).unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(conductances, expected)

    # Con tutti i reversal_potential uguali a +1, drive e totale coincidono.
    torch.testing.assert_close(drive, total)

    print("Forme corrette.")
    print("Sorgente sulla soglia: apertura al 50% verificata.")

    # Cambiamo soltanto la seconda sequenza.
    sources[1] = 1.0
    changed = compute_conductances(sources, params)

    torch.testing.assert_close(changed[0], conductances[0])
    assert torch.all(changed[1] > conductances[1]).item()

    print("Seconda sequenza: conduttanze aumentate.")
    print("Prima sequenza: invariata.")

    # Verifichiamo che il calcolo mantenga i gradienti verso gli ingressi.
    differentiable_sources = torch.full(
        (2, 3), 0.25, device="cuda", requires_grad=True
    )

    _, total = compute_synaptic_effects(differentiable_sources, params)
    total.sum().backward()

    gradient = differentiable_sources.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all().item()
    assert torch.all(gradient > 0).item()

    print("Gradienti verso gli ingressi verificati.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()