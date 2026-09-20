import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from ltc.sequence import run_sequence
from models.sequence_classifier import SequenceClassifier


@torch.no_grad()
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "outputs/event_order_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = SequenceClassifier(
        LTCConfig(**checkpoint["config"]),
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    inputs, labels = make_event_order(
        pairs=checkpoint["validation_pairs"],
        seed=checkpoint["validation_seed"],
    )
    inputs = inputs.to(device)
    labels = labels.to(device)

    history, final_state = run_sequence(
        model.cell,
        inputs,
        dt=checkpoint["dt"],
    )

    # Ogni coppia occupa due righe consecutive.
    abc_states = history[0::2]
    bac_states = history[1::2]

    distances = torch.linalg.vector_norm(
        abc_states - bac_states,
        dim=2,
    )

    # Il canale 2 contiene un solo evento C positivo per sequenza.
    c_positions = inputs[0::2, :, 2].argmax(dim=1)
    pair_indices = torch.arange(len(c_positions), device=device)

    before_c = distances[pair_indices, c_positions - 1]
    after_c = distances[pair_indices, c_positions]
    at_end = distances[:, -1]

    delays = inputs.shape[1] - 1 - c_positions

    print("Epoca del checkpoint:", checkpoint["epoch"])
    print("Coppie analizzate:", len(c_positions))

    print("\nDistanza media tra gli stati delle coppie:")
    print(f"Prima di C: {before_c.mean().item():.6e}")
    print(f"Dopo C:     {after_c.mean().item():.6e}")
    print(f"Alla fine:  {at_end.mean().item():.6e}")

    print("\nDistanze medie raggruppate per attesa dopo C:")

    for lower, upper in ((0, 9), (10, 19), (20, 30)):
        mask = (delays >= lower) & (delays <= upper)
        count = mask.sum().item()

        if count == 0:
            continue

        print(
            f"Attesa {lower:2d}-{upper:2d} | coppie {count:3d} | "
            f"prima C {before_c[mask].mean().item():.6e} | "
            f"dopo C {after_c[mask].mean().item():.6e} | "
            f"fine {at_end[mask].mean().item():.6e}"
        )

    logits = model.readout(final_state)

    # Positivo: favorisce classe 0. Negativo: favorisce classe 1. Margine vicino a zero: i punteggi sono quasi uguali.
    margins = logits[:, 0] - logits[:, 1]

    abc_margins = margins[0::2]
    bac_margins = margins[1::2]
    pair_gaps = abc_margins - bac_margins

    print("\nMargini finali dello strato di uscita:")

    for name, values in (("A-B-C", abc_margins), ("B-A-C", bac_margins)):
        print(
            f"{name} | minimo {values.min().item():+.6e} | "
            f"medio {values.mean().item():+.6e} | "
            f"massimo {values.max().item():+.6e}"
        )

    correctly_ranked = (pair_gaps > 0).sum().item()
    correct = (logits.argmax(dim=1) == labels).sum().item()

    print(
        "\nCoppie con margine A-B-C maggiore di B-A-C:",
        f"{correctly_ranked}/{len(pair_gaps)}",
    )
    print(
        "Differenza media dei margini nelle coppie:",
        f"{pair_gaps.mean().item():+.6e}",
    )
    print("Classificazioni corrette:", f"{correct}/{len(labels)}")


if __name__ == "__main__":
    main()