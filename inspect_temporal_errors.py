import torch

from data.temporal_order import make_temporal_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


@torch.no_grad()
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "outputs/tiny_checkpoint.pt",
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

    inputs, labels = make_temporal_order(pairs=500, seed=2026)

    dt = checkpoint["dt"]
    batch_size = 64
    error_count = 0

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(device)
        logits = model(batch, dt=dt)
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1)

        expected = labels[start:start + batch_size]
        mistakes = torch.nonzero(
            predictions != expected, as_tuple=True
        )[0]

        for local_index in mistakes.tolist():
            index = start + local_index
            signal = inputs[index, :, 0]

            positions = torch.nonzero(
                signal, as_tuple=True
            )[0]

            first = positions[0].item()
            second = positions[1].item()
            trailing_zeros = len(signal) - second - 1

            p0, p1 = probabilities[local_index].tolist()

            print(f"\nSequenza {index}")
            print(
                f"Classe attesa: {labels[index].item()} | "
                f"prevista: {predictions[local_index].item()}"
            )
            print(
                f"Primo impulso: indice {first}, "
                f"valore {signal[first].item():+.4f}"
            )
            print(
                f"Secondo impulso: indice {second}, "
                f"valore {signal[second].item():+.4f}"
            )
            print(f"Distanza tra gli impulsi: {second - first} passi")
            print(
                f"Attesa dopo il secondo impulso: "
                f"{trailing_zeros} passi ({trailing_zeros * dt:.2f})"
            )
            print(f"P(0): {p0:.6f} | P(1): {p1:.6f}")

            error_count += 1

    print(f"\nErrori totali: {error_count}/{len(inputs)}")


if __name__ == "__main__":
    main()