import matplotlib.pyplot as plt
import torch

from data.memory_grid import make_memory_grid
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

    class_zero_rows = []
    class_one_rows = []
    delays = []

    for amplitudes, delay, inputs, labels in make_memory_grid():
        logits = model(inputs.to(device), dt=checkpoint["dt"])
        probabilities = torch.softmax(logits, dim=1).cpu()

        correct_probabilities = probabilities.gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)

        class_zero_rows.append(correct_probabilities[0::2])
        class_one_rows.append(correct_probabilities[1::2])
        delays.append(delay)

    grids = (
        torch.stack(class_zero_rows).numpy(),
        torch.stack(class_one_rows).numpy(),
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, layout="constrained"
    )

    titles = (
        "Classe 0: positivo → negativo",
        "Classe 1: negativo → positivo",
    )

    for ax, grid, title in zip(axes, grids, titles):
        image = ax.pcolormesh(
            amplitudes.numpy(),
            delays,
            grid,
            shading="nearest",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
        )

        ax.axvline(0.8, color="black", linestyle="--", linewidth=1)
        ax.axhline(22, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Ampiezza degli impulsi")

    axes[0].set_ylabel("Passi di attesa dopo il secondo impulso")

    fig.colorbar(
        image,
        ax=axes,
        label="Probabilità assegnata alla classe corretta",
    )

    fig.suptitle("LTC addestrata — sensibilità ad ampiezza e attesa")
    plt.show()


if __name__ == "__main__":
    main()