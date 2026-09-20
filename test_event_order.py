import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.decision import collect_margins, choose_threshold


def summarize(margins, labels, threshold):
    predictions = (margins < threshold).long()

    # Positivo quando il margine favorisce la classe corretta.
    centered = margins - threshold
    signed = torch.where(labels == 0, centered, -centered)

    return {
        "total": len(labels),
        "correct": (predictions == labels).sum().item(),
        "mean_signed_margin": signed.mean().item(),
        "minimum_signed_margin": signed.min().item(),
        "class_0_correct": (
            (predictions == labels) & (labels == 0)
        ).sum().item(),
        "class_1_correct": (
            (predictions == labels) & (labels == 1)
        ).sum().item(),
    }


def print_summary(name, result):
    accuracy = 100.0 * result["correct"] / result["total"]

    print(
        f"{name} | "
        f"{result['correct']}/{result['total']} ({accuracy:.2f}%) | "
        f"margine medio {result['mean_signed_margin']:+.6e} | "
        f"minimo {result['minimum_signed_margin']:+.6e}"
    )


def main():
    checkpoint_path = Path("outputs/event_order_best.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["num_classes"] != 2:
        raise ValueError("Questo esperimento richiede due classi.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SequenceClassifier(
        LTCConfig(**checkpoint["config"]),
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    train_inputs, train_labels = make_event_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["training_seed"],
    )
    train_margins = collect_margins(
        model, train_inputs, dt=checkpoint["dt"]
    )

    # La soglia viene scelta prima di generare e osservare il test.
    threshold = choose_threshold(train_margins, train_labels)

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = Path("outputs") / f"event_order_test_{run_name}"
    run_directory.mkdir(parents=True, exist_ok=False)

    protocol = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "threshold": threshold,
        "threshold_source": "training",
        "test_seed": 314159,
        "test_pairs": 1000,
        "dt": checkpoint["dt"],
        "config": checkpoint["config"],
        "torch_version": str(torch.__version__),
    }

    (run_directory / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    inputs, labels = make_event_order(
        pairs=protocol["test_pairs"],
        seed=protocol["test_seed"],
    )
    margins = collect_margins(model, inputs, dt=protocol["dt"])

    c_positions = inputs[:, :, 2].argmax(dim=1)
    delays = inputs.shape[1] - 1 - c_positions

    results = {
        "original": summarize(margins, labels, 0.0),
        "adjusted": summarize(margins, labels, threshold),
        "by_delay": {},
    }

    print(f"Soglia dal training: {threshold:.12f}")
    print_summary("Decisione originale", results["original"])
    print_summary("Decisione con soglia", results["adjusted"])

    for lower, upper in ((0, 9), (10, 19), (20, 30)):
        mask = (delays >= lower) & (delays <= upper)

        if not mask.any().item():
            continue

        name = f"{lower}-{upper}"
        group = summarize(margins[mask], labels[mask], threshold)
        results["by_delay"][name] = group
        print_summary(f"Attesa {name}", group)

    (run_directory / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("Risultati salvati in:", run_directory)


if __name__ == "__main__":
    main()