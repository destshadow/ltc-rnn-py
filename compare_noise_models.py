import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.decision import collect_margins, choose_threshold


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model = SequenceClassifier(
        LTCConfig(**checkpoint["config"]),
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def evaluate_model(path, inputs, labels, noises, levels):
    model, checkpoint = load_model(path)

    train_inputs, train_labels = make_event_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["training_seed"],
    )

    train_margins = collect_margins(
        model, train_inputs, dt=checkpoint["dt"]
    )
    threshold = choose_threshold(train_margins, train_labels)

    results = {}

    for level in levels:
        accuracies = []

        for noise in noises:
            margins = collect_margins(
                model,
                inputs + level * noise,
                dt=checkpoint["dt"],
            )

            predictions = (margins < threshold).long()
            accuracy = (
                (predictions == labels).double().mean().item() * 100
            )
            accuracies.append(accuracy)

        results[str(level)] = {
            "accuracies_percent": accuracies,
            "mean_accuracy_percent": sum(accuracies) / len(accuracies),
        }

    return {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "threshold": threshold,
        "epoch": checkpoint["epoch"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy-checkpoint", required=True)
    args = parser.parse_args()

    inputs, labels = make_event_order(pairs=128, seed=2026)

    levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    noise_seeds = [101, 202, 303]

    noises = [
        torch.randn(
            inputs.shape,
            generator=torch.Generator().manual_seed(seed),
            dtype=inputs.dtype,
        )
        for seed in noise_seeds
    ]

    print("Valutazione modello originale...", flush=True)
    original = evaluate_model(
        "outputs/event_order_best.pt",
        inputs, labels, noises, levels,
    )

    print("Valutazione modello candidato...", flush=True)
    augmented = evaluate_model(
        args.noisy_checkpoint,
        inputs, labels, noises, levels,
    )

    print(f"\nSoglia originale: {original['threshold']:.12f}")
    print(f"Soglia nuovo modello: {augmented['threshold']:.12f}")

    for level in levels:
        old = original["results"][str(level)]["mean_accuracy_percent"]
        new = augmented["results"][str(level)]["mean_accuracy_percent"]

        print(
            f"Rumore {level:.3f} | "
            f"originale {old:.2f}% | "
            f"nuovo {new:.2f}% | "
            f"differenza {new - old:+.2f} punti"
        )

    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path("outputs") / f"noise_comparison_{name}"
    directory.mkdir(parents=True, exist_ok=False)

    report = {
        "validation_seed": 2026,
        "validation_pairs": 128,
        "noise_seeds": noise_seeds,
        "threshold_source": "clean_training",
        "original": original,
        "augmented": augmented,
    }

    (directory / "results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Confronto salvato in:", directory)


if __name__ == "__main__":
    main()
