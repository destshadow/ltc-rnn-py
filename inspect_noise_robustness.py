import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.event_order import make_event_order
from inference.loading import load_inference_model
from training.decision import collect_margins


def count_correct(model, inputs, labels, *, dt, threshold):
    margins = collect_margins(model, inputs, dt=dt)
    predictions = (margins < threshold).long()

    return (predictions == labels).sum().item()


def main():
    model, bundle = load_inference_model(
        "outputs/event_order_inference.pt",
        device="cpu",
    )

    # Riutilizziamo la validazione del progetto, non il test finale.
    inputs, labels = make_event_order(pairs=128, seed=2026)

    dt = bundle["dt"]
    threshold = bundle["decision_threshold"]

    noise_seeds = [101, 202, 303]
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    noise_samples = []

    for seed in noise_seeds:
        generator = torch.Generator().manual_seed(seed)

        noise_samples.append(
            torch.randn(
                inputs.shape,
                generator=generator,
                dtype=inputs.dtype,
            )
        )

    results = []

    for level in noise_levels:
        counts = []

        for noise in noise_samples:
            noisy_inputs = inputs + level * noise

            correct = count_correct(
                model,
                noisy_inputs,
                labels,
                dt=dt,
                threshold=threshold,
            )
            counts.append(correct)

        accuracies = [
            100.0 * count / len(labels)
            for count in counts
        ]
        mean_accuracy = sum(accuracies) / len(accuracies)

        print(
            f"Rumore {level:.3f} | "
            f"media {mean_accuracy:.2f}% | "
            f"min {min(accuracies):.2f}% | "
            f"max {max(accuracies):.2f}%",
            flush=True,
        )

        results.append({
            "noise_std": level,
            "correct_counts": counts,
            "mean_accuracy_percent": mean_accuracy,
        })

    run_name = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    directory = Path("outputs") / f"noise_validation_{run_name}"
    directory.mkdir(parents=True, exist_ok=False)

    report = {
        "source_checkpoint_sha256": bundle["source_checkpoint_sha256"],
        "threshold": threshold,
        "dt": dt,
        "validation_seed": 2026,
        "validation_pairs": 128,
        "noise_seeds": noise_seeds,
        "results": results,
    }

    (directory / "results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Risultati salvati in:", directory)


if __name__ == "__main__":
    main()