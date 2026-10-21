import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.decision import collect_margins, choose_threshold

import argparse

from data.noise import add_gaussian_noise


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint da valutare.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["num_classes"] != 2:
        raise ValueError("Questo esperimento richiede due classi.")

    # CPU anche per scegliere la soglia: coerente con il confronto.
    device = torch.device("cpu")

    model = SequenceClassifier(
        LTCConfig(**checkpoint["config"]),
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    train_inputs, train_labels = make_event_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["training_seed"],
    )
    train_margins = collect_margins(
        model, train_inputs, dt=checkpoint["dt"]
    )
    threshold = choose_threshold(train_margins, train_labels)

    run_name = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    run_directory = Path("outputs") / f"event_order_test_{run_name}"
    run_directory.mkdir(parents=True, exist_ok=False)

    protocol = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "threshold_source": "clean_training",
        "test_seed": 271828,
        "test_pairs": 1000,
        "noise_levels": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
        "noise_seeds": [1101, 2202, 3303],
        "noise_type": "independent_additive_gaussian",
        "delay_groups": [[0, 9], [10, 19], [20, 30]],
        "dt": checkpoint["dt"],
        "config": checkpoint["config"],
        "device": str(device),
        "torch_version": str(torch.__version__),
    }

    used_data_seeds = {
        checkpoint["training_seed"],
        checkpoint["validation_seed"],
    }
    if protocol["test_seed"] in used_data_seeds:
        raise ValueError("Il seme del test deve essere separato.")

    # Salviamo il protocollo prima di osservare i risultati.
    (run_directory / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    inputs, labels = make_event_order(
        pairs=protocol["test_pairs"],
        seed=protocol["test_seed"],
    )

    # Le posizioni degli eventi si ricavano dai dati puliti.
    c_positions = inputs[:, :, 2].argmax(dim=1)
    delays = inputs.shape[1] - 1 - c_positions

    print("Dispositivo:", device)
    print("Epoca del checkpoint:", checkpoint["epoch"])
    print(f"Soglia dal training: {threshold:.12f}")

    results = {"evaluations": []}

    for std in protocol["noise_levels"]:
        # Il caso pulito si valuta una sola volta.
        seeds = [None] if std == 0.0 else protocol["noise_seeds"]

        for seed in seeds:
            if seed is None:
                observed_inputs = inputs
            else:
                generator = torch.Generator().manual_seed(seed)
                observed_inputs = add_gaussian_noise(
                    inputs,
                    std=std,
                    generator=generator,
                )

            margins = collect_margins(
                model, observed_inputs, dt=protocol["dt"]
            )

            if not torch.isfinite(margins).all().item():
                raise RuntimeError("Punteggi non finiti nel test.")

            decisions = evaluate_decisions(
                margins, labels, delays, threshold
            )

            results["evaluations"].append({
                "noise_std": std,
                "noise_seed": seed,
                "decisions": decisions,
            })

            seed_name = "nessuno" if seed is None else str(seed)
            print(f"\nRumore {std:.3f} | seme {seed_name}")

            for key, title in (
                ("original", "Originale"),
                ("adjusted", "Con soglia"),
            ):
                result = decisions[key]
                print_summary(title, result)

                for name, group in result["by_delay"].items():
                    print_summary(
                        f"  {title}, attesa {name}", group
                    )

            # Conserviamo anche i risultati parziali.
            (run_directory / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )

    print("\nRisultati salvati in:", run_directory)

def evaluate_decisions(margins, labels, delays, threshold):
    results = {}

    for name, boundary in (
        ("original", 0.0),
        ("adjusted", threshold),
    ):
        result = summarize(margins, labels, boundary)
        result["by_delay"] = {}

        for lower, upper in ((0, 9), (10, 19), (20, 30)):
            mask = (delays >= lower) & (delays <= upper)

            if mask.any().item():
                group_name = f"{lower}-{upper}"
                result["by_delay"][group_name] = summarize(
                    margins[mask], labels[mask], boundary
                )

        results[name] = result

    return results


if __name__ == "__main__":
    main()