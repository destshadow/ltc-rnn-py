from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.epoch import train_epoch
from training.evaluation import evaluate_classifier

import csv
from datetime import datetime, timezone

from data.noise import add_gaussian_noise


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = LTCConfig(input_size=3, hidden_size=8, substeps=6)
    model = SequenceClassifier(config).to(device)

    training_inputs, training_labels = make_event_order(
        pairs=128, seed=123
    )
    validation_inputs, validation_labels = make_event_order(
        pairs=128, seed=2026
    )

    training_inputs = training_inputs.to(device)
    training_labels = training_labels.to(device)

    training_noise_std = 0.0
    validation_noise_std = 0.01

    training_noise_generator = torch.Generator().manual_seed(707)
    validation_noise_generator = torch.Generator().manual_seed(808)

    noisy_validation_inputs = add_gaussian_noise(
        validation_inputs,
        std=validation_noise_std,
        generator=validation_noise_generator,
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dt = 0.1
    max_epochs = 1000
    best_validation_loss = float("inf")
    best_epoch = 0

    run_name = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    run_directory = Path("outputs") / f"event_order_noise_{run_name}"
    run_directory.mkdir(parents=True, exist_ok=False)

    checkpoint_path = run_directory / "best.pt"
    metrics_path = run_directory / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            "epoch",
            "train_clean_loss",
            "train_clean_correct",
            "validation_clean_loss",
            "validation_clean_correct",
            "validation_noisy_loss",
            "validation_noisy_correct",
            "selection_loss",
        ])

    print("Cartella esperimento:", run_directory)

    print("Dispositivo:", device)
    print("Sequenze di addestramento:", len(training_labels))
    print("Sequenze di validazione:", len(validation_labels))
    print("Parametri:", sum(p.numel() for p in model.parameters()))

    for epoch in range(max_epochs + 1):
        if epoch > 0:
            noisy_training_inputs = add_gaussian_noise(
                training_inputs,
                std=training_noise_std,
                generator=training_noise_generator,
            )

            train_epoch(
                model,
                optimizer,
                loss_function,
                noisy_training_inputs,
                training_labels,
                dt=dt,
                batch_size=64,
            )

        if epoch % 10 != 0:
            continue

        training = evaluate_classifier(
            model, training_inputs, training_labels, dt=dt
        )
        validation = evaluate_classifier(
            model, validation_inputs, validation_labels, dt=dt
        )

        noisy_validation = evaluate_classifier(
            model,
            noisy_validation_inputs,
            validation_labels,
            dt=dt,
        )

        selection_loss = (
            validation["loss"] + noisy_validation["loss"]
        ) / 2.0

        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                epoch,
                training["loss"],
                training["correct"],
                validation["loss"],
                validation["correct"],
                noisy_validation["loss"],
                noisy_validation["correct"],
                selection_loss,
            ])

        print(
            f"Epoca {epoch:3d} | "
            f"train pulito {training['loss']:.4f}, "
            f"{training['correct']}/{training['total']} | "
            f"val pulita {validation['loss']:.4f}, "
            f"{validation['correct']}/{validation['total']} | "
            f"val rumore {noisy_validation['loss']:.4f}, "
            f"{noisy_validation['correct']}/{noisy_validation['total']}",
            flush=True,
        )

        if selection_loss < best_validation_loss:
            best_validation_loss = selection_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "num_classes": 2,
                    "dt": dt,
                    "epoch": epoch,
                    "task": "event_order_abc",
                    "model_seed": 42,
                    "training_seed": 123,
                    "validation_seed": 2026,
                    "training_pairs": 128,
                    "validation_pairs": 128,
                    "training_loss": training["loss"],
                    "validation_loss": validation["loss"],
                    "validation_correct": validation["correct"],
                    "training_noise_std": training_noise_std,
                    "validation_noise_std": validation_noise_std,
                    "training_noise_seed": 707,
                    "validation_noise_seed": 808,
                    "validation_noisy_loss": noisy_validation["loss"],
                    "validation_noisy_correct": noisy_validation["correct"],
                    "selection_loss": selection_loss,
                    "learning_rate": 0.01,
                    "batch_size": 64,
                    "torch_version": str(torch.__version__),
                },
                checkpoint_path,
            )

    print(f"\nMigliore epoca valutata: {best_epoch}")
    print(f"Checkpoint salvato in: {checkpoint_path}")
    print(f"Errore medio di selezione: {best_validation_loss:.6f}")


if __name__ == "__main__":
    main()
