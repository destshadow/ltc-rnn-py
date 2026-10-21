from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.epoch import train_epoch
from training.evaluation import evaluate_classifier


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

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dt = 0.1
    max_epochs = 200
    best_validation_loss = float("inf")
    best_epoch = 0

    checkpoint_path = Path("outputs/event_order_best.pt")
    checkpoint_path.parent.mkdir(exist_ok=True)

    print("Dispositivo:", device)
    print("Sequenze di addestramento:", len(training_labels))
    print("Sequenze di validazione:", len(validation_labels))
    print("Parametri:", sum(p.numel() for p in model.parameters()))

    for epoch in range(max_epochs + 1):
        if epoch > 0:
            train_epoch(
                model,
                optimizer,
                loss_function,
                training_inputs,
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

        print(
            f"Epoca {epoch:3d} | "
            f"train {training['loss']:.4f}, "
            f"{training['correct']}/{training['total']} | "
            f"val {validation['loss']:.4f}, "
            f"{validation['correct']}/{validation['total']}",
            flush=True,
        )

        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
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
                },
                checkpoint_path,
            )

    print(f"\nMigliore epoca valutata: {best_epoch}")
    print(f"Errore di validazione: {best_validation_loss:.6f}")
    print(f"Checkpoint salvato in: {checkpoint_path}")


if __name__ == "__main__":
    main()