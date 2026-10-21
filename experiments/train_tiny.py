import torch
from torch import nn

from data.temporal_order import make_temporal_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.step import train_step

from dataclasses import asdict
from pathlib import Path

def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = LTCConfig(input_size=1, hidden_size=8, substeps=6)
    model = SequenceClassifier(config).to(device)

    inputs, labels = make_temporal_order(pairs=4, seed=123)
    inputs = inputs.to(device)
    labels = labels.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dt = 0.1
    max_updates = 1000

    print("Dispositivo:", device)
    print("Esempi di addestramento:", len(labels))

    for update in range(max_updates + 1):
        if update > 0:
            train_step(
                model,
                optimizer,
                loss_function,
                inputs,
                labels,
                dt=dt,
            )

        if update % 20 == 0:
            model.eval()

            with torch.no_grad():
                logits = model(inputs, dt=dt)
                loss = loss_function(logits, labels).item()
                predictions = logits.argmax(dim=1)
                correct = (predictions == labels).sum().item()

            print(
                f"Aggiornamento {update:3d} | "
                f"errore {loss:.6f} | "
                f"corrette {correct}/{len(labels)}",
                flush=True,
            )

            if correct == len(labels) and loss < 0.05:
                print("Obiettivo sul piccolo insieme raggiunto.")
                break

    print("Etichette:", labels.cpu().tolist())
    print("Previsioni finali:", predictions.cpu().tolist())

    #qui
    model.eval()

    with torch.no_grad():
        final_logits = model(inputs, dt=dt)
        final_loss = loss_function(final_logits, labels).item()
        probabilities = torch.softmax(final_logits, dim=1).cpu()
        final_predictions = final_logits.argmax(dim=1).cpu()

    expected_labels = labels.cpu()
    final_correct = (final_predictions == expected_labels).sum().item()

    print("\nDettaglio delle sequenze:")

    for index in range(len(expected_labels)):
        expected = expected_labels[index].item()
        predicted = final_predictions[index].item()
        p0, p1 = probabilities[index].tolist()

        print(
            f"Sequenza {index} | "
            f"attesa {expected} | prevista {predicted} | "
            f"P(0) {p0:.4f} | P(1) {p1:.4f}"
        )

    output_directory = Path("outputs")
    output_directory.mkdir(exist_ok=True)
    checkpoint_path = output_directory / "tiny_checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "num_classes": 2,
            "dt": dt,
            "update": update,
            "model_seed": 42,
            "data_seed": 123,
            "training_pairs": 4,
            "training_loss": final_loss,
            "training_correct": final_correct,
        },
        checkpoint_path,
    )

    print(f"\nModello salvato in: {checkpoint_path}")


if __name__ == "__main__":
    main()