import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.decision import collect_margins, choose_threshold

def report(name, margins, labels, threshold):
    predictions = (margins < threshold).long()
    correct = (predictions == labels).sum().item()

    print(
        f"{name} | soglia {threshold:.9f} | "
        f"corrette {correct}/{len(labels)}"
    )

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

    train_inputs, train_labels = make_event_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["training_seed"],
    )

    val_inputs, val_labels = make_event_order(
        pairs=checkpoint["validation_pairs"],
        seed=checkpoint["validation_seed"],
    )

    train_margins = collect_margins(
        model, train_inputs, dt=checkpoint["dt"]
    )
    val_margins = collect_margins(
        model, val_inputs, dt=checkpoint["dt"]
    )

    threshold = choose_threshold(train_margins, train_labels)

    print("Decisione originale:")
    report("Train", train_margins, train_labels, 0.0)
    report("Val", val_margins, val_labels, 0.0)

    print("\nSoglia scelta esclusivamente sul training:")
    report("Train", train_margins, train_labels, threshold)
    report("Val", val_margins, val_labels, threshold)

    print("\nCheckpoint invariato.")


if __name__ == "__main__":
    main()