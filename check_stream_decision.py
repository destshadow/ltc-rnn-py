import torch

from data.event_order import make_event_order
from inference.decision import classify_binary
from inference.stream import LTCStream
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


@torch.no_grad()
def main():
    bundle = torch.load(
        "outputs/event_order_inference.pt",
        map_location="cpu",
        weights_only=True,
    )

    if bundle["format_version"] != 1:
        raise ValueError("Formato del file non supportato.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SequenceClassifier(
        LTCConfig(**bundle["config"]),
        num_classes=bundle["num_classes"],
    )
    model.load_state_dict(bundle["model_state_dict"])
    model = model.to(device)
    model.eval()

    inputs, labels = make_event_order(pairs=2, seed=2026)
    inputs = inputs.to(device)
    threshold = bundle["decision_threshold"]

    full_logits = model(inputs, dt=bundle["dt"])
    full_predictions = classify_binary(
        full_logits, threshold=threshold
    ).cpu()

    stream = LTCStream(model, dt=bundle["dt"])
    streamed_predictions = []

    for sequence in inputs:
        stream.reset()

        for sample in sequence:
            logits = stream.push(sample)

        prediction = classify_binary(
            logits, threshold=threshold
        ).item()

        streamed_predictions.append(prediction)

    streamed_predictions = torch.tensor(streamed_predictions)

    assert torch.equal(streamed_predictions, full_predictions)
    assert torch.equal(streamed_predictions, labels)

    print("Etichette:", labels.tolist())
    print("Sequenze intere:", full_predictions.tolist())
    print("Streaming:", streamed_predictions.tolist())
    print("Decisione con soglia salvata: verificata.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()