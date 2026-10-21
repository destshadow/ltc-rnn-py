import torch

from data.event_order import make_event_order
from inference.stream import LTCStream
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


@torch.no_grad()
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
    model.eval()

    inputs, _ = make_event_order(pairs=1, seed=123)
    samples = inputs[0].to(device)
    dt = checkpoint["dt"]

    expected = model(samples.unsqueeze(0), dt=dt)

    stream = LTCStream(model, dt=dt)

    stream.push_block(samples[:7])
    stream.push_block(samples[7:20])
    from_blocks = stream.push_block(samples[20:])

    torch.testing.assert_close(
        from_blocks, expected, rtol=1e-4, atol=1e-6
    )
    print("Blocchi: risultato equivalente alla sequenza intera.")

    stream.reset()

    for sample in samples:
        from_samples = stream.push(sample)

    torch.testing.assert_close(
        from_samples, expected, rtol=1e-4, atol=1e-6
    )
    print("Singoli campioni: risultato equivalente.")

    stream.reset()
    after_reset = stream.push_block(samples)

    torch.testing.assert_close(
        after_reset, expected, rtol=1e-4, atol=1e-6
    )
    print("Reset: ripartenza corretta.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()