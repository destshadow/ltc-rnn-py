import torch
from torch import nn

from data.temporal_order import make_temporal_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


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

    logits = model(inputs, dt=0.1)

    assert tuple(logits.shape) == (8, 2)
    assert torch.isfinite(logits).all().item()

    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, labels)

    assert torch.isfinite(loss).item()

    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Gradiente assente: {name}"
        assert torch.isfinite(parameter.grad).all().item(), name

    for group_name, module in (
        ("LTC", model.cell),
        ("Uscita", model.readout),
    ):
        magnitude = sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
        )

        assert magnitude > 0, f"Gradienti tutti nulli: {group_name}"

    predictions = logits.argmax(dim=1)
    total = sum(parameter.numel() for parameter in model.parameters())

    print("Dispositivo:", device)
    print("Numeri apprendibili:", total)
    print("Forma dei punteggi:", tuple(logits.shape))
    print("Etichette:", labels.cpu().tolist())
    print("Previsioni iniziali:", predictions.cpu().tolist())
    print(f"Errore iniziale: {loss.item():.6f}")
    print("Gradienti verificati nella LTC e nello strato di uscita.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()