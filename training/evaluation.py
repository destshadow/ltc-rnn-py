import torch
from torch import nn


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    dt: float,
    batch_size: int = 64,
) -> dict:
    """Valuta il classificatore binario senza aggiornare i parametri."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size deve essere un intero positivo.")

    if len(inputs) == 0 or labels.shape != (len(inputs),):
        raise ValueError("Servono ingressi non vuoti e un'etichetta per esempio.")

    model.eval()
    device = next(model.parameters()).device
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    confusion = torch.zeros(2, 2, dtype=torch.long)

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start:start + batch_size].to(device)
        batch_labels = labels[start:start + batch_size].to(device)

        logits = model(batch_inputs, dt=dt)

        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Il modello ha prodotto punteggi non finiti.")

        total_loss += loss_function(logits, batch_labels).item()

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == batch_labels).sum().item()

        indices = (batch_labels * 2 + predictions).cpu()
        confusion += torch.bincount(indices, minlength=4).reshape(2, 2)

    return {
        "loss": total_loss / len(inputs),
        "correct": total_correct,
        "total": len(inputs),
        "confusion": confusion,
    }