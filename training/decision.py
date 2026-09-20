import torch

@torch.no_grad()
def collect_margins(model, inputs, *, dt, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    collected = []

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(device)
        logits = model(batch, dt=dt)

        margins = logits[:, 0] - logits[:, 1]
        collected.append(margins.cpu().double())

    return torch.cat(collected)



def choose_threshold(margins, labels):
    values = torch.unique(margins, sorted=True)

    midpoints = values[:-1] + (values[1:] - values[:-1]) * 0.5

    candidates = torch.cat([
        values[:1] - 1.0,
        midpoints,
        values[-1:] + 1.0,
    ])

    # Sotto la soglia: classe 1. Altrimenti: classe 0.
    predictions = (
        margins.unsqueeze(0) < candidates.unsqueeze(1)
    ).long()

    correct = (predictions == labels.unsqueeze(0)).sum(dim=1)

    best_indices = torch.nonzero(
        correct == correct.max(), as_tuple=True
    )[0]

    selected = best_indices[len(best_indices) // 2]
    return candidates[selected].item()
