import torch


def make_temporal_order(
    pairs: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crea coppie di sequenze con ordine degli impulsi opposto."""
    if type(pairs) is not int:
        raise TypeError("pairs deve essere un numero intero.")

    if pairs <= 0:
        raise ValueError("pairs deve essere maggiore di zero.")

    generator = torch.Generator().manual_seed(seed)

    sequence_length = 30
    inputs = torch.zeros(2 * pairs, sequence_length, 1)
    labels = torch.zeros(2 * pairs, dtype=torch.long)

    for pair in range(pairs):
        first = int(torch.randint(3, 8, (1,), generator=generator).item())
        gap = int(torch.randint(4, 9, (1,), generator=generator).item())
        second = first + gap

        amplitude = float(
            torch.empty(1).uniform_(0.8, 1.2, generator=generator).item()
        )

        positive_first = 2 * pair
        negative_first = positive_first + 1

        inputs[positive_first, first, 0] = amplitude
        inputs[positive_first, second, 0] = -amplitude

        inputs[negative_first, first, 0] = -amplitude
        inputs[negative_first, second, 0] = amplitude

        labels[positive_first] = 0
        labels[negative_first] = 1

    return inputs, labels