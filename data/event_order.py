import torch


def make_event_order(
    pairs: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Genera coppie A-B-C e B-A-C con tempi variabili."""
    if type(pairs) is not int:
        raise TypeError("pairs deve essere un numero intero.")

    if pairs <= 0:
        raise ValueError("pairs deve essere maggiore di zero.")

    generator = torch.Generator().manual_seed(seed)

    sequence_length = 48
    inputs = torch.zeros(2 * pairs, sequence_length, 3)
    labels = torch.tensor([0, 1], dtype=torch.long).repeat(pairs)

    for pair in range(pairs):
        delay = int(
            torch.randint(0, 31, (1,), generator=generator).item()
        )
        first_gap = int(
            torch.randint(3, 9, (1,), generator=generator).item()
        )
        second_gap = int(
            torch.randint(2, 7, (1,), generator=generator).item()
        )

        third = sequence_length - 1 - delay
        second = third - second_gap
        first = second - first_gap

        amplitude = float(
            torch.empty(1).uniform_(
                0.8, 1.2, generator=generator
            ).item()
        )

        abc = 2 * pair
        bac = abc + 1

        inputs[abc, first, 0] = amplitude
        inputs[abc, second, 1] = amplitude
        inputs[abc, third, 2] = amplitude

        inputs[bac, first, 1] = amplitude
        inputs[bac, second, 0] = amplitude
        inputs[bac, third, 2] = amplitude

    return inputs, labels