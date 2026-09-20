import torch


def make_memory_grid():
    """Varia ampiezza e attesa, mantenendo fisse le posizioni."""
    amplitudes = torch.linspace(0.6, 1.2, 25)
    delays = list(range(0, 41, 2))

    first = 3
    second = 7

    for delay in delays:
        length = second + 1 + delay

        inputs = torch.zeros(2 * len(amplitudes), length, 1)
        labels = torch.tensor([0, 1]).repeat(len(amplitudes))

        for index, amplitude in enumerate(amplitudes):
            positive_first = 2 * index
            negative_first = positive_first + 1

            inputs[positive_first, first, 0] = amplitude
            inputs[positive_first, second, 0] = -amplitude

            inputs[negative_first, first, 0] = -amplitude
            inputs[negative_first, second, 0] = amplitude

        yield amplitudes, delay, inputs, labels