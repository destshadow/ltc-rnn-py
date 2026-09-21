import math

import torch


def add_gaussian_noise(
    inputs: torch.Tensor,
    *,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Aggiunge rumore gaussiano generato sulla CPU."""
    if not math.isfinite(std) or std < 0:
        raise ValueError("std deve essere finita e non negativa.")

    if not inputs.is_floating_point():
        raise TypeError("Gli ingressi devono essere in virgola mobile.")

    noise = torch.randn(
        inputs.shape,
        generator=generator,
        dtype=inputs.dtype,
        device="cpu",
    ).to(inputs.device)

    return inputs + std * noise