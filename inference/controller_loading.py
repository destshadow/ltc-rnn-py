from dataclasses import asdict
from pathlib import Path

import torch

from ltc.config import LTCConfig
from ltc.validation import validate_duration
from models.continuous_controller import ContinuousController


def save_controller(
    path,
    model: ContinuousController,
    *,
    dt: float,
) -> None:
    """Salva una copia dei parametri e della configurazione."""
    validate_duration(dt)

    # Copie su CPU: il file non dipende dalla GPU di origine.
    parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    if not all(
        torch.isfinite(value).all().item()
        for value in parameters.values()
    ):
        raise ValueError("Il modello contiene parametri non finiti.")

    bundle = {
        "format_version": 1,
        "model_type": "ltc_continuous_controller",
        "config": asdict(model.cell.config),
        "output_size": model.readout.out_features,
        "output_activation": "sigmoid",
        "dt": float(dt),
        "dtype": str(model.readout.weight.dtype),
        "model_state_dict": parameters,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Evita di sostituire accidentalmente un modello già salvato.
    with path.open("xb") as file:
        torch.save(bundle, file)


def load_controller(path, *, device="cpu"):
    """Carica un controllore pronto per l'inferenza."""
    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if bundle["format_version"] != 1:
        raise ValueError("Versione del formato non supportata.")

    if bundle["model_type"] != "ltc_continuous_controller":
        raise ValueError("Il file non contiene un controllore LTC.")

    if bundle["output_activation"] != "sigmoid":
        raise ValueError("Attivazione di uscita non supportata.")

    supported_dtypes = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    if bundle["dtype"] not in supported_dtypes:
        raise ValueError("Tipo numerico non supportato.")

    validate_duration(bundle["dt"])

    model = ContinuousController(
        LTCConfig(**bundle["config"]),
        output_size=bundle["output_size"],
    ).to(dtype=supported_dtypes[bundle["dtype"]])

    model.load_state_dict(bundle["model_state_dict"])

    if not all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    ):
        raise ValueError("Il file contiene parametri non finiti.")

    model = model.to(device)
    model.eval()

    return model, bundle