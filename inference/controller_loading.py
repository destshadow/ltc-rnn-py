from dataclasses import asdict
from pathlib import Path

import torch

from ltc.config import LTCConfig
from ltc.validation import validate_duration
from models.continuous_controller import ContinuousController


SUPPORTED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def _validate_parameters(parameters, dtype):
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("model_state_dict deve essere un dizionario non vuoto.")
    for name, value in parameters.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model_state_dict deve associare nomi a tensori.")
        if value.dtype != dtype:
            raise ValueError(f"Tipo numerico incoerente per {name}.")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Parametro non finito: {name}.")


def save_controller(
    path,
    model: ContinuousController,
    *,
    dt: float,
) -> None:
    """Salva una copia dei parametri e della configurazione."""
    validate_duration(dt)
    dtype_name = str(model.readout.weight.dtype)
    if dtype_name not in SUPPORTED_DTYPES:
        raise ValueError("Tipo numerico non supportato: usare float32 o float64.")

    # Copie su CPU: il file non dipende dalla GPU di origine.
    parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    _validate_parameters(parameters, SUPPORTED_DTYPES[dtype_name])

    bundle = {
        "format_version": 1,
        "model_type": "ltc_continuous_controller",
        "config": asdict(model.cell.config),
        "output_size": model.readout.out_features,
        "output_activation": "sigmoid",
        "dt": float(dt),
        "dtype": dtype_name,
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

    if not isinstance(bundle, dict):
        raise ValueError("Il bundle deve essere un dizionario.")
    required = {
        "format_version", "model_type", "config", "output_size",
        "output_activation", "dt", "dtype", "model_state_dict",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Campi mancanti nel bundle: {', '.join(sorted(missing))}.")

    if type(bundle["format_version"]) is not int or bundle["format_version"] != 1:
        raise ValueError("Versione del formato non supportata.")

    if bundle["model_type"] != "ltc_continuous_controller":
        raise ValueError("Il file non contiene un controllore LTC.")

    if bundle["output_activation"] != "sigmoid":
        raise ValueError("Attivazione di uscita non supportata.")

    if not isinstance(bundle["dtype"], str) or bundle["dtype"] not in SUPPORTED_DTYPES:
        raise ValueError("Tipo numerico non supportato.")

    try:
        validate_duration(bundle["dt"])
    except TypeError as error:
        raise ValueError(f"dt non valido: {error}") from error
    if not isinstance(bundle["config"], dict):
        raise ValueError("config deve essere un dizionario.")
    try:
        config = LTCConfig(**bundle["config"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Configurazione LTC non valida: {error}") from error
    if type(bundle["output_size"]) is not int or bundle["output_size"] <= 0:
        raise ValueError("output_size deve essere un intero positivo.")

    dtype = SUPPORTED_DTYPES[bundle["dtype"]]
    _validate_parameters(bundle["model_state_dict"], dtype)

    model = ContinuousController(
        config,
        output_size=bundle["output_size"],
    ).to(dtype=dtype)

    try:
        model.load_state_dict(bundle["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Pesi incompatibili con il controller: {error}") from error

    model = model.to(device)
    model.eval()

    return model, bundle
