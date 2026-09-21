from pathlib import Path

import torch

from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier


def load_inference_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SequenceClassifier, dict]:
    """Carica il modello e le impostazioni necessarie all'esecuzione."""
    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if bundle["format_version"] != 1:
        raise ValueError("Formato del file non supportato.")

    if bundle["num_classes"] != 2:
        raise ValueError("Questa configurazione di esecuzione è binaria.")

    model = SequenceClassifier(
        LTCConfig(**bundle["config"]),
        num_classes=bundle["num_classes"],
    )

    model.load_state_dict(bundle["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, bundle