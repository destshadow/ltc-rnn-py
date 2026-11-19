import os
from pathlib import Path

import torch


def save_atomically(payload: dict, path: Path) -> None:
    """Salva con torch.save su un file temporaneo e poi lo sostituisce al finale.

    Un'interruzione durante la scrittura lascia intatto il checkpoint precedente.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
