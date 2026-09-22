from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from inference.controller_loading import (
    load_controller,
    save_controller,
)
from inference.controller_stream import ControllerStream
from ltc.config import LTCConfig
from models.continuous_controller import ContinuousController


def main():
    torch.manual_seed(42)

    model = ContinuousController(
        LTCConfig(input_size=3, hidden_size=8, substeps=6),
        output_size=1,
    )
    dt = 0.1
    samples = torch.randn(12, 3)

    original = ControllerStream(model, dt=dt)
    expected = torch.stack([
        original.push(sample) for sample in samples
    ])

    with TemporaryDirectory() as directory:
        path = Path(directory) / "controller.pt"

        save_controller(path, model, dt=dt)
        loaded, bundle = load_controller(path)

        assert bundle["dt"] == dt
        assert loaded.cell.config == model.cell.config
        assert not loaded.training

        restored = ControllerStream(loaded, dt=bundle["dt"])
        assert restored.state_snapshot() is None

        actual = torch.stack([
            restored.push(sample) for sample in samples
        ])

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            restored.state_snapshot(),
            original.state_snapshot(),
        )
        print("Modello ricaricato: stessi comandi e stesso stato finale.")

        try:
            save_controller(path, model, dt=dt)
        except FileExistsError:
            print("Sovrascrittura accidentale rifiutata.")
        else:
            raise AssertionError("Il file esistente è stato sovrascritto.")

    print("Controlli superati.")


if __name__ == "__main__":
    main()