import pytest
import torch

from inference.controller_loading import load_controller, save_controller
from inference.controller_stream import ControllerStream
from ltc.config import LTCConfig
from models.continuous_controller import ContinuousController


def make_controller(dtype=torch.float32):
    torch.manual_seed(42)
    return ContinuousController(LTCConfig(2, 4), output_size=1).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roundtrip_commands_state_and_reset(tmp_path, dtype):
    model = make_controller(dtype)
    path = tmp_path / "controller.pt"
    save_controller(path, model, dt=0.1)
    restored, metadata = load_controller(path)
    assert next(restored.parameters()).dtype == dtype
    assert not restored.training
    original = ControllerStream(model, dt=0.1)
    loaded = ControllerStream(restored, dt=metadata["dt"])
    samples = torch.randn(8, 2, dtype=dtype)
    expected = torch.stack([original.push(sample) for sample in samples])
    actual = torch.stack([loaded.push(sample) for sample in samples])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(loaded.state_snapshot(), original.state_snapshot())
    loaded.reset()
    assert loaded.state_snapshot() is None
    torch.testing.assert_close(
        torch.stack([loaded.push(sample) for sample in samples]), expected,
    )
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_controller(path, model, dt=0.1)
    assert path.read_bytes() == before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unsupported_save_does_not_create_file(tmp_path, dtype):
    path = tmp_path / "new" / "controller.pt"
    with pytest.raises(ValueError, match="Tipo numerico"):
        save_controller(path, make_controller(dtype), dt=0.1)
    assert not path.parent.exists()


def test_mixed_dtype_save_is_rejected(tmp_path):
    model = make_controller()
    model.cell.double()
    with pytest.raises(ValueError, match="Tipo numerico incoerente"):
        save_controller(tmp_path / "controller.pt", model, dt=0.1)


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "valid.pt"
    save_controller(path, make_controller(), dt=0.1)
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize("field,value", [
    ("format_version", 2), ("format_version", True),
    ("model_type", "classifier"), ("output_activation", "tanh"),
    ("dtype", "torch.float16"), ("dtype", []),
    ("config", []), ("config", {"input_size": 0, "hidden_size": 4}),
    ("output_size", True), ("output_size", 0), ("dt", 0.0), ("dt", "0.1"),
    ("model_state_dict", {}),
])
def test_invalid_metadata(tmp_path, bundle, field, value):
    bundle[field] = value
    path = tmp_path / "invalid.pt"
    torch.save(bundle, path)
    with pytest.raises(ValueError):
        load_controller(path)


def test_missing_field(tmp_path, bundle):
    del bundle["dt"]
    path = tmp_path / "invalid.pt"
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="Campi mancanti.*dt"):
        load_controller(path)


@pytest.mark.parametrize("fault", ["nan", "dtype", "shape", "missing"])
def test_invalid_weights(tmp_path, bundle, fault):
    weights = bundle["model_state_dict"]
    key = "readout.weight"
    if fault == "nan":
        weights[key].fill_(float("nan"))
    elif fault == "dtype":
        weights[key] = weights[key].double()
    elif fault == "shape":
        weights[key] = torch.zeros(1, 99)
    else:
        del weights[key]
    path = tmp_path / "invalid.pt"
    torch.save(bundle, path)
    with pytest.raises(ValueError):
        load_controller(path)
