"""Atomic checkpoint saving: the final file is complete, no temporary file left."""

import torch

from training.checkpoint import save_atomically


def test_save_replaces_previous_checkpoint_without_leftovers(tmp_path):
    path = tmp_path / "best.pt"
    save_atomically({"epoch": 1, "weights": torch.zeros(2)}, path)
    save_atomically({"epoch": 7, "weights": torch.ones(2)}, path)

    loaded = torch.load(path, weights_only=True)
    assert loaded["epoch"] == 7
    torch.testing.assert_close(loaded["weights"], torch.ones(2))
    assert sorted(item.name for item in tmp_path.iterdir()) == ["best.pt"]
