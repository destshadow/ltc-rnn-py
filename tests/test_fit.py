"""Shared training loop: epoch choice, final-epoch mode and checkpoint callback."""

import pytest
import torch
from torch import nn

from ltc.config import LTCConfig
from timeseries.evaluation import predict
from timeseries.model import ReturnForecaster
from training.fit import fit_and_select


class Log:
    def __init__(self):
        self.rows = []

    def write(self, epoch, train_loss, validation_loss):
        self.rows.append((epoch, train_loss, validation_loss))


@pytest.mark.parametrize("final_epoch", [False, True])
def test_choice_matches_the_logged_validation(final_epoch):
    torch.manual_seed(0)
    model = ReturnForecaster(LTCConfig(2, 2, substeps=1))
    data = (torch.randn(32, 4, 2), torch.randn(32))
    log, saved = Log(), []

    epoch, state = fit_and_select(
        model, torch.optim.Adam(model.parameters(), lr=0.05), nn.MSELoss(), data, data,
        dt=0.1, epochs=4, batch_size=8, final_epoch=final_epoch, metrics=log, predict=predict,
        save_best=lambda state, epoch, loss: saved.append((epoch, loss)),
    )

    assert [row[0] for row in log.rows] == [0, 1, 2, 3, 4] and log.rows[0][1] is None
    losses = [row[2] for row in log.rows]
    expected = 4 if final_epoch else min(range(5), key=losses.__getitem__)
    assert epoch == expected and saved[-1][0] == expected
    assert saved[-1][1] == pytest.approx(losses[expected])
    if final_epoch:
        assert [entry[0] for entry in saved] == [0, 1, 2, 3, 4]   # saved every epoch
    assert state.keys() == model.state_dict().keys()
