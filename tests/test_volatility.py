"""Volatility targets: alignment, no look-ahead into the window, no crossing gaps."""

from datetime import datetime, timedelta

import pytest
import torch

from timeseries.dataset import Piece
from timeseries.volatility import volatility_dataset
from timeseries.windows import Standardizer

DAY = timedelta(days=1)


def piece(values, start=datetime(2024, 1, 1)):
    values = torch.tensor(values, dtype=torch.float64)
    return Piece(values, torch.ones(len(values), dtype=torch.float64),
                 [start + k * DAY for k in range(len(values))])


def test_targets_are_future_and_persistence_is_past():
    returns = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07]
    scaler = Standardizer.fit(torch.tensor(returns, dtype=torch.float64))
    inputs, targets, persistence = volatility_dataset([piece(returns)], scaler=scaler,
                                                      lookback=3, horizon=2)

    assert inputs.shape == (3, 3, 2)                       # 7 − 3 − 2 + 1 windows
    absolute = [abs(value) for value in returns]
    for k in range(3):
        window_end = k + 3                                 # first future index
        assert targets[k].item() == pytest.approx(sum(absolute[window_end:window_end + 2]) / 2)
        assert persistence[k].item() == pytest.approx(sum(absolute[window_end - 2:window_end]) / 2)
        torch.testing.assert_close(scaler.inverse(inputs[k, :, 0].double()),
                                   torch.tensor(returns[k:k + 3], dtype=torch.float64),
                                   rtol=0, atol=1e-6)


def test_windows_never_join_two_pieces():
    # The second piece has huge returns: no target may mix them with the first piece.
    first, second = piece([0.01] * 6), piece([0.5] * 6, start=datetime(2024, 2, 1))
    scaler = Standardizer.fit(torch.tensor([0.01, 0.02, 0.5, 0.4], dtype=torch.float64))
    _, targets, persistence = volatility_dataset([first, second], scaler=scaler,
                                                 lookback=3, horizon=2)

    assert sorted(set(targets.tolist())) == pytest.approx([0.01, 0.5])
    assert sorted(set(persistence.tolist())) == pytest.approx([0.01, 0.5])
    assert len(targets) == 2 * (6 - 3 - 2 + 1)


@pytest.mark.parametrize("lookback,horizon", [(3, 0), (0, 1), (2, 3)])
def test_invalid_sizes_are_rejected(lookback, horizon):
    scaler = Standardizer.fit(torch.tensor([0.01, 0.02], dtype=torch.float64))
    with pytest.raises(ValueError):
        volatility_dataset([piece([0.01] * 10)], scaler=scaler, lookback=lookback, horizon=horizon)


def test_short_pieces_give_a_clear_error():
    scaler = Standardizer.fit(torch.tensor([0.01, 0.02], dtype=torch.float64))
    with pytest.raises(ValueError, match="abbastanza lungo"):
        volatility_dataset([piece([0.01] * 4)], scaler=scaler, lookback=3, horizon=2)


def test_log_ratio_round_trip_and_zero_correction_is_persistence():
    from timeseries.volatility import from_log_ratio, to_log_ratio

    targets = torch.tensor([0.01, 0.03, 0.02], dtype=torch.float64)
    persistence = torch.tensor([0.02, 0.01, 0.02], dtype=torch.float64)
    ratio = to_log_ratio(targets, persistence)

    torch.testing.assert_close(ratio, torch.log(targets / persistence))
    torch.testing.assert_close(from_log_ratio(ratio, persistence), targets)
    torch.testing.assert_close(from_log_ratio(torch.zeros(3), persistence), persistence)
    # A zero window does not produce infinities.
    assert torch.isfinite(to_log_ratio(torch.zeros(1), torch.zeros(1))).all()
    with pytest.raises(ValueError):
        to_log_ratio(targets, persistence[:2])
