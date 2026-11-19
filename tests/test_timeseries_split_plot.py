"""Split plot: builds without a display and shows one span per part."""

from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from timeseries.calendar import split_segments  # noqa: E402
from timeseries.dataset import returns_by_segment, split_pieces  # noqa: E402
from visualization.timeseries_split import plot_split  # noqa: E402


def test_plot_split_has_price_and_return_panels():
    day = timedelta(days=1)
    starts = [datetime(2024, 1, 1) + k * day for k in range(60) if k != 30]
    prices = torch.linspace(100, 160, len(starts), dtype=torch.float64)
    segments = split_segments(starts, prices, period=day, calendar="continuous")
    parts = split_pieces(returns_by_segment(segments, period=day),
                         validation_fraction=0.2, test_fraction=0.2)

    figure = plot_split(starts, prices, segments,
                        dict(zip(("training", "validation", "test"), parts)), title="t")

    price_axis, return_axis = figure.axes
    assert len(price_axis.patches) == 3          # one shaded span per part
    assert len(return_axis.lines) == 3 + 1       # returns per part + one segment break
    assert price_axis.get_yscale() == "log"
