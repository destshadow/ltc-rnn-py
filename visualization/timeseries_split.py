import matplotlib.pyplot as plt
import torch

from timeseries.dataset import Piece


COLORS = {"training": "tab:blue", "validation": "tab:orange", "test": "tab:green"}


def plot_split(starts, prices, segments, parts: dict[str, list[Piece]], *, title: str):
    """Prezzo (scala log) con le parti della divisione e rendimenti per parte.

    Le linee grigie tratteggiate segnano dove i dati mancanti spezzano la serie.
    """
    figure, (price_axis, return_axis) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True,
    )

    price_axis.plot(starts, prices.tolist(), color="0.3", linewidth=0.8)
    price_axis.set_yscale("log")
    price_axis.set_ylabel("Prezzo (scala log)")
    price_axis.set_title(title)

    for name, pieces in parts.items():
        times = [time for piece in pieces for time in piece.end_times]
        returns = torch.cat([piece.returns for piece in pieces])
        count = len(returns)
        price_axis.axvspan(times[0], times[-1], color=COLORS[name], alpha=0.12,
                           label=f"{name}: {times[0]:%Y-%m-%d} → {times[-1]:%Y-%m-%d} ({count})")
        return_axis.plot(times, (100 * returns).tolist(), ".", markersize=1,
                         color=COLORS[name])

    for segment_starts, _ in segments[1:]:
        for axis in (price_axis, return_axis):
            axis.axvline(segment_starts[0], color="0.5", linestyle="--", linewidth=0.8)

    price_axis.legend(loc="upper left")
    return_axis.set_ylabel("Rendimento giornaliero (%)")
    for axis in (price_axis, return_axis):
        axis.grid(alpha=0.25)

    return figure
