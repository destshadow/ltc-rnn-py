import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider

from ltc.synapse_parameters import SynapseParameters
from .synapse_curve import sample_curve


def main():
    # Un collegamento: una sorgente e una destinazione.
    params = SynapseParameters(source_size=1, target_size=1)

    # Valutiamo il collegamento su 601 ingressi indipendenti.
    sources = torch.linspace(-6.0, 6.0, 601).unsqueeze(1)
    x_values = sources[:, 0].numpy()

    initial_curve = sample_curve(
        sources,
        params,
        threshold=0.0,
        slope=1.0,
        strength=1.0,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(bottom=0.35)

    line, = ax.plot(x_values, initial_curve, linewidth=2)
    midpoint, = ax.plot([0.0], [0.5], "o", color="darkorange")
    threshold_line = ax.axvline(0.0, color="gray", linestyle="--")

    ax.set(
        title="LTC — risposta di un singolo collegamento",
        xlabel="Valore della sorgente",
        ylabel="Conduttanza attiva",
        xlim=(-6.0, 6.0),
        ylim=(0.0, 2.1),
    )
    ax.grid(alpha=0.25)

    threshold_slider = Slider(
        fig.add_axes([0.22, 0.22, 0.65, 0.03]),
        "Soglia", -3.0, 3.0, valinit=0.0,
    )
    slope_slider = Slider(
        fig.add_axes([0.22, 0.15, 0.65, 0.03]),
        "Pendenza", 0.2, 8.0, valinit=1.0,
    )
    strength_slider = Slider(
        fig.add_axes([0.22, 0.08, 0.65, 0.03]),
        "Intensità", 0.1, 2.0, valinit=1.0,
    )

    def update(_):
        threshold = threshold_slider.val
        strength = strength_slider.val

        curve = sample_curve(
            sources,
            params,
            threshold=threshold,
            slope=slope_slider.val,
            strength=strength,
        )

        line.set_ydata(curve)
        midpoint.set_data([threshold], [strength * 0.5])
        threshold_line.set_xdata([threshold, threshold])
        fig.canvas.draw_idle()

    for slider in (threshold_slider, slope_slider, strength_slider):
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()