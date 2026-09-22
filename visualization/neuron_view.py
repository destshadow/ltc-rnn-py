import numpy as np
from matplotlib.colors import Normalize


class NeuronView:
    """Mostra lo stato corrente dei neuroni della LTC."""

    def __init__(self, figure, hidden_size, color_limit=4.0):
        self.hidden_size = hidden_size

        self.axis = figure.add_axes([0.70, 0.27, 0.28, 0.43])
        self.axis.set_aspect("equal")
        self.axis.set_xlim(-1.6, 1.6)
        self.axis.set_ylim(-1.6, 1.6)
        self.axis.set_axis_off()

        # Disposizione circolare: non rappresenta strati successivi.
        angles = (
            np.pi / 2
            - np.arange(hidden_size) * 2 * np.pi / hidden_size
        )
        self.positions = np.column_stack((
            np.cos(angles),
            np.sin(angles),
        ))

        self.nodes = self.axis.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            c=np.zeros(hidden_size),
            cmap="coolwarm",
            norm=Normalize(
                vmin=-color_limit,
                vmax=color_limit,
                clip=True,
            ),
            s=950,
            edgecolors="#333333",
            linewidths=1.2,
            zorder=2,
        )

        self.value_labels = []

        for index, (x, y) in enumerate(self.positions):
            self.axis.text(
                x * 1.35,
                y * 1.35,
                f"N{index}",
                ha="center",
                va="center",
                fontsize=10,
            )

            label = self.axis.text(
                x, y, "0.00",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                zorder=3,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
            self.value_labels.append(label)

        color_axis = figure.add_axes([0.74, 0.22, 0.20, 0.025])
        colorbar = figure.colorbar(
            self.nodes,
            cax=color_axis,
            orientation="horizontal",
            extend="both",
        )
        colorbar.set_label("Stato interno — scala colore fissa")

        self.reset()

    def reset(self):
        self.nodes.set_array(np.zeros(self.hidden_size))

        for label in self.value_labels:
            label.set_text("0.00")

        self.axis.set_title("Neuroni LTC\nStato iniziale")

    def display(self, snapshot):
        values = snapshot.state.detach().cpu().numpy()

        if values.shape != (self.hidden_size,):
            raise ValueError("Forma dello stato non compatibile.")

        self.nodes.set_array(values)

        for label, value in zip(self.value_labels, values):
            label.set_text(f"{value:+.2f}")

        self.axis.set_title(
            f"Neuroni LTC\n"
            f"Campione {snapshot.index + 1} — t={snapshot.time:.2f}"
        )