import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch

from inference.connection_inspection import recurrent_conductances


class NeuronConnections:
    """Mostra strength o conduttanze verso il nodo selezionato."""

    def __init__(
        self, figure, neuron_view, recurrent, *, mode="strength"
    ):
        if mode not in ("strength", "conductance"):
            raise ValueError("Modalità delle connessioni non supportata.")

        self.figure = figure
        self.view = neuron_view
        self.recurrent = recurrent
        self.mode = mode
        self.selected = 0
        self.arrows = []

        self.strengths = (
            recurrent.strength.detach().cpu().numpy().copy()
        )

        expected = (neuron_view.hidden_size, neuron_view.hidden_size)
        if self.strengths.shape != expected:
            raise ValueError("Forma delle connessioni non compatibile.")

        # Stessa scala per entrambe le modalità.
        # Ogni conduttanza è al massimo pari alla propria strength.
        self.scale = max(float(self.strengths.max()), 1e-12)
        self.values = self.strengths.copy()

        self.description = figure.text(
            0.71, 0.92, "",
            ha="left",
            va="top",
            fontsize=9,
            linespacing=1.4,
        )

        self.connection_id = figure.canvas.mpl_connect(
            "button_press_event",
            self._on_click,
        )

        self.reset()

    def reset(self):
        """Ripristina i valori relativi allo stato iniziale nullo."""
        self._set_state(torch.zeros(self.view.hidden_size))
        self.select(self.selected)

    def display(self, snapshot):
        """Aggiorna le frecce senza modificare lo stato della rete."""
        if self.mode == "conductance":
            self._set_state(snapshot.state)
            self._refresh()

    def _set_state(self, state):
        if self.mode == "strength":
            self.values = self.strengths.copy()
        else:
            self.values = recurrent_conductances(
                self.recurrent, state
            ).numpy()

    def _on_click(self, event):
        if event.inaxes is not self.view.axis or event.button != 1:
            return

        toolbar = getattr(self.figure.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return

        hit, details = self.view.nodes.contains(event)
        if not hit or len(details["ind"]) == 0:
            return

        self.select(int(details["ind"][0]))
        self.figure.canvas.draw_idle()

    def select(self, target):
        if not 0 <= target < self.view.hidden_size:
            raise ValueError("Indice del neurone non valido.")

        self.selected = target

        for _, arrow in self.arrows:
            arrow.remove()
        self.arrows.clear()

        positions = self.view.positions

        for source, origin in enumerate(positions):
            if source == target:
                continue

            arrow = FancyArrowPatch(
                posA=origin,
                posB=positions[target],
                arrowstyle="-|>",
                mutation_scale=12,
                color="#666666",
                alpha=0.65,
                shrinkA=18,
                shrinkB=18,
                zorder=1,
            )
            self.view.axis.add_patch(arrow)
            self.arrows.append((source, arrow))

        colors = ["#333333"] * self.view.hidden_size
        colors[target] = "#d89000"

        widths = np.full(self.view.hidden_size, 1.2)
        widths[target] = 3.0

        self.view.nodes.set_edgecolors(colors)
        self.view.nodes.set_linewidths(widths)

        self._refresh()

    def _refresh(self):
        target = self.selected

        for source, arrow in self.arrows:
            value = float(self.values[source, target])
            arrow.set_linewidth(0.6 + 3.0 * value / self.scale)

        label = (
            "strength appresa"
            if self.mode == "strength"
            else "conduttanza sullo stato mostrato"
        )
        self_value = float(self.values[target, target])

        self.description.set_text(
            f"Connessioni verso N{target}\n"
            f"Spessore: {label}\n"
            f"Autoconnessione: {self_value:.4f} (non disegnata)"
        )