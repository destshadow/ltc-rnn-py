import argparse

import matplotlib.pyplot as plt
import torch
from inference.session import SequenceSession

from data.event_order import make_event_order
from inference.loading import load_inference_model
from inference.stream import LTCStream
from visualization.stream_plot import create_panels
from visualization.playback import PlaybackControls

import math

from data.noise import add_gaussian_noise

from visualization.neuron_view import NeuronView
from visualization.neuron_connections import NeuronConnections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="outputs/event_order_inference_1000ep.pt",
    )
    parser.add_argument(
        "--class-id", type=int, choices=(0, 1), default=0,
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=1101)

    parser.add_argument(
        "--connections",
        choices=("strength", "conductance"),
        default="strength",
        help="Valore rappresentato dallo spessore delle connessioni.",
    )

    args = parser.parse_args()

    if not math.isfinite(args.noise) or args.noise < 0:
        parser.error("--noise deve essere finito e non negativo.")

    model, bundle = load_inference_model(
        args.bundle, device="cpu"
    )

    inputs, labels = make_event_order(
        pairs=1,
        seed=args.seed,
    )

    if args.noise > 0:
        generator = torch.Generator().manual_seed(args.noise_seed)
        inputs = add_gaussian_noise(
            inputs,
            std=args.noise,
            generator=generator,
        )

    sequence = inputs[args.class_id]
    expected = labels[args.class_id].item()

    dt = bundle["dt"]
    threshold = bundle["decision_threshold"]
    hidden_size = model.cell.config.hidden_size
    stream = LTCStream(model, dt=dt)
    session = SequenceSession(stream, sequence)

    (
        figure, axes, input_lines, state_lines,
        decision_line, decision_text,
    ) = create_panels(sequence, hidden_size, dt)

    neuron_view = NeuronView(
        figure,
        hidden_size=hidden_size,
    )

    connections = NeuronConnections(
        figure,
        neuron_view,
        model.cell.recurrent,
        mode=args.connections,
    )

    names = {0: "A-B-C", 1: "B-A-C"}
    figure.suptitle(
        f"LTC in streaming — attesa: {names[expected]} | "
        f"seme: {args.seed} | rumore: {args.noise:g}"
    )

    # Lo stato iniziale è quello nullo usato dalla nostra cella.
    state_times = [0.0]
    states = [torch.zeros(hidden_size)]
    decision_times = []
    margins = []

    def clear_plot():
        """Azzera soltanto la visualizzazione."""
        state_times[:] = [0.0]
        states[:] = [torch.zeros(hidden_size)]
        decision_times.clear()
        margins.clear()

        for line in [*input_lines, *state_lines, decision_line]:
            line.set_data([], [])

        decision_text.set_text("In attesa del primo campione.")
        neuron_view.reset()
        connections.reset()

    def display_snapshot(snapshot):
        """Aggiunge ai grafici un risultato già calcolato."""
        index = snapshot.index

        state_times.append(snapshot.time)
        states.append(snapshot.state)
        decision_times.append(snapshot.time)

        margin = (
            (snapshot.output[0] - snapshot.output[1]).double().item()
            - threshold
        )
        margins.append(margin)

        # Il campione viene mantenuto nel suo intervallo dt.
        edges = torch.arange(index + 2).numpy() * dt
        visible = sequence[:index + 1]
        held = torch.cat((visible, visible[-1:]), dim=0)

        for channel, line in enumerate(input_lines):
            line.set_data(edges, held[:, channel].numpy())

        history = torch.stack(states).numpy()
        for neuron, line in enumerate(state_lines):
            line.set_data(state_times, history[:, neuron])

        decision_line.set_data(decision_times, margins)

        for axis in axes[1:]:
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        prediction = 0 if margin >= 0 else 1
        stage = (
            "Finale" if index + 1 == len(sequence)
            else "Provvisoria"
        )

        decision_text.set_text(
            f"{stage}: {names[prediction]} | "
            f"campione {index + 1}/{len(sequence)}"
        )
        neuron_view.display(snapshot)
        connections.display(snapshot)

    controls = PlaybackControls(
        figure=figure,
        session=session,
        on_snapshot=display_snapshot,
        on_reset=clear_plot,
        interval_ms=100,
    )

    clear_plot()
    controls.start()
    plt.show()

    # Il riferimento resta vivo fino alla chiusura della finestra.
    plt.show()


if __name__ == "__main__":
    main()
