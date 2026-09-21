import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from data.event_order import make_event_order
from inference.loading import load_inference_model
from inference.stream import LTCStream
from visualization.stream_plot import create_panels
from visualization.playback import PlaybackControls

import math

from data.noise import add_gaussian_noise


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

    (
        figure, axes, input_lines, state_lines,
        decision_line, decision_text,
    ) = create_panels(sequence, hidden_size, dt)

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
    artists = [
        *input_lines, *state_lines,
        decision_line, decision_text,
    ]

    controls = None

    def initialize():
        stream.reset()
        state_times[:] = [0.0]
        states[:] = [torch.zeros(hidden_size)]
        decision_times.clear()
        margins.clear()

        for line in [*input_lines, *state_lines, decision_line]:
            line.set_data([], [])

        decision_text.set_text("In attesa del primo campione.")
        return artists

    def update(index):
        logits = stream.push(sequence[index])
        snapshot = stream.state_snapshot()

        if snapshot is None:
            raise RuntimeError("Stato assente dopo il campione.")

        end_time = (index + 1) * dt
        state_times.append(end_time)
        states.append(snapshot[0].cpu())

        # Positivo: classe 0. Negativo: classe 1.
        margin = (
            (logits[0, 0] - logits[0, 1]).double().item()
            - threshold
        )
        decision_times.append(end_time)
        margins.append(margin)

        # Ogni ingresso è mantenuto durante il suo intervallo dt.
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
        final = index == len(sequence) - 1
        stage = "Finale" if final else "Provvisoria"

        decision_text.set_text(
            f"{stage}: {names[prediction]} | "
            f"campione {index + 1}/{len(sequence)}"
        )

        if controls is not None:
            controls.mark_frame(index)

        return artists

    animation = FuncAnimation(
        figure,
        update,
        frames=len(sequence),
        init_func=initialize,
        interval=100,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )

    controls = PlaybackControls(
        figure=figure,
        animation=animation,
        initialize=initialize,
        update=update,
        frame_count=len(sequence),
    )

    # Il riferimento resta vivo fino alla chiusura della finestra.
    plt.show()


if __name__ == "__main__":
    main()
