import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from data.event_order import make_event_order
from inference.loading import load_inference_model
from inference.stream import LTCStream


def create_panels(sequence, hidden_size, dt):
    figure, axes = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True,
        constrained_layout=True,
    )

    duration = sequence.shape[0] * dt

    for axis in axes:
        axis.set_xlim(0, duration)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Ingresso")
    axes[0].set_ylim(-0.1, sequence.max().item() * 1.15)

    input_lines = [
        axes[0].step([], [], where="post", label=name)[0]
        for name in ("A", "B", "C")
    ]
    axes[0].legend(loc="upper right")

    axes[1].set_ylabel("Stato interno")
    state_lines = [
        axes[1].plot([], [], label=f"N{index}")[0]
        for index in range(hidden_size)
    ]
    axes[1].legend(loc="upper right", ncol=4)

    axes[2].set_ylabel("Margine rispetto\nalla soglia")
    axes[2].set_xlabel("Tempo del modello")
    axes[2].axhline(0, color="black", linestyle="--")
    decision_line, = axes[2].plot([], [], color="purple")
    decision_text = axes[2].text(
        0.02, 0.95, "",
        transform=axes[2].transAxes,
        va="top",
    )

    return (
        figure, axes, input_lines, state_lines,
        decision_line, decision_text,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="outputs/event_order_inference_1000ep.pt",
    )
    parser.add_argument(
        "--class-id", type=int, choices=(0, 1), default=0,
    )
    args = parser.parse_args()

    model, bundle = load_inference_model(
        args.bundle, device="cpu"
    )

    inputs, labels = make_event_order(pairs=1, seed=2026)
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
        f"LTC in streaming — sequenza attesa: {names[expected]}"
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

    # Il riferimento resta vivo fino alla chiusura della finestra.
    plt.show()


if __name__ == "__main__":
    main()