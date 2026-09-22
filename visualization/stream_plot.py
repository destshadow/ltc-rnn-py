import matplotlib.pyplot as plt


def create_panels(sequence, hidden_size, dt):
    figure, axes = plt.subplots(
        3, 1,
        figsize=(15, 8),
        sharex=True,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(
        rect=(0, 0.12, 0.67, 0.82)
    )

    duration = sequence.shape[0] * dt

    for axis in axes:
        axis.set_xlim(0, duration)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Ingresso")
    minimum = min(0.0, sequence.min().item())
    maximum = max(0.0, sequence.max().item())
    padding = max(0.05, 0.1 * (maximum - minimum))
    axes[0].set_ylim(minimum - padding, maximum + padding)

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
