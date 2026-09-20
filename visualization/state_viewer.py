import matplotlib.pyplot as plt
import torch

from ltc.cell import LTCCell
from ltc.config import LTCConfig
from ltc.sequence import run_sequence


def main():
    torch.manual_seed(42)

    config = LTCConfig(input_size=1, hidden_size=8, substeps=6)
    cell = LTCCell(config)
    cell.eval()

    dt = 0.1
    steps = 150

    pulse = torch.zeros(1, steps, 1)
    pulse[:, 20:50, :] = 1.0

    baseline = torch.zeros_like(pulse)

    with torch.no_grad():
        pulse_history, _ = run_sequence(cell, pulse, dt=dt)
        baseline_history, _ = run_sequence(cell, baseline, dt=dt)

    # Aggiungiamo lo stato iniziale per disegnare anche l'istante zero.
    initial = cell.initial_state(batch_size=1)
    pulse_states = torch.cat(
        [initial.unsqueeze(1), pulse_history], dim=1
    )
    baseline_states = torch.cat(
        [initial.unsqueeze(1), baseline_history], dim=1
    )

    difference = torch.linalg.vector_norm(
        pulse_states - baseline_states, dim=2
    )[0]

    times = torch.arange(steps + 1, dtype=torch.float32) * dt
    signal = torch.cat([pulse[0, :, 0], pulse[0, -1:, 0]])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].step(times.numpy(), signal.numpy(), where="post")
    axes[0].set_ylabel("Ingresso")
    axes[0].set_title("LTC non addestrata — risposta a un impulso")

    for neuron in range(config.hidden_size):
        axes[1].plot(
            times.numpy(),
            pulse_states[0, :, neuron].numpy(),
            label=f"N{neuron}",
        )

    axes[1].set_ylabel("Stato")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(times.numpy(), difference.numpy(), color="darkorange")
    axes[2].set_ylabel("Differenza\ntra gli stati")
    axes[2].set_xlabel("Tempo simulato")

    for ax in axes:
        ax.axvspan(2.0, 5.0, color="steelblue", alpha=0.12)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()