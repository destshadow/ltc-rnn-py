import torch

from ltc.cell import LTCCell
from ltc.config import LTCConfig


def main():
    torch.manual_seed(42)

    config = LTCConfig(input_size=3, hidden_size=8, substeps=6)
    cell = LTCCell(config)

    total = sum(parameter.numel() for parameter in cell.parameters())
    assert total == 376
    print("Numeri apprendibili:", total)

    inputs_cpu = torch.randn(2, config.input_size)
    state_cpu = cell.initial_state(batch_size=2)

    assert tuple(state_cpu.shape) == (2, 8)
    assert state_cpu.device.type == "cpu"

    with torch.no_grad():
        output_cpu = cell(inputs_cpu, state_cpu, dt=0.1)

    assert tuple(output_cpu.shape) == (2, 8)
    assert torch.isfinite(output_cpu).all().item()

    # La cella non deve modificare lo stato ricevuto.
    torch.testing.assert_close(
        state_cpu, torch.zeros_like(state_cpu)
    )

    print("Esecuzione CPU: corretta.")
    print("Stato ricevuto: non modificato.")

    if torch.cuda.is_available():
        cell = cell.to("cuda")
        inputs_gpu = inputs_cpu.to("cuda")
        state_gpu = cell.initial_state(batch_size=2)

        assert state_gpu.device.type == "cuda"
        assert state_gpu.dtype == cell.neurons.resting_potential.dtype

        with torch.no_grad():
            output_gpu = cell(inputs_gpu, state_gpu, dt=0.1)

        torch.testing.assert_close(
            output_cpu,
            output_gpu.cpu(),
            rtol=1e-4,
            atol=1e-6,
        )

        print("Esecuzione GPU: coerente con la CPU.")
    else:
        print("Confronto GPU saltato: CUDA non disponibile.")

    print("Controlli superati.")


if __name__ == "__main__":
    main()