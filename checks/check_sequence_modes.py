import torch

from data.event_order import make_event_order
from ltc.cell import LTCCell
from ltc.config import LTCConfig
from ltc.sequence import run_sequence


def calculate(cell, inputs, collect_history):
    cell.zero_grad(set_to_none=True)

    history, final_state = run_sequence(
        cell,
        inputs,
        dt=0.1,
        collect_history=collect_history,
    )

    final_state.square().mean().backward()

    gradients = {}

    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all().item(), name
        gradients[name] = parameter.grad.detach().clone()

    return history, final_state.detach().clone(), gradients


def main():
    torch.manual_seed(42)

    cell = LTCCell(
        LTCConfig(input_size=3, hidden_size=8, substeps=6)
    )

    inputs, _ = make_event_order(pairs=2, seed=123)

    history, full_state, full_gradients = calculate(
        cell, inputs, collect_history=True
    )

    no_history, final_state, final_gradients = calculate(
        cell, inputs, collect_history=False
    )

    assert history is not None
    assert tuple(history.shape) == (4, 48, 8)
    assert no_history is None

    torch.testing.assert_close(full_state, final_state)

    for name in full_gradients:
        torch.testing.assert_close(
            full_gradients[name],
            final_gradients[name],
        )

    print("Modalità completa: storia disponibile.")
    print("Modalità finale: storia non raccolta.")
    print("Stati finali equivalenti.")
    print("Gradienti equivalenti.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()