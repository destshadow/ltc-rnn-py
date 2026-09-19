import torch

from ltc.config import LTCConfig
from ltc.state import create_initial_state


def main():
    config = LTCConfig(input_size=3, hidden_size=8)

    state = create_initial_state(
        config,
        batch_size=2,
        device="cuda",
    )

    print("Forma:", tuple(state.shape))
    print("Dispositivo:", state.device)
    print("Tipo numerico:", state.dtype)

    assert tuple(state.shape) == (2, 8)
    assert state.device.type == "cuda"
    assert torch.count_nonzero(state).item() == 0

    # Modifichiamo solo il primo neurone della prima sequenza.
    state[0, 0] = 1.0

    print("Prima sequenza:", state[0].cpu().tolist())
    print("Seconda sequenza:", state[1].cpu().tolist())

    assert torch.count_nonzero(state[1]).item() == 0

    # Una nuova chiamata deve creare uno stato nuovo e azzerato.
    fresh_state = create_initial_state(
        config,
        batch_size=2,
        device="cuda",
    )

    assert torch.count_nonzero(fresh_state).item() == 0

    print("Controlli superati.")


if __name__ == "__main__":
    main()