import torch

from inference.controller_stream import ControllerStream
from ltc.config import LTCConfig
from models.continuous_controller import ContinuousController


@torch.no_grad()
def main():
    torch.manual_seed(42)

    model = ContinuousController(
        LTCConfig(input_size=3, hidden_size=8, substeps=6),
        output_size=1,
    )
    model.eval()

    samples = torch.randn(12, 3)
    dt = 0.1

    # Riferimento: gestione esplicita dello stato.
    state = model.initial_state(batch_size=1)
    expected = []

    for sample in samples:
        commands, state = model(
            sample.unsqueeze(0),
            state,
            dt=dt,
        )
        expected.append(commands[0])

    expected = torch.stack(expected)

    # Stessi campioni attraverso l'interfaccia streaming.
    stream = ControllerStream(model, dt=dt)
    actual = torch.stack([stream.push(sample) for sample in samples])

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(stream.state_snapshot(), state)
    print("Streaming equivalente alla gestione esplicita dello stato.")

    snapshot = stream.state_snapshot()
    snapshot.fill_(999.0)
    torch.testing.assert_close(stream.state_snapshot(), state)
    print("Fotografia indipendente dalla memoria interna.")

    # Un campione non valido deve essere rifiutato senza avanzare.
    before = stream.state_snapshot()

    try:
        stream.push(torch.full((3,), float("nan")))
    except ValueError:
        pass
    else:
        raise AssertionError("Il campione non valido è stato accettato.")

    torch.testing.assert_close(stream.state_snapshot(), before)
    print("Campione non valido rifiutato, memoria invariata.")

    stream.reset()
    assert stream.state_snapshot() is None

    replay = torch.stack([stream.push(sample) for sample in samples])
    torch.testing.assert_close(replay, expected)
    print("Reset: riproduzione equivalente.")

    print("Controlli superati.")


if __name__ == "__main__":
    main()