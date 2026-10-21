import argparse

import torch

from data.event_order import make_event_order
from inference.loading import load_inference_model
from inference.session import SequenceSession
from inference.stream import LTCStream


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="outputs/event_order_inference_1000ep.pt",
    )
    args = parser.parse_args()

    model, bundle = load_inference_model(
        args.bundle, device="cpu"
    )
    inputs, _ = make_event_order(pairs=1, seed=2026)
    sequence = inputs[0]
    dt = bundle["dt"]

    session = SequenceSession(
        LTCStream(model, dt=dt),
        sequence,
    )

    assert session.current() is None
    assert session.processed == 0

    first = session.advance()
    assert first.index == 0
    assert first.time == dt
    assert session.processed == 1

    # Leggere più volte non deve consumare altri campioni.
    for _ in range(3):
        current = session.current()
        torch.testing.assert_close(current.state, first.state)
        torch.testing.assert_close(current.output, first.output)

    assert session.processed == 1
    print("Lettura del risultato: nessun avanzamento.")

    # Le fotografie restituite devono essere indipendenti.
    first.state.fill_(123.0)
    torch.testing.assert_close(
        session.current().state,
        current.state,
    )
    print("Fotografie: indipendenti dallo stato della sessione.")

    while not session.finished:
        session.advance()

    final = session.current()
    assert session.processed == len(sequence)
    assert final.index == len(sequence) - 1

    full_output = model(sequence.unsqueeze(0), dt=dt)[0]
    torch.testing.assert_close(
        final.output, full_output,
        rtol=1e-5, atol=1e-5,
    )
    print("Uscita finale: equivalente alla sequenza intera.")

    try:
        session.advance()
    except StopIteration:
        pass
    else:
        raise AssertionError("Fine sequenza non rispettata.")

    assert session.processed == len(sequence)
    print("Fine sequenza: nessun campione aggiuntivo.")

    session.reset()
    assert session.current() is None
    assert session.processed == 0

    while not session.finished:
        session.advance()

    repeated = session.current()
    torch.testing.assert_close(repeated.state, final.state)
    torch.testing.assert_close(repeated.output, final.output)
    print("Reset: riproduzione equivalente.")

    print("Controlli superati.")


if __name__ == "__main__":
    main()