import torch

from data.event_order import make_event_order


def main():
    inputs, labels = make_event_order(pairs=4, seed=123)

    assert tuple(inputs.shape) == (8, 48, 3)
    assert labels.tolist() == [0, 1] * 4

    observed_delays = []

    for index in range(0, len(inputs), 2):
        abc = inputs[index]
        bac = inputs[index + 1]

        # Ogni coppia differisce soltanto per lo scambio dei canali A e B.
        torch.testing.assert_close(
            abc, bac[:, [1, 0, 2]]
        )

        positions = torch.nonzero(
            abc.sum(dim=1) > 0, as_tuple=True
        )[0]

        assert positions.numel() == 3

        first, second, third = positions.tolist()

        assert abc[positions].argmax(dim=1).tolist() == [0, 1, 2]
        assert bac[positions].argmax(dim=1).tolist() == [1, 0, 2]

        # Un solo canale attivo per ogni evento.
        assert torch.all(
            (abc[positions] != 0).sum(dim=1) == 1
        ).item()

        assert 3 <= second - first <= 8
        assert 2 <= third - second <= 6

        delay = len(abc) - third - 1
        assert 0 <= delay <= 30
        observed_delays.append(delay)

        # Da C in poi, le due sequenze sono identiche.
        torch.testing.assert_close(abc[third:], bac[third:])

        # Stessa quantità complessiva di A, B e C.
        torch.testing.assert_close(
            abc.sum(dim=0), bac.sum(dim=0)
        )

    repeated_inputs, repeated_labels = make_event_order(
        pairs=4, seed=123
    )

    assert torch.equal(inputs, repeated_inputs)
    assert torch.equal(labels, repeated_labels)

    print("Forma ingressi:", tuple(inputs.shape))
    print("Etichette:", labels.tolist())
    print("Attese dopo C:", observed_delays)
    print("Ordine A-B-C / B-A-C verificato.")
    print("Ultimo evento e parte finale identici in ogni coppia.")
    print("Somme per canale identiche in ogni coppia.")
    print("Generazione riproducibile.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()