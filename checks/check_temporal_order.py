import torch

from data.temporal_order import make_temporal_order


def main():
    inputs, labels = make_temporal_order(pairs=4, seed=123)

    assert tuple(inputs.shape) == (8, 30, 1)
    assert tuple(labels.shape) == (8,)
    assert labels.dtype == torch.long

    assert torch.all(inputs[:, -1, :] == 0).item()

    nonzero_counts = (inputs != 0).sum(dim=(1, 2))
    assert torch.all(nonzero_counts == 2).item()

    for index in range(0, len(inputs), 2):
        positive_first = inputs[index, :, 0]
        negative_first = inputs[index + 1, :, 0]

        torch.testing.assert_close(negative_first, -positive_first)

        positions = torch.nonzero(
            positive_first, as_tuple=True
        )[0]

        assert positive_first[positions[0]].item() > 0
        assert positive_first[positions[1]].item() < 0
        assert labels[index].item() == 0
        assert labels[index + 1].item() == 1

    repeated_inputs, repeated_labels = make_temporal_order(
        pairs=4, seed=123
    )

    assert torch.equal(inputs, repeated_inputs)
    assert torch.equal(labels, repeated_labels)

    print("Forma ingressi:", tuple(inputs.shape))
    print("Etichette:", labels.tolist())
    print("Due impulsi per sequenza, ultimo ingresso sempre zero.")
    print("Ordine e coppie verificati.")
    print("Generazione riproducibile.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()