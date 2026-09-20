import torch

from data.temporal_order import make_temporal_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.evaluation import evaluate_classifier


def print_results(name, results):
    accuracy = 100.0 * results["correct"] / results["total"]

    print(f"\n{name}")
    print(f"Errore medio: {results['loss']:.6f}")
    print(
        f"Corrette: {results['correct']}/{results['total']} "
        f"({accuracy:.2f}%)"
    )
    print("Matrice: righe = classe vera, colonne = classe prevista")
    print(results["confusion"].tolist())


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "outputs/tiny_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )

    config = LTCConfig(**checkpoint["config"])

    model = SequenceClassifier(
        config,
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    dt = checkpoint["dt"]

    training_inputs, training_labels = make_temporal_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["data_seed"],
    )

    evaluation_inputs, evaluation_labels = make_temporal_order(
        pairs=500,
        seed=2026,
    )

    print("Dispositivo:", device)
    print("Checkpoint all'aggiornamento:", checkpoint["update"])

    training_results = evaluate_classifier(
        model, training_inputs, training_labels, dt=dt
    )
    print_results("Esempi usati per l'addestramento", training_results)

    evaluation_results = evaluate_classifier(
        model, evaluation_inputs, evaluation_labels, dt=dt
    )
    print_results("Insieme di valutazione separato", evaluation_results)


if __name__ == "__main__":
    main()