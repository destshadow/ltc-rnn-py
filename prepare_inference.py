import hashlib
from pathlib import Path

import torch

from data.event_order import make_event_order
from ltc.config import LTCConfig
from models.sequence_classifier import SequenceClassifier
from training.decision import collect_margins, choose_threshold

import argparse


def main():

    parser = argparse.ArgumentParser(
        description="Prepara un modello LTC per l'inferenza."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.checkpoint
    output_path = args.output

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} esiste già: scegli un nuovo nome per l'esportazione."
        )

    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["num_classes"] != 2:
        raise ValueError("Questa esportazione richiede due classi.")

    device = torch.device("cpu")

    model = SequenceClassifier(
        LTCConfig(**checkpoint["config"]),
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    inputs, labels = make_event_order(
        pairs=checkpoint["training_pairs"],
        seed=checkpoint["training_seed"],
    )

    margins = collect_margins(
        model, inputs, dt=checkpoint["dt"]
    )
    threshold = choose_threshold(margins, labels)

    bundle = {
        "format_version": 1,
        "task": checkpoint["task"],
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint["config"],
        "num_classes": checkpoint["num_classes"],
        "dt": checkpoint["dt"],
        "decision_threshold": threshold,
        "threshold_source": "training",
        "training_seed": checkpoint["training_seed"],
        "training_pairs": checkpoint["training_pairs"],
        "source_checkpoint": str(source_path),
        "source_epoch": checkpoint["epoch"],
        "threshold_device": str(device),
        "torch_version": str(torch.__version__),
        "source_checkpoint_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("xb") as output_file:
        torch.save(bundle, output_file)

    print(f"Soglia salvata: {threshold:.12f}")
    print("File per l'esecuzione:", output_path)


if __name__ == "__main__":
    main()
