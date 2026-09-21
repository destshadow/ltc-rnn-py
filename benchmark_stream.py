import math
from statistics import median
from time import perf_counter

import torch

from data.event_order import make_event_order
from inference.decision import classify_binary
from inference.loading import load_inference_model
from inference.stream import LTCStream


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def process_sample(stream, sample, threshold):
    logits = stream.push(sample)

    return classify_binary(
        logits,
        threshold=threshold,
    ).item()


def benchmark(device_name):
    device = torch.device(device_name)

    model, bundle = load_inference_model(
        "outputs/event_order_inference.pt",
        device=device,
    )

    stream = LTCStream(model, dt=bundle["dt"])
    threshold = bundle["decision_threshold"]

    # I campioni rimangono sulla CPU, come dati appena acquisiti.
    inputs, _ = make_event_order(pairs=1, seed=123)

    # Riscaldamento: eseguiamo entrambe le sequenze senza misurare.
    for sequence in inputs:
        stream.reset()

        for sample in sequence:
            process_sample(stream, sample, threshold)

    synchronize(device)

    durations_ms = []

    for _ in range(3):
        for sequence in inputs:
            stream.reset()

            for sample in sequence:
                synchronize(device)
                start = perf_counter()

                process_sample(stream, sample, threshold)

                synchronize(device)
                elapsed_ms = (perf_counter() - start) * 1000.0
                durations_ms.append(elapsed_ms)

    ordered = sorted(durations_ms)
    p95_index = math.ceil(0.95 * len(ordered)) - 1

    print(f"\nDispositivo: {device}")
    print("Campioni misurati:", len(ordered))
    print(f"Mediana: {median(ordered):.3f} ms")
    print(f"95° percentile: {ordered[p95_index]:.3f} ms")
    print(f"Massimo osservato: {ordered[-1]:.3f} ms")

    if device.type == "cpu":
        print("Thread PyTorch CPU:", torch.get_num_threads())


def main():
    print("Latenza per un campione, inclusa la decisione.")
    print("Esclusi caricamento del modello e acquisizione del sensore.")

    benchmark("cpu")

    if torch.cuda.is_available():
        benchmark("cuda")
    else:
        print("\nGPU non disponibile: confronto saltato.")


if __name__ == "__main__":
    main()