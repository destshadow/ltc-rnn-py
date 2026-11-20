"""Velocità di LTC e CfC: latenza di un passo (inferenza) e sequenze a lotti.

    python -m experiments.benchmark_cells
"""

import argparse
import json
from statistics import median
from time import perf_counter

import torch

from cfc.cell import CfcCell
from cfc.config import CfcConfig
from ltc.cell import LTCCell
from ltc.config import LTCConfig


def timed(function, device, repeats: int) -> float:
    """Mediana in millisecondi di `repeats` esecuzioni, dopo un riscaldamento."""
    for _ in range(3):
        function()
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1000 * (perf_counter() - start))
    return median(samples)


@torch.no_grad()
def measure(cell, device, *, inputs: int, hidden: int, steps: int, batch: int, repeats: int):
    cell = cell.to(device).eval()
    one, state = torch.randn(1, inputs, device=device), torch.zeros(1, hidden, device=device)
    many, states = torch.randn(batch, inputs, device=device), torch.zeros(batch, hidden, device=device)

    def sequence():
        current = states
        for _ in range(steps):
            current = cell(many, current, dt=0.1)

    return {
        "single_step_ms": timed(lambda: cell(one, state, dt=0.1), device, repeats),
        "sequence_ms": timed(sequence, device, repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(0)
    cells = {
        "ltc (6 sottopassi)": LTCCell(LTCConfig(input_size=2, hidden_size=args.hidden, substeps=6)),
        "cfc": CfcCell(CfcConfig(input_size=2, hidden_size=args.hidden)),
    }
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    results = {}
    for device in devices:
        for name, cell in cells.items():
            values = measure(cell, device, inputs=2, hidden=args.hidden, steps=args.steps,
                             batch=args.batch, repeats=args.repeats)
            results[f"{name} | {device.type}"] = values
            print(f"{name:20s} {device.type:4s} | un passo {values['single_step_ms']:.3f} ms | "
                  f"{args.steps} passi × {args.batch} {values['sequence_ms']:.2f} ms")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
