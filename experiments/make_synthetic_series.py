"""Scrive una serie sintetica AR(1) in CSV, da usare con experiments.train_timeseries.

Controllo di sicurezza: se phi ≠ 0 la LTC deve battere i riferimenti (MSE
standardizzato atteso ≈ 1 − phi²); se phi = 0 non deve riuscirci.

    python -m experiments.make_synthetic_series --phi 0.3
    python -m experiments.train_timeseries --csv outputs/synthetic/ar1_phi0.3_seed0.csv --epochs 20
"""

import argparse
from pathlib import Path

from timeseries.synthetic import ar1_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phi", type=float, required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()

    starts, prices = ar1_prices(args.count, phi=args.phi, sigma=args.sigma, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"ar1_phi{args.phi:g}_seed{args.seed}.csv"
    lines = ["Date,Close"] + [f"{day:%Y-%m-%d},{price!r}" for day, price in zip(starts, prices.tolist())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{path}: {args.count} giorni, phi {args.phi}, R² teorico {args.phi ** 2:.3f}")


if __name__ == "__main__":
    main()
