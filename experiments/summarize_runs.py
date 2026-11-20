"""Riassume più training (es. semi diversi): LTC contro un riferimento, per parte.

    python -m experiments.summarize_runs outputs/timeseries_2026112*        # tutti i run indicati
    python -m experiments.summarize_runs outputs/timeseries_* --baseline linear_ar1
"""

import argparse
import json
from pathlib import Path


def summarize(directory: Path, baseline: str, part: str) -> dict | None:
    """Riga di riepilogo di un run: vantaggio relativo della LTC e p-value di Diebold–Mariano.

    None se il report non contiene il riferimento richiesto (formato precedente);
    p_value None se il test manca nel report o non era stimabile.
    """
    report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    results = report["results"][part]
    if baseline not in results:
        return None
    ltc, reference = results["ltc"]["mse"], results[baseline]["mse"]
    test = results.get("ltc_significance", {}).get(baseline, {})
    return {
        "run": directory.name,
        "seed": report["protocol"]["arguments"]["seed"],
        "epoch": report["protocol"]["best_epoch"],
        # Con la scelta dell'epoca sulla validazione, i suoi p-value non sono una verifica.
        "selection": "finale" if report["protocol"]["arguments"].get("final_epoch") else "validazione",
        "ltc_mse": ltc,
        "baseline_mse": reference,
        "relative": ltc / reference - 1,
        "p_value": test.get("p_value"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Cartelle outputs/timeseries_*.")
    parser.add_argument("--baseline", default="training_mean",
                        choices=("training_mean", "zero", "linear_ar1"))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    runs = sorted(path for path in args.runs if (path / "results.json").is_file())
    if not runs:
        parser.error("Nessuna cartella con results.json fra quelle indicate.")

    summary = {}
    for part in ("validation", "test"):
        rows = [row for path in runs if (row := summarize(path, args.baseline, part))]
        skipped = len(runs) - len(rows)
        tested = [row for row in rows if row["p_value"] is not None]
        better = sum(row["relative"] < 0 for row in rows)
        significant = sum(row["relative"] < 0 and row["p_value"] < args.alpha for row in tested)
        summary[part] = {"rows": rows, "better": better, "significant_better": significant,
                         "with_test": len(tested), "skipped_runs": skipped}

        print(f"\n{part}: LTC contro {args.baseline} ({len(rows)} run"
              + (f", {skipped} esclusi: riferimento assente)" if skipped else ")"))
        for row in rows:
            p_text = "n/d" if row["p_value"] is None else f"{row['p_value']:.3f}"
            print(f"  seme {row['seed']:>3} | epoca {row['epoch']:>3} ({row['selection']}) | "
                  f"MSE {row['relative']:+.3%} | p {p_text} | {row['run']}")
        print(f"  migliore in {better}/{len(rows)}; migliore e significativo (p<{args.alpha}) "
              f"in {significant}/{len(tested)} con test disponibile. I p-value non sono una "
              "verifica indipendente se la validazione è servita a scegliere.")

    print(json.dumps({part: {k: v for k, v in data.items() if k != "rows"}
                      for part, data in summary.items()}))


if __name__ == "__main__":
    main()
