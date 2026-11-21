"""Riassume più run di experiments.check_event_model: rapporto CfC / stima a finestra.

    python -m experiments.summarize_event_runs outputs/event_model_*
"""

import argparse
import json
from pathlib import Path
from statistics import median


def summarize(runs: list[Path], output: str) -> dict:
    """Per l'uscita indicata: rapporto MSE CfC / finestra di ogni run e statistiche."""
    rows = []
    for path in runs:
        report = json.loads((path / "results.json").read_text(encoding="utf-8"))
        values = report["results"][output]
        rows.append({"run": path.name, "seed": report["arguments"]["seed"],
                     "device": report.get("device", "cpu"),   # i primi report erano tutti su CPU
                     "ratio": values["cfc_mse"] / values["naive_mse"]})
    ratios = [row["ratio"] for row in rows]
    return {"rows": rows, "better": sum(ratio < 1 for ratio in ratios), "count": len(rows),
            "min": min(ratios), "median": median(ratios), "max": max(ratios)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", choices=("volatility", "drift"), default="volatility")
    args = parser.parse_args()

    runs = sorted(path for path in args.runs if (path / "results.json").is_file())
    if not runs:
        parser.error("Nessuna cartella con results.json fra quelle indicate.")
    devices = {json.loads((path / "results.json").read_text(encoding="utf-8")).get("device", "cpu")
               for path in runs}
    if len(devices) > 1:
        parser.error("Run su dispositivi diversi: riassumerli separatamente.")

    summary = summarize(runs, args.output)
    print(f"{args.output}: rapporto MSE CfC / finestra ({summary['count']} run, {devices.pop()})")
    for row in sorted(summary["rows"], key=lambda row: row["seed"]):
        print(f"  seme {row['seed']:>3} | {row['ratio']:.3f} | {row['run']}")
    print(f"  CfC migliore in {summary['better']}/{summary['count']}; "
          f"min {summary['min']:.3f}, mediana {summary['median']:.3f}, max {summary['max']:.3f}")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}))


if __name__ == "__main__":
    main()
