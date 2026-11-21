"""End-to-end runs of the time-series scripts on a small synthetic series.

They replace the manual smoke runs: each script is executed as from the command
line, in a temporary directory, and its output files are checked.
"""

import csv
import json
import sys

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from experiments import data_quality_report, make_synthetic_series  # noqa: E402
from experiments import plot_timeseries_split, train_timeseries  # noqa: E402


def run(monkeypatch, module, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    module.main()


def strict_json(path):
    def reject(token):
        raise ValueError(f"token JSON non standard: {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture
def series(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, make_synthetic_series, "--phi", "0.3", "--count", "400",
        "--output-dir", "series")
    return tmp_path / "series" / "ar1_phi0.3_seed0.csv"


def only(paths):
    paths = list(paths)
    assert len(paths) == 1, paths
    return paths[0]


def test_training_writes_metrics_checkpoint_and_strict_report(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "2", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1")
    directory = only(tmp_path.glob("outputs/timeseries_*"))

    with (directory / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    assert rows[0]["train_loss"] == "" and float(rows[2]["train_loss"]) > 0

    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert {"model_state_dict", "epoch", "validation_loss", "calendar", "max_skipped",
            "inputs", "standardizer", "lookback"} <= checkpoint.keys()
    assert checkpoint["validation_loss"] == min(float(row["validation_loss"]) for row in rows)

    report = strict_json(directory / "results.json")
    test = report["results"]["test"]
    assert set(test) == {"ltc", "zero", "training_mean", "linear_ar1", "ltc_significance"}
    assert set(test["ltc_significance"]) == {"zero", "training_mean", "linear_ar1"}
    assert all(0 <= value["p_value"] <= 1 for value in test["ltc_significance"].values())

    predictions = torch.load(directory / "predictions.pt", weights_only=True)
    for part in ("validation", "test"):
        assert set(predictions[part]) == {"ltc", "zero", "training_mean", "linear_ar1", "targets"}
        assert len({len(values) for values in predictions[part].values()}) == 1
    assert len(predictions["test"]["targets"]) == test["ltc"]["count"]
    assert report["results"]["test"]["zero"]["direction_accuracy"] is None
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("include_test", [False, True])
def test_quality_report_hides_the_test_by_default(series, tmp_path, monkeypatch, include_test):
    arguments = ["--csv", str(series), "--by-decade"] + (["--include-test"] if include_test else [])
    run(monkeypatch, data_quality_report, *arguments)

    report = strict_json(only(tmp_path.glob("outputs/data_quality_*/report.json")))
    expected = {"training", "validation"} | ({"test"} if include_test else set())
    assert set(report["parts"]) == expected
    assert report["decades"]


def test_split_plot_is_saved(series, tmp_path, monkeypatch):
    run(monkeypatch, plot_timeseries_split, "--csv", str(series), "--output", "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_decade_runs_never_cross_a_gap_or_a_decade():
    from datetime import datetime, timedelta

    from timeseries.dataset import Piece

    day = timedelta(days=1)
    first = Piece(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3),
                  [datetime(1959, 12, 30) + k * day for k in range(3)])   # 30/12/1959 → 01/01/1960
    second = Piece(torch.tensor([4.0, 5.0]), torch.ones(2),
                   [datetime(1960, 1, 10) + k * day for k in range(2)])   # after a gap

    runs = data_quality_report.decade_runs([first, second])
    assert [run.tolist() for run in runs[1950]] == [[1.0, 2.0]]
    assert [run.tolist() for run in runs[1960]] == [[3.0], [4.0, 5.0]]


def test_partial_or_constant_groups_do_not_stop_the_report(tmp_path, monkeypatch):
    from datetime import date, timedelta

    monkeypatch.chdir(tmp_path)
    # Starts on 2009-12-30: the 2000s get a single return, the 2010s the rest.
    rows = ["Date,Close"] + [f"{date(2009, 12, 30) + timedelta(days=k)},{100 + (k % 7)}"
                             for k in range(300)]
    (tmp_path / "prices.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    run(monkeypatch, data_quality_report, "--csv", "prices.csv", "--by-decade", "--include-test")

    report = strict_json(only(tmp_path.glob("outputs/data_quality_*/report.json")))
    assert report["decades"]["2000"]["count"] == 1
    assert "unavailable" in report["decades"]["2000"]
    assert "lag1_autocorrelation" in report["decades"]["2010"]
    assert {"training", "validation", "test"} == set(report["parts"])

    constant = data_quality_report.safe_stats([torch.ones(10)])
    assert constant["count"] == 10 and "costanti" in constant["unavailable"]


def test_training_with_date_split_records_the_dates(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--train-start", "2000-03-01",
        "--validation-start", "2000-09-01", "--test-start", "2000-11-01")

    report = strict_json(only(tmp_path.glob("outputs/timeseries_*/results.json")))
    arguments = report["protocol"]["arguments"]
    assert arguments["train_start"].startswith("2000-03-01")
    assert arguments["test_start"].startswith("2000-11-01")
    assert report["results"]["test"]["ltc"]["count"] > 0


def test_quality_report_uses_the_same_dates_as_training(series, tmp_path, monkeypatch):
    run(monkeypatch, data_quality_report, "--csv", str(series), "--train-start", "2000-03-01",
        "--validation-start", "2000-09-01", "--test-start", "2000-11-01")

    parts = strict_json(only(tmp_path.glob("outputs/data_quality_*/report.json")))["parts"]
    assert parts["training"]["first"] == "2000-03-01"
    assert parts["validation"]["first"] == "2000-09-01"
    assert set(parts) == {"training", "validation"}


@pytest.mark.parametrize("dates,message", [
    (["--train-start", "2000-03-01"], "richiede"),
    (["--validation-start", "2000-09-01"], "insieme"),
])
def test_quality_report_rejects_incomplete_dates_before_reading(tmp_path, monkeypatch, capsys,
                                                               dates, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):   # the CSV does not even exist: rejected before reading
        run(monkeypatch, data_quality_report, "--csv", "missing.csv", *dates)
    assert message in capsys.readouterr().err
    assert not (tmp_path / "outputs").exists()


def test_final_epoch_uses_the_last_model_without_validation_choice(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "3", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--final-epoch")
    directory = only(tmp_path.glob("outputs/timeseries_*"))

    assert torch.load(directory / "best.pt", weights_only=True)["epoch"] == 3
    protocol = strict_json(directory / "results.json")["protocol"]
    assert protocol["best_epoch"] == 3 and protocol["arguments"]["final_epoch"] is True


def test_summarize_runs_counts_seeds_better_than_the_baseline(series, tmp_path, monkeypatch, capsys):
    from experiments import summarize_runs

    for seed in ("1", "2"):
        run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
            "--hidden-size", "2", "--substeps", "1", "--final-epoch", "--seed", seed)
    runs = sorted(tmp_path.glob("outputs/timeseries_*"))
    assert len(runs) == 2
    capsys.readouterr()

    run(monkeypatch, summarize_runs, *map(str, runs))
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    rows = [summarize_runs.summarize(path, "training_mean", "validation") for path in runs]
    assert sorted(row["seed"] for row in rows) == [1, 2]
    assert summary["validation"]["better"] == sum(row["relative"] < 0 for row in rows)
    assert 0 <= summary["validation"]["significant_better"] <= summary["validation"]["better"]


def test_run_is_saved_when_significance_cannot_be_estimated(series, tmp_path, monkeypatch):
    # 400 days from 2000-01-01: a test starting on 2001-01-27 leaves only a few windows.
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--validation-start", "2000-09-01",
        "--test-start", "2001-01-27")
    directory = only(tmp_path.glob("outputs/timeseries_*"))

    test = strict_json(directory / "results.json")["results"]["test"]
    assert test["ltc"]["count"] <= 6
    assert all(value["p_value"] is None for value in test["ltc_significance"].values())
    assert (directory / "predictions.pt").is_file()


def test_summarize_runs_handles_reports_in_the_old_format(series, tmp_path, monkeypatch, capsys):
    from experiments import summarize_runs

    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--final-epoch")
    new = only(tmp_path.glob("outputs/timeseries_*"))

    # An older report: no significance and no linear baseline.
    old = tmp_path / "outputs" / "timeseries_old"
    old.mkdir()
    report = strict_json(new / "results.json")
    for part in report["results"].values():
        part.pop("ltc_significance")
        part.pop("linear_ar1")
    report["protocol"]["arguments"].pop("final_epoch")
    (old / "results.json").write_text(json.dumps(report), encoding="utf-8")
    capsys.readouterr()

    run(monkeypatch, summarize_runs, str(new), str(old))
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["validation"]["with_test"] == 1          # only the new run has a p-value
    assert summary["validation"]["skipped_runs"] == 0

    run(monkeypatch, summarize_runs, str(new), str(old), "--baseline", "linear_ar1")
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["validation"]["skipped_runs"] == 1        # old run lacks linear_ar1
    assert summarize_runs.summarize(old, "training_mean", "test")["selection"] == "validazione"


def test_volatility_training_end_to_end(series, tmp_path, monkeypatch):
    from experiments import train_volatility

    # Calendar options passed explicitly, to check that the checkpoint records them.
    run(monkeypatch, train_volatility, "--csv", str(series), "--epochs", "2", "--lookback", "6",
        "--hidden-size", "2", "--substeps", "1", "--horizon", "3",
        "--calendar", "continuous", "--max-skipped", "1", "--period-hours", "24")
    directory = only(tmp_path.glob("outputs/volatility_*"))

    report = strict_json(directory / "results.json")
    test = report["results"]["test"]
    assert set(test) == {"ltc", "persistence", "training_mean", "ltc_significance"}
    assert test["ltc"]["tail"]["tail_count"] >= 1 and test["persistence"]["tail"]["rest_mse"] >= 0
    assert report["protocol"]["arguments"]["horizon"] == 3
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert checkpoint["horizon"] == 3 and "target_standardizer" in checkpoint
    assert checkpoint["mode"] == "level" and "standardizer" in checkpoint
    assert (checkpoint["period_hours"], checkpoint["calendar"], checkpoint["max_skipped"]) == (
        24.0, "continuous", 1)
    assert checkpoint["inputs"] == ["standardized_return", "elapsed_periods"]

    predictions = torch.load(directory / "predictions.pt", weights_only=True)
    assert (predictions["test"]["targets"] >= 0).all()
    assert len(predictions["test"]["ltc"]) == test["ltc"]["count"]


def test_volatility_persistence_ratio_mode(series, tmp_path, monkeypatch):
    from experiments import train_volatility

    run(monkeypatch, train_volatility, "--csv", str(series), "--epochs", "1", "--lookback", "6",
        "--hidden-size", "2", "--substeps", "1", "--horizon", "3", "--mode", "persistence_ratio")
    directory = only(tmp_path.glob("outputs/volatility_*"))

    assert torch.load(directory / "best.pt", weights_only=True)["mode"] == "persistence_ratio"
    predictions = torch.load(directory / "predictions.pt", weights_only=True)["test"]
    # Corrections multiply persistence: forecasts stay positive and finite.
    assert (predictions["ltc"] > 0).all() and torch.isfinite(predictions["ltc"]).all()
    assert strict_json(directory / "results.json")["protocol"]["arguments"]["mode"] == "persistence_ratio"


@pytest.mark.parametrize("mode", ["level", "persistence_ratio"])
def test_volatility_run_is_saved_with_a_single_test_window(series, tmp_path, monkeypatch, mode):
    from experiments import train_volatility

    # 400 days from 2000-01-01; a test from 2001-01-26 has 9 returns = lookback 6 + horizon 3.
    run(monkeypatch, train_volatility, "--csv", str(series), "--epochs", "1", "--lookback", "6",
        "--hidden-size", "2", "--substeps", "1", "--horizon", "3", "--mode", mode,
        "--validation-start", "2000-09-01", "--test-start", "2001-01-26")
    directory = only(tmp_path.glob("outputs/volatility_*"))

    test = strict_json(directory / "results.json")["results"]["test"]
    assert test["ltc"]["count"] == 1
    assert test["ltc"]["tail"]["tail_count"] == 1 and test["ltc"]["tail"]["rest_mse"] is None
    assert all(value["p_value"] is None for value in test["ltc_significance"].values())
    assert (directory / "predictions.pt").is_file()


def test_cell_benchmark_reports_both_cells(monkeypatch, capsys):
    from experiments import benchmark_cells

    run(monkeypatch, benchmark_cells, "--steps", "2", "--batch", "4", "--repeats", "2")
    results = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    names = {key.split(" | ")[0] for key in results}
    assert names == {"ltc (6 sottopassi)", "cfc"}
    assert all(value["single_step_ms"] > 0 and value["sequence_ms"] > 0 for value in results.values())


@pytest.mark.parametrize("script", ["train_timeseries", "train_volatility"])
def test_training_scripts_run_with_the_cfc_cell(series, tmp_path, monkeypatch, script):
    import importlib

    module = importlib.import_module(f"experiments.{script}")
    extra = ["--horizon", "3"] if script == "train_volatility" else []
    run(monkeypatch, module, "--csv", str(series), "--epochs", "1", "--lookback", "6",
        "--hidden-size", "3", "--cell", "cfc", "--backbone-units", "4", *extra)
    directory = only([*tmp_path.glob("outputs/timeseries_*"), *tmp_path.glob("outputs/volatility_*")])

    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert checkpoint["cell"] == "cfc" and checkpoint["config"]["backbone_units"] == 4
    assert any(name.startswith("cell.backbone") for name in checkpoint["model_state_dict"])
    assert strict_json(directory / "results.json")["results"]["test"]["ltc"]["count"] > 0


def test_training_runs_with_the_lstm_baseline(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "6",
        "--hidden-size", "3", "--cell", "lstm")
    directory = only(tmp_path.glob("outputs/timeseries_*"))
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert checkpoint["cell"] == "lstm" and checkpoint["config"] == {"input_size": 2, "hidden_size": 3}
    assert any(name.startswith("lstm.") for name in checkpoint["model_state_dict"])
    assert strict_json(directory / "results.json")["results"]["test"]["ltc"]["count"] > 0


def test_summarize_runs_never_mixes_ltc_and_cfc(series, tmp_path, monkeypatch, capsys):
    from experiments import summarize_runs

    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--cell", "cfc", "--backbone-units", "4")
    cfc_run = only(tmp_path.glob("outputs/timeseries_*"))
    old_ltc = tmp_path / "outputs" / "timeseries_old"          # report written before --cell
    old_ltc.mkdir()
    report = strict_json(cfc_run / "results.json")
    report["protocol"]["arguments"].pop("cell")
    (old_ltc / "results.json").write_text(json.dumps(report), encoding="utf-8")
    assert (summarize_runs.run_cell(cfc_run), summarize_runs.run_cell(old_ltc)) == ("cfc", "ltc")

    with pytest.raises(SystemExit):                             # mixed group refused
        run(monkeypatch, summarize_runs, str(cfc_run), str(old_ltc))
    assert "--cell" in capsys.readouterr().err

    for cell, expected in (("cfc", cfc_run.name), ("ltc", old_ltc.name)):
        run(monkeypatch, summarize_runs, str(cfc_run), str(old_ltc), "--cell", cell)
        output = capsys.readouterr().out
        assert f"rete {cell}" in output and expected in output
        assert json.loads(output.strip().splitlines()[-1])["validation"]["better"] <= 1


def test_event_model_check_end_to_end(tmp_path, monkeypatch):
    from experiments import check_event_model

    monkeypatch.chdir(tmp_path)
    run(monkeypatch, check_event_model, "--events", "600", "--lookback", "10", "--epochs", "1",
        "--hidden-size", "3", "--backbone-units", "4", "--device", "cpu")
    report = strict_json(only(tmp_path.glob("outputs/event_model_*/results.json")))
    assert report["device"] == "cpu" and report["training_seconds"] > 0
    assert set(report["results"]) == {"drift", "volatility"}
    assert all(value["cfc_mse"] >= 0 and value["naive_mse"] >= 0
               for value in report["results"].values())
    for value in report["results"].values():
        assert value["oracle_constant_mse"] <= value["constant_mse"] + 1e-12


def test_event_model_check_on_the_book_data(tmp_path, monkeypatch):
    from experiments import check_event_model

    monkeypatch.chdir(tmp_path)
    run(monkeypatch, check_event_model, "--data", "book", "--events", "600", "--lookback", "10",
        "--epochs", "1", "--hidden-size", "3", "--backbone-units", "4", "--device", "cpu")
    report = strict_json(only(tmp_path.glob("outputs/event_model_*/results.json")))
    assert list(report["results"]) == ["drift", "volatility", "toxicity", "k"]
    assert report["arguments"]["data"] == "book"


def test_event_model_check_with_the_agreement_input(tmp_path, monkeypatch):
    from experiments import check_event_model

    monkeypatch.chdir(tmp_path)
    run(monkeypatch, check_event_model, "--data", "book_agreement", "--events", "600",
        "--lookback", "10", "--epochs", "1", "--hidden-size", "3", "--backbone-units", "4",
        "--device", "cpu")
    report = strict_json(only(tmp_path.glob("outputs/event_model_*/results.json")))
    assert report["arguments"]["data"] == "book_agreement" and "toxicity" in report["results"]


def test_constant_baseline_uses_the_training_mean_not_the_test_mean():
    from experiments.check_event_model import constant_baselines

    truth = torch.tensor([1.0, 3.0])                 # test mean 2, variance 1
    baselines = constant_baselines(truth, training_mean=0.5)
    assert baselines["oracle_constant_mse"] == pytest.approx(1.0)
    # Honest constant: variance + (test mean − training mean)² = 1 + 1.5².
    assert baselines["constant_mse"] == pytest.approx(1.0 + 1.5 ** 2)


def test_summarize_event_runs_reports_ratios_and_refuses_mixed_devices(tmp_path, monkeypatch, capsys):
    from experiments import summarize_event_runs

    def fake_run(name, seed, cfc, naive, device=None, epochs=15):
        directory = tmp_path / name
        directory.mkdir()
        report = {"arguments": {"seed": seed, "events": 600, "lookback": 10, "epochs": epochs,
                                "hidden_size": 3, "backbone_units": 4}, "results": {
            "volatility": {"cfc_mse": cfc, "naive_mse": naive},
            "drift": {"cfc_mse": 1.0, "naive_mse": 1.0}}}
        if device:
            report["device"] = device
        (directory / "results.json").write_text(json.dumps(report), encoding="utf-8")
        return directory

    runs = [fake_run("a", 0, 0.8, 1.0), fake_run("b", 1, 1.2, 1.0, "cpu"), fake_run("c", 2, 0.5, 1.0, "cpu")]
    run(monkeypatch, summarize_event_runs, *map(str, runs))
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"better": 2, "count": 3, "min": 0.5, "median": 0.8, "max": 1.2}

    with pytest.raises(SystemExit):   # fake reports only have volatility and drift
        run(monkeypatch, summarize_event_runs, *map(str, runs), "--output", "toxicity")
    assert "Uscita toxicity assente" in capsys.readouterr().err

    gpu = fake_run("d", 0, 0.9, 1.0, "cuda")
    with pytest.raises(SystemExit):
        run(monkeypatch, summarize_event_runs, str(runs[0]), str(gpu))
    assert "dispositivi" in capsys.readouterr().err

    # The same folder passed twice counts once.
    run(monkeypatch, summarize_event_runs, *map(str, runs), str(runs[0]))
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["count"] == 3

    for extra, message in ((fake_run("run_dup", 0, 0.7, 1.0, "cpu"), "Semi ripetuti"),
                           (fake_run("f", 5, 0.7, 1.0, "cpu", epochs=30), "epochs")):
        with pytest.raises(SystemExit):
            run(monkeypatch, summarize_event_runs, *map(str, runs), str(extra))
        error = capsys.readouterr().err
        assert message in error
        if message == "Semi ripetuti":
            assert "seme 0: a, run_dup" in error            # both folders are named
