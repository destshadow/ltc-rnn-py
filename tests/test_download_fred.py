"""FRED downloader: format checks only, no network access."""

import pytest

from experiments.download_fred import check_fred_csv


def test_valid_series_counts_rows_and_values():
    content = b"observation_date,SP500\n2024-01-01,\n2024-01-02,4742.83\n2024-01-03,.\n2024-01-04,4688.68\n"
    assert check_fred_csv(content, "SP500") == {"rows": 4, "values": 2}


@pytest.mark.parametrize("content,match", [
    (b"<!DOCTYPE html><html>verify</html>", "intestazione"),  # anti-bot page
    (b"observation_date,DJIA\n2024-01-02,1\n", "intestazione"),  # other series
    (b"observation_date,SP500\n2024-01-02,1,2\n", "campi"),
    (b"observation_date,SP500\n2024-01-01,\n2024-01-02,.\n", "valori"),
])
def test_unexpected_responses_are_rejected(content, match):
    with pytest.raises(ValueError, match=match):
        check_fred_csv(content, "SP500")
