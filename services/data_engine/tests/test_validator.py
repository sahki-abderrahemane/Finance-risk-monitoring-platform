from datetime import datetime, timezone

import pandas as pd

from data_engine.validator import MarketDataValidator


def create_dataframe() -> pd.DataFrame:
    """Create a valid canonical market dataset."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ],
                utc=True,
            ),
            "open": [
                100.0,
                105.0,
                110.0,
            ],
            "high": [
                110.0,
                115.0,
                120.0,
            ],
            "low": [
                95.0,
                100.0,
                105.0,
            ],
            "close": [
                105.0,
                110.0,
                115.0,
            ],
            "volume": [
                1_000_000.0,
                1_100_000.0,
                1_200_000.0,
            ],
        }
    )


def test_valid_dataframe() -> None:
    validator = MarketDataValidator()

    report = validator.validate(
        create_dataframe()
    )

    assert report.valid is True
    assert report.rows == 3
    assert report.issue_count == 0


def test_detects_missing_values() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = None

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "missing_values" in rules


def test_detects_duplicate_timestamps() -> None:
    dataframe = create_dataframe()

    dataframe.loc[2, "timestamp"] = dataframe.loc[
        1,
        "timestamp",
    ]

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "duplicate_timestamps" in rules


def test_detects_unsorted_timestamps() -> None:
    dataframe = create_dataframe()

    dataframe = dataframe.iloc[
        [0, 2, 1]
    ].reset_index(drop=True)

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "chronological_order" in rules


def test_detects_invalid_ohlc() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "high"] = 90.0

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "ohlc_consistency" in rules


def test_detects_negative_prices() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = -10.0

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "negative_prices" in rules


def test_detects_zero_prices() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = 0.0

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "zero_prices" in rules


def test_detects_negative_volume() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "volume"] = -100.0

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "negative_volume" in rules


def test_detects_zero_volume() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "volume"] = 0.0

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False

    rules = {
        issue.rule
        for issue in report.issues
    }

    assert "zero_volume" in rules


def test_detects_missing_required_column() -> None:
    dataframe = create_dataframe()

    dataframe = dataframe.drop(
        columns=["volume"]
    )

    validator = MarketDataValidator()

    report = validator.validate(dataframe)

    assert report.valid is False
    assert report.issue_count == 1

    issue = report.issues[0]

    assert issue.rule == "required_columns"
    assert "volume" in issue.message