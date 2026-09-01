from pathlib import Path

import pandas as pd
import pytest

from data_engine.cleaner import MarketDataCleaner
from data_engine.loader import DataLoadError


def create_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                ],
                utc=True,
            ),
            "open": [
                110.0,
                100.0,
                105.0,
            ],
            "high": [
                120.0,
                110.0,
                115.0,
            ],
            "low": [
                105.0,
                95.0,
                100.0,
            ],
            "close": [
                115.0,
                105.0,
                110.0,
            ],
            "volume": [
                1_200_000.0,
                1_000_000.0,
                1_100_000.0,
            ],
        }
    )


def test_clean_sorts_timestamps() -> None:
    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        create_dataframe(),
        asset="TEST",
    )

    assert cleaned["timestamp"].is_monotonic_increasing
    assert report.input_rows == 3
    assert report.output_rows == 3
    assert report.rows_removed == 0


def test_clean_removes_duplicate_timestamps() -> None:
    dataframe = create_dataframe()

    duplicate = dataframe.iloc[[1]].copy()

    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 3
    assert report.duplicates_removed == 1


def test_clean_removes_missing_rows() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = None

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 2
    assert report.missing_rows_removed == 1
    assert cleaned["close"].isna().sum() == 0


def test_clean_removes_invalid_ohlc_rows() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "high"] = 80.0

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 2
    assert report.invalid_rows_removed == 1


def test_clean_removes_negative_prices() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = -10.0

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 2
    assert report.invalid_rows_removed == 1


def test_clean_removes_zero_prices() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "close"] = 0.0

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 2
    assert report.invalid_rows_removed == 1


def test_clean_removes_negative_volume() -> None:
    dataframe = create_dataframe()

    dataframe.loc[1, "volume"] = -100.0

    cleaner = MarketDataCleaner()

    cleaned, report = cleaner.clean(
        dataframe,
        asset="TEST",
    )

    assert len(cleaned) == 2
    assert report.invalid_rows_removed == 1


def test_cleaned_data_is_valid() -> None:
    cleaner = MarketDataCleaner()

    cleaned, _ = cleaner.clean(
        create_dataframe(),
        asset="TEST",
    )

    report = cleaner.validator.validate(
        cleaned
    )

    assert report.valid is True


def test_clean_directory(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "raw"
    output_directory = tmp_path / "processed"

    input_directory.mkdir()

    dataframe = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01",
                "2025-01-02",
            ],
            "open": [100.0, 105.0],
            "high": [110.0, 115.0],
            "low": [95.0, 100.0],
            "close": [105.0, 110.0],
            "volume": [
                1_000_000.0,
                1_100_000.0,
            ],
        }
    )

    dataframe.to_csv(
        input_directory / "TEST.csv",
        index=False,
    )

    cleaner = MarketDataCleaner()

    reports = cleaner.clean_directory(
        input_directory,
        output_directory,
    )

    assert len(reports) == 1
    assert reports[0].asset == "TEST"
    assert reports[0].rows_removed == 0

    output_file = (
        output_directory / "TEST.csv"
    )

    assert output_file.exists()

    processed = pd.read_csv(
        output_file
    )

    assert len(processed) == 2
    assert list(processed.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_clean_rejects_empty_dataframe() -> None:
    cleaner = MarketDataCleaner()

    with pytest.raises(
        DataLoadError,
        match="empty dataset",
    ):
        cleaner.clean(
            pd.DataFrame(),
            asset="TEST",
        )