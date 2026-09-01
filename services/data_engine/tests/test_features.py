from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engine.features import (
    FeatureConfig,
    MarketFeatureEngineer,
)


def make_market_dataframe(rows: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    close = np.arange(100.0, 100.0 + rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.arange(
                1_000_000,
                1_000_000 + rows,
            ),
        }
    )


def test_feature_engineering_creates_expected_columns() -> None:
    dataframe = make_market_dataframe()

    engineer = MarketFeatureEngineer()

    result, report = engineer.transform(dataframe)

    expected_features = {
        "return_1d",
        "log_return_1d",
        "sma_5",
        "sma_20",
        "ema_5",
        "ema_20",
        "volatility_20",
        "volume_sma_20",
        "volume_ratio",
        "high_low_spread",
        "open_close_spread",
    }

    expected_targets = {
        "target_return_1d",
        "target_direction_1d",
    }

    assert expected_features.issubset(result.columns)
    assert expected_targets.issubset(result.columns)

    assert set(report.feature_columns) == expected_features


def test_first_complete_row_has_20_day_features() -> None:
    dataframe = make_market_dataframe(30)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    assert result["sma_20"].notna().all()
    assert result["ema_20"].notna().all()
    assert result["volatility_20"].notna().all()
    assert result["volume_sma_20"].notna().all()


def test_daily_return_is_correct() -> None:
    dataframe = make_market_dataframe(30)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    first_return = result.iloc[0]["return_1d"]

    expected = (
        dataframe.iloc[20]["close"]
        / dataframe.iloc[19]["close"]
    ) - 1.0

    assert first_return == pytest.approx(expected)


def test_log_return_is_correct() -> None:
    dataframe = make_market_dataframe(30)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    expected = np.log(
        dataframe.iloc[20]["close"]
    ) - np.log(
        dataframe.iloc[19]["close"]
    )

    assert result.iloc[0]["log_return_1d"] == pytest.approx(
        expected
    )


def test_sma_5_is_correct() -> None:
    dataframe = make_market_dataframe(30)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    expected = dataframe["close"].iloc[16:21].mean()

    assert result.iloc[0]["sma_5"] == pytest.approx(
        expected
    )


def test_target_uses_next_day_close() -> None:
    dataframe = make_market_dataframe(30)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    source_position = 20

    expected = (
        dataframe.iloc[source_position + 1]["close"]
        / dataframe.iloc[source_position]["close"]
    ) - 1.0

    assert result.iloc[0]["target_return_1d"] == pytest.approx(
        expected
    )


def test_target_is_not_a_feature() -> None:
    engineer = MarketFeatureEngineer()

    features = engineer.feature_columns()

    assert "target_return_1d" not in features
    assert "target_direction_1d" not in features


def test_no_missing_values_remain() -> None:
    dataframe = make_market_dataframe(50)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    assert not result.isna().any().any()


def test_output_remains_chronological() -> None:
    dataframe = make_market_dataframe(50)

    shuffled = dataframe.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(shuffled)

    timestamps = result["timestamp"]

    assert timestamps.is_monotonic_increasing


def test_source_dataframe_is_not_modified() -> None:
    dataframe = make_market_dataframe(50)

    original = dataframe.copy(deep=True)

    engineer = MarketFeatureEngineer()

    engineer.transform(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_transform_file_writes_features(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "features" / "output.csv"

    dataframe = make_market_dataframe(50)
    dataframe.to_csv(source, index=False)

    engineer = MarketFeatureEngineer()

    report = engineer.transform_file(
        source,
        destination,
    )

    assert destination.exists()
    assert report.input_rows == 50
    assert report.output_rows > 0

    output = pd.read_csv(destination)

    assert "return_1d" in output.columns
    assert "target_return_1d" in output.columns


def test_transform_directory_processes_all_csvs(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "processed"
    output_directory = tmp_path / "features"

    input_directory.mkdir()

    for ticker in ("AAPL", "MSFT", "NVDA"):
        dataframe = make_market_dataframe(50)
        dataframe.to_csv(
            input_directory / f"{ticker}.csv",
            index=False,
        )

    engineer = MarketFeatureEngineer()

    reports = engineer.transform_directory(
        input_directory,
        output_directory,
    )

    assert set(reports) == {
        "AAPL.csv",
        "MSFT.csv",
        "NVDA.csv",
    }

    assert (output_directory / "AAPL.csv").exists()
    assert (output_directory / "MSFT.csv").exists()
    assert (output_directory / "NVDA.csv").exists()


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureConfig(
            short_window=20,
            medium_window=5,
        )


def test_empty_dataframe_is_rejected() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    engineer = MarketFeatureEngineer()

    with pytest.raises(ValueError):
        engineer.transform(dataframe)