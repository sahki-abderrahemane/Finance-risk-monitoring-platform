from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engine.features import MarketFeatureEngineer
from data_engine.statistics import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    MarketStatisticalAnalyzer,
)


def make_market_dataframe(rows: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    close = np.arange(100.0, 100.0 + rows)

    dataframe = pd.DataFrame(
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

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(dataframe)

    return result


def test_analysis_returns_expected_results() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(
        dataframe,
        dataset_name="TEST",
    )

    assert "summary" in results
    assert "correlation_matrix" in results
    assert "target_correlation" in results
    assert "report" in results


def test_report_contains_correct_dimensions() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(
        dataframe,
        dataset_name="TEST",
    )

    report = results["report"]

    assert report.dataset_name == "TEST"
    assert report.rows == len(dataframe)
    assert report.columns == len(dataframe.columns)
    assert report.feature_count == len(FEATURE_COLUMNS)
    assert report.target_count == len(TARGET_COLUMNS)


def test_summary_contains_descriptive_statistics() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    summary = results["summary"]

    assert "mean" in summary.columns
    assert "std" in summary.columns
    assert "min" in summary.columns
    assert "max" in summary.columns

    assert "close" in summary.index
    assert "return_1d" in summary.index
    assert "volatility_20" in summary.index


def test_return_statistics_are_correct() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    report = results["report"]

    assert report.mean_return == pytest.approx(
        dataframe["return_1d"].mean()
    )

    assert report.return_std == pytest.approx(
        dataframe["return_1d"].std()
    )


def test_volatility_statistics_are_correct() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    report = results["report"]

    assert report.mean_volatility == pytest.approx(
        dataframe["volatility_20"].mean()
    )


def test_target_ratio_is_between_zero_and_one() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    report = results["report"]

    assert 0.0 <= report.positive_target_ratio <= 1.0


def test_correlation_matrix_contains_features_and_targets() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    correlation = results["correlation_matrix"]

    expected_columns = [
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]

    assert list(correlation.columns) == expected_columns
    assert list(correlation.index) == expected_columns


def test_correlation_diagonal_is_one() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    correlation = results["correlation_matrix"]

    for column in correlation.columns:
        assert correlation.loc[column, column] == pytest.approx(
            1.0
        )


def test_target_correlation_contains_all_features() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    target_correlation = results["target_correlation"]

    assert set(target_correlation.index) == set(
        FEATURE_COLUMNS
    )

    assert (
        "correlation_with_target"
        in target_correlation.columns
    )


def test_target_correlations_are_sorted_by_absolute_value() -> None:
    dataframe = make_market_dataframe()

    analyzer = MarketStatisticalAnalyzer()

    results = analyzer.analyze(dataframe)

    target_correlation = results["target_correlation"]

    values = target_correlation[
        "correlation_with_target"
    ].abs().tolist()

    assert values == sorted(
        values,
        reverse=True,
    )


def test_analyze_does_not_modify_dataframe() -> None:
    dataframe = make_market_dataframe()

    original = dataframe.copy(deep=True)

    analyzer = MarketStatisticalAnalyzer()

    analyzer.analyze(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_analyze_file_writes_results(
    tmp_path: Path,
) -> None:
    dataframe = make_market_dataframe()

    source = tmp_path / "AAPL.csv"
    output_directory = tmp_path / "statistics"

    dataframe.to_csv(
        source,
        index=False,
    )

    analyzer = MarketStatisticalAnalyzer()

    report = analyzer.analyze_file(
        source,
        output_directory,
    )

    assert report.dataset_name == "AAPL"

    assert (
        output_directory / "AAPL_summary.csv"
    ).exists()

    assert (
        output_directory / "AAPL_correlation.csv"
    ).exists()

    assert (
        output_directory / "AAPL_target_correlation.csv"
    ).exists()

    assert (
        output_directory / "AAPL_report.csv"
    ).exists()


def test_analyze_directory_processes_all_datasets(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "features"
    output_directory = tmp_path / "statistics"

    input_directory.mkdir()

    for ticker in (
        "AAPL",
        "AMZN",
        "GOOGL",
    ):
        dataframe = make_market_dataframe()

        dataframe.to_csv(
            input_directory / f"{ticker}.csv",
            index=False,
        )

    analyzer = MarketStatisticalAnalyzer()

    reports = analyzer.analyze_directory(
        input_directory,
        output_directory,
    )

    assert set(reports) == {
        "AAPL.csv",
        "AMZN.csv",
        "GOOGL.csv",
    }

    assert (
        output_directory / "AAPL_summary.csv"
    ).exists()

    assert (
        output_directory / "AMZN_summary.csv"
    ).exists()

    assert (
        output_directory / "GOOGL_summary.csv"
    ).exists()


def test_empty_dataframe_is_rejected() -> None:
    analyzer = MarketStatisticalAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(
            pd.DataFrame()
        )


def test_missing_feature_is_rejected() -> None:
    dataframe = make_market_dataframe()

    dataframe = dataframe.drop(
        columns=["sma_20"]
    )

    analyzer = MarketStatisticalAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(dataframe)