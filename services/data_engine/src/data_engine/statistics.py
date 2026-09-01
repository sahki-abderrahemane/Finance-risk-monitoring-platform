from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

FEATURE_COLUMNS = (
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
)

TARGET_COLUMNS = (
    "target_return_1d",
    "target_direction_1d",
)


@dataclass(frozen=True)
class StatisticalAnalysisReport:
    """Summary of statistical analysis for one dataset."""

    dataset_name: str
    rows: int
    columns: int
    feature_count: int
    target_count: int
    mean_return: float
    return_std: float
    mean_volatility: float
    positive_target_ratio: float


class MarketStatisticalAnalyzer:
    """
    Perform descriptive and relationship analysis on engineered
    market datasets.

    This module is analytical only. It does not train models and does
    not generate financial recommendations.
    """

    def analyze(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> dict[str, pd.DataFrame | StatisticalAnalysisReport]:
        """
        Analyze one engineered market dataframe.

        Returns a dictionary containing:
        - summary statistics
        - feature correlations
        - target correlations
        - report
        """
        self._validate_input(dataframe)

        df = dataframe.copy()

        numeric_columns = [
            column
            for column in (
                *REQUIRED_COLUMNS[1:],
                *FEATURE_COLUMNS,
                *TARGET_COLUMNS,
            )
            if column in df.columns
        ]

        summary = (
            df[numeric_columns]
            .describe()
            .transpose()
        )

        correlation_columns = [
            *FEATURE_COLUMNS,
            *TARGET_COLUMNS,
        ]

        correlation_matrix = (
            df[correlation_columns]
            .corr()
        )

        for col in correlation_matrix.columns:
            correlation_matrix.loc[col, col] = 1.0

        target_correlation = (
            df[
                [
                    *FEATURE_COLUMNS,
                    "target_return_1d",
                ]
            ]
            .corr()["target_return_1d"]
            .drop("target_return_1d")
            .sort_values(
                key=lambda values: values.abs(),
                ascending=False,
            )
            .to_frame("correlation_with_target")
        )

        report = StatisticalAnalysisReport(
            dataset_name=dataset_name,
            rows=len(df),
            columns=len(df.columns),
            feature_count=len(FEATURE_COLUMNS),
            target_count=len(TARGET_COLUMNS),
            mean_return=float(
                df["return_1d"].mean()
            ),
            return_std=float(
                df["return_1d"].std()
            ),
            mean_volatility=float(
                df["volatility_20"].mean()
            ),
            positive_target_ratio=float(
                df["target_direction_1d"].mean()
            ),
        )

        return {
            "summary": summary,
            "correlation_matrix": correlation_matrix,
            "target_correlation": target_correlation,
            "report": report,
        }

    def analyze_file(
        self,
        input_path: str | Path,
        output_directory: str | Path,
    ) -> StatisticalAnalysisReport:
        """
        Analyze one engineered CSV and write analysis artifacts.
        """
        source = Path(input_path)
        destination = Path(output_directory)

        if not source.exists():
            raise FileNotFoundError(
                f"Input dataset does not exist: {source}"
            )

        if source.suffix.lower() != ".csv":
            raise ValueError(
                f"Input dataset must be a CSV file: {source}"
            )

        dataframe = pd.read_csv(source)

        results = self.analyze(
            dataframe,
            dataset_name=source.stem,
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        results["summary"].to_csv(
            destination / f"{source.stem}_summary.csv"
        )

        results["correlation_matrix"].to_csv(
            destination / f"{source.stem}_correlation.csv"
        )

        results["target_correlation"].to_csv(
            destination / f"{source.stem}_target_correlation.csv"
        )

        report = results["report"]

        report_dataframe = pd.DataFrame(
            [
                {
                    "dataset_name": report.dataset_name,
                    "rows": report.rows,
                    "columns": report.columns,
                    "feature_count": report.feature_count,
                    "target_count": report.target_count,
                    "mean_return": report.mean_return,
                    "return_std": report.return_std,
                    "mean_volatility": report.mean_volatility,
                    "positive_target_ratio": (
                        report.positive_target_ratio
                    ),
                }
            ]
        )

        report_dataframe.to_csv(
            destination / f"{source.stem}_report.csv",
            index=False,
        )

        return report

    def analyze_directory(
        self,
        input_directory: str | Path,
        output_directory: str | Path,
    ) -> dict[str, StatisticalAnalysisReport]:
        """
        Analyze every engineered CSV in a directory.
        """
        source_directory = Path(input_directory)
        destination_directory = Path(output_directory)

        if not source_directory.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{source_directory}"
            )

        if not source_directory.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: "
                f"{source_directory}"
            )

        csv_files = sorted(
            source_directory.glob("*.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV datasets found in: "
                f"{source_directory}"
            )

        reports: dict[str, StatisticalAnalysisReport] = {}

        for source_file in csv_files:
            reports[source_file.name] = self.analyze_file(
                source_file,
                destination_directory,
            )

        return reports

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame,
    ) -> None:
        if dataframe.empty:
            raise ValueError(
                "Cannot analyze an empty dataframe."
            )

        missing = [
            column
            for column in (
                *REQUIRED_COLUMNS,
                *FEATURE_COLUMNS,
                *TARGET_COLUMNS,
            )
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Missing required statistical columns: "
                + ", ".join(missing)
            )