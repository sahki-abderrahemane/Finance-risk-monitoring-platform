from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for market feature engineering."""

    short_window: int = 5
    medium_window: int = 20
    volatility_window: int = 20

    def __post_init__(self) -> None:
        if self.short_window <= 0:
            raise ValueError("short_window must be greater than zero.")

        if self.medium_window <= 0:
            raise ValueError("medium_window must be greater than zero.")

        if self.volatility_window <= 1:
            raise ValueError("volatility_window must be greater than one.")

        if self.short_window >= self.medium_window:
            raise ValueError(
                "short_window must be smaller than medium_window."
            )


@dataclass(frozen=True)
class FeatureEngineeringReport:
    """Summary of a feature engineering operation."""

    input_rows: int
    output_rows: int
    rows_removed: int
    feature_columns: tuple[str, ...]


class MarketFeatureEngineer:
    """
    Generate machine-learning features from cleaned OHLCV market data.

    The engineer never modifies the source dataframe in-place.

    All historical features are calculated using information available
    at the current timestamp. The target is deliberately shifted into
    the future so it cannot leak future information into the features.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
        """
        Transform a cleaned market dataframe into an ML-ready dataframe.

        Parameters
        ----------
        dataframe:
            Cleaned OHLCV market data.

        Returns
        -------
        tuple[pd.DataFrame, FeatureEngineeringReport]
            Feature dataframe and operation report.
        """
        self._validate_input_columns(dataframe)

        df = dataframe.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="raise",
            utc=True,
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        input_rows = len(df)

        self._validate_numeric_columns(df)

        # ------------------------------------------------------------------
        # Price returns
        # ------------------------------------------------------------------

        df["return_1d"] = df["close"].pct_change()

        df["log_return_1d"] = np.log(df["close"]).diff()

        # ------------------------------------------------------------------
        # Moving averages
        # ------------------------------------------------------------------

        df["sma_5"] = df["close"].rolling(
            window=self.config.short_window,
            min_periods=self.config.short_window,
        ).mean()

        df["sma_20"] = df["close"].rolling(
            window=self.config.medium_window,
            min_periods=self.config.medium_window,
        ).mean()

        df["ema_5"] = df["close"].ewm(
            span=self.config.short_window,
            adjust=False,
            min_periods=self.config.short_window,
        ).mean()

        df["ema_20"] = df["close"].ewm(
            span=self.config.medium_window,
            adjust=False,
            min_periods=self.config.medium_window,
        ).mean()

        # ------------------------------------------------------------------
        # Volatility
        # ------------------------------------------------------------------

        df["volatility_20"] = df["log_return_1d"].rolling(
            window=self.config.volatility_window,
            min_periods=self.config.volatility_window,
        ).std()

        # ------------------------------------------------------------------
        # Volume features
        # ------------------------------------------------------------------

        df["volume_sma_20"] = df["volume"].rolling(
            window=self.config.medium_window,
            min_periods=self.config.medium_window,
        ).mean()

        df["volume_ratio"] = (
            df["volume"] / df["volume_sma_20"]
        )

        # ------------------------------------------------------------------
        # Intraday price structure
        # ------------------------------------------------------------------

        df["high_low_spread"] = (
            (df["high"] - df["low"]) / df["close"]
        )

        df["open_close_spread"] = (
            (df["close"] - df["open"]) / df["open"]
        )

        # ------------------------------------------------------------------
        # Future target
        #
        # target_return_1d represents the return from t -> t+1.
        # It is NOT used as an input feature.
        # ------------------------------------------------------------------

        df["target_return_1d"] = (
            df["close"].shift(-1) / df["close"]
        ) - 1.0

        df["target_direction_1d"] = (
            df["target_return_1d"] > 0.0
        ).astype("Int64")

        feature_columns = self.feature_columns()

        target_columns = (
            "target_return_1d",
            "target_direction_1d",
        )

        required_ml_columns = (
            *feature_columns,
            *target_columns,
        )

        # Rows at the beginning cannot have complete rolling features.
        # The final row cannot have a future target.
        df = df.dropna(
            subset=required_ml_columns
        ).reset_index(drop=True)

        # Convert target direction back to a regular integer after
        # removing the rows containing the missing final target.
        df["target_direction_1d"] = (
            df["target_direction_1d"].astype(int)
        )

        output_rows = len(df)

        report = FeatureEngineeringReport(
            input_rows=input_rows,
            output_rows=output_rows,
            rows_removed=input_rows - output_rows,
            feature_columns=feature_columns,
        )

        return df, report

    def transform_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> FeatureEngineeringReport:
        """
        Read one cleaned CSV, generate features, and write the result.
        """
        source = Path(input_path)
        destination = Path(output_path)

        if not source.exists():
            raise FileNotFoundError(
                f"Input dataset does not exist: {source}"
            )

        if source.suffix.lower() != ".csv":
            raise ValueError(
                f"Input dataset must be a CSV file: {source}"
            )

        dataframe = pd.read_csv(source)

        transformed, report = self.transform(dataframe)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        transformed.to_csv(
            destination,
            index=False,
        )

        return report

    def transform_directory(
        self,
        input_directory: str | Path,
        output_directory: str | Path,
    ) -> dict[str, FeatureEngineeringReport]:
        """
        Generate features for every CSV in a directory.

        Source files are never modified.
        """
        source_directory = Path(input_directory)
        destination_directory = Path(output_directory)

        if not source_directory.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {source_directory}"
            )

        if not source_directory.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: {source_directory}"
            )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_files = sorted(
            source_directory.glob("*.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV datasets found in: {source_directory}"
            )

        reports: dict[str, FeatureEngineeringReport] = {}

        for source_file in csv_files:
            destination_file = (
                destination_directory / source_file.name
            )

            reports[source_file.name] = self.transform_file(
                source_file,
                destination_file,
            )

        return reports

    @staticmethod
    def feature_columns() -> tuple[str, ...]:
        """Return the columns that are valid model input features."""
        return (
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

    @staticmethod
    def _validate_input_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Missing required market columns: "
                + ", ".join(missing)
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot engineer features from an empty dataframe."
            )

    @staticmethod
    def _validate_numeric_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        numeric_columns = (
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        for column in numeric_columns:
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise TypeError(
                    f"Column '{column}' must be numeric."
                )