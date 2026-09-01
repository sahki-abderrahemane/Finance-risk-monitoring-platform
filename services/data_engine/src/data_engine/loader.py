from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_engine.schemas import MarketDataPoint


REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

CANONICAL_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class DataLoadError(Exception):
    """Raised when raw market data cannot be loaded."""


class CSVMarketDataLoader:
    """Load raw market-data CSV files into Sentinel's canonical schema."""

    def load(self, path: str | Path) -> pd.DataFrame:
        file_path = Path(path)

        self._validate_file(file_path)

        dataframe = self._read_csv(file_path)
        dataframe = self._normalize_columns(dataframe)
        dataframe = self._validate_columns(dataframe)
        dataframe = self._convert_types(dataframe)

        self._validate_rows(dataframe)

        return dataframe

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        if not file_path.exists():
            raise DataLoadError(
                f"Dataset not found: {file_path}"
            )

        if not file_path.is_file():
            raise DataLoadError(
                f"Dataset path is not a file: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise DataLoadError(
                f"Dataset is empty: {file_path}"
            )

    @staticmethod
    def _read_csv(file_path: Path) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv(
                file_path,
                header=None,
            )
        except (
            OSError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            raise DataLoadError(
                f"Failed to read dataset: {file_path}"
            ) from exc

        if dataframe.empty:
            raise DataLoadError(
                f"Dataset contains no rows: {file_path}"
            )

        return dataframe

    @staticmethod
    def _normalize_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert supported source formats into Sentinel's
        canonical column names.

        Supported formats:

        1. yfinance multi-level CSV:

           Price,Close,High,Low,Open,Volume
           Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
           Date,,,,,

        2. Flat CSV:

           date,open,high,low,close,volume

        3. Canonical CSV:

           timestamp,open,high,low,close,volume
        """

        dataframe = dataframe.copy()

        if dataframe.empty:
            raise DataLoadError(
                "Dataset contains no rows"
            )

        # ---------------------------------------------------------
        # yfinance multi-level format
        # ---------------------------------------------------------

        if CSVMarketDataLoader._is_yfinance_format(dataframe):
            normalized = dataframe.iloc[3:, :6].copy()

            normalized.columns = [
                "timestamp",
                "close",
                "high",
                "low",
                "open",
                "volume",
            ]

            return normalized.reset_index(drop=True)

        # ---------------------------------------------------------
        # Flat / canonical format
        # ---------------------------------------------------------

        header = [
            str(value).strip().lower()
            for value in dataframe.iloc[0].tolist()
        ]

        if "date" in header or "timestamp" in header:
            normalized = dataframe.iloc[1:].copy()
            normalized.columns = header

            if "date" in normalized.columns:
                normalized = normalized.rename(
                    columns={"date": "timestamp"}
                )

            normalized = normalized[
                [c for c in CANONICAL_COLUMNS if c in normalized.columns]
            ]
            return normalized.reset_index(drop=True)

        raise DataLoadError(
            "Unsupported CSV format. "
            "Expected yfinance or flat OHLCV format."
        )

    @staticmethod
    def _is_yfinance_format(
        dataframe: pd.DataFrame,
    ) -> bool:
        if len(dataframe) < 3:
            return False

        return (
            str(dataframe.iloc[0, 0]).strip().lower()
            == "price"
            and
            str(dataframe.iloc[1, 0]).strip().lower()
            == "ticker"
            and
            str(dataframe.iloc[2, 0]).strip().lower()
            == "date"
        )

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise DataLoadError(
                f"Missing required columns: {sorted(missing)}"
            )

        return dataframe

    @staticmethod
    def _convert_types(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        try:
            dataframe["timestamp"] = pd.to_datetime(
                dataframe["timestamp"],
                errors="raise",
                utc=True,
            )

            for column in NUMERIC_COLUMNS:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="raise",
                )

        except (
            ValueError,
            TypeError,
            OverflowError,
        ) as exc:
            raise DataLoadError(
                "Failed to convert dataset values "
                "to expected types"
            ) from exc

        return dataframe

    @staticmethod
    def _validate_rows(
        dataframe: pd.DataFrame,
    ) -> None:
        for row_number, row in enumerate(
            dataframe.to_dict(orient="records"),
            start=1,
        ):
            try:
                MarketDataPoint(**row)

            except Exception as exc:
                raise DataLoadError(
                    f"Invalid market-data row "
                    f"{row_number}: {exc}"
                ) from exc