from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_engine.loader import CSVMarketDataLoader, DataLoadError
from data_engine.validator import MarketDataValidator


@dataclass(frozen=True)
class CleaningReport:
    """Summary of changes made during cleaning."""

    asset: str
    input_rows: int
    output_rows: int
    duplicates_removed: int
    missing_rows_removed: int
    invalid_rows_removed: int

    @property
    def rows_removed(self) -> int:
        return self.input_rows - self.output_rows


class MarketDataCleaner:
    """Clean canonical market data without modifying raw files."""

    REQUIRED_COLUMNS = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    PRICE_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
    ]

    def __init__(
        self,
        loader: CSVMarketDataLoader | None = None,
        validator: MarketDataValidator | None = None,
    ) -> None:
        self.loader = loader or CSVMarketDataLoader()
        self.validator = validator or MarketDataValidator()

    def clean(
        self,
        dataframe: pd.DataFrame,
        asset: str = "UNKNOWN",
    ) -> tuple[pd.DataFrame, CleaningReport]:
        self._validate_input(dataframe)

        input_rows = len(dataframe)
        cleaned = dataframe.copy()

        cleaned = cleaned[
            self.REQUIRED_COLUMNS
        ].copy()

        cleaned = (
            cleaned
            .sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        before = len(cleaned)

        cleaned = (
            cleaned
            .drop_duplicates(
                subset=["timestamp"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        duplicates_removed = (
            before - len(cleaned)
        )

        before = len(cleaned)

        cleaned = (
            cleaned
            .dropna(
                subset=self.REQUIRED_COLUMNS
            )
            .reset_index(drop=True)
        )

        missing_rows_removed = (
            before - len(cleaned)
        )

        before = len(cleaned)

        cleaned = self._remove_invalid_rows(
            cleaned
        ).reset_index(drop=True)

        invalid_rows_removed = (
            before - len(cleaned)
        )

        if cleaned.empty:
            raise DataLoadError(
                "Cleaning produced an empty dataset"
            )

        validation = self.validator.validate(
            cleaned
        )

        if not validation.valid:
            messages = "; ".join(
                issue.message
                for issue in validation.issues
            )

            raise DataLoadError(
                "Cleaned dataset failed validation: "
                f"{messages}"
            )

        return (
            cleaned,
            CleaningReport(
                asset=asset,
                input_rows=input_rows,
                output_rows=len(cleaned),
                duplicates_removed=duplicates_removed,
                missing_rows_removed=missing_rows_removed,
                invalid_rows_removed=invalid_rows_removed,
            ),
        )

    def clean_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> CleaningReport:
        input_file = Path(input_path)
        output_file = Path(output_path)

        dataframe = self.loader.load(
            input_file
        )

        cleaned, report = self.clean(
            dataframe,
            asset=input_file.stem.upper(),
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cleaned.to_csv(
            output_file,
            index=False,
        )

        return report

    def clean_directory(
        self,
        input_directory: str | Path,
        output_directory: str | Path,
    ) -> list[CleaningReport]:
        input_path = Path(input_directory)
        output_path = Path(output_directory)

        if not input_path.exists():
            raise DataLoadError(
                f"Dataset directory not found: {input_path}"
            )

        if not input_path.is_dir():
            raise DataLoadError(
                f"Dataset path is not a directory: {input_path}"
            )

        files = sorted(
            input_path.glob("*.csv")
        )

        if not files:
            raise DataLoadError(
                f"No CSV datasets found in: {input_path}"
            )

        reports: list[CleaningReport] = []

        for file_path in files:
            reports.append(
                self.clean_file(
                    file_path,
                    output_path / file_path.name,
                )
            )

        return reports

    def _validate_input(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise DataLoadError(
                "Cleaning input must be a pandas DataFrame"
            )

        if dataframe.empty:
            raise DataLoadError(
                "Cannot clean an empty dataset"
            )

        missing = (
            set(self.REQUIRED_COLUMNS)
            - set(dataframe.columns)
        )

        if missing:
            raise DataLoadError(
                "Cannot clean dataset with "
                f"missing columns: {sorted(missing)}"
            )

    @classmethod
    def _remove_invalid_rows(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        prices = dataframe[
            cls.PRICE_COLUMNS
        ]

        valid = ~(
            prices < 0
        ).any(axis=1)

        valid &= ~(
            prices == 0
        ).any(axis=1)

        valid &= (
            dataframe["volume"] >= 0
        )

        valid &= (
            dataframe["high"]
            >= dataframe["low"]
        )

        valid &= (
            dataframe["high"]
            >= dataframe["open"]
        )

        valid &= (
            dataframe["high"]
            >= dataframe["close"]
        )

        valid &= (
            dataframe["low"]
            <= dataframe["open"]
        )

        valid &= (
            dataframe["low"]
            <= dataframe["close"]
        )

        return dataframe.loc[
            valid
        ].copy()