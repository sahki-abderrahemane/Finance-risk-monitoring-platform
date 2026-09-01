from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_engine.loader import (
    CSVMarketDataLoader,
    DataLoadError,
)


@dataclass(frozen=True)
class DatasetResult:
    """Result of loading one market dataset."""

    asset: str
    path: Path
    dataframe: pd.DataFrame | None
    error: str | None

    @property
    def success(self) -> bool:
        """Return True when the dataset loaded successfully."""
        return self.error is None and self.dataframe is not None


class MarketDatasetIngestion:
    """Load multiple market datasets from a raw-data directory."""

    def __init__(
        self,
        loader: CSVMarketDataLoader | None = None,
    ) -> None:
        self.loader = loader or CSVMarketDataLoader()

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[DatasetResult]:
        directory_path = Path(directory)

        if not directory_path.exists():
            raise DataLoadError(
                f"Raw dataset directory not found: "
                f"{directory_path}"
            )

        if not directory_path.is_dir():
            raise DataLoadError(
                f"Raw dataset path is not a directory: "
                f"{directory_path}"
            )

        files = sorted(directory_path.glob("*.csv"))

        if not files:
            raise DataLoadError(
                f"No CSV datasets found in: "
                f"{directory_path}"
            )

        results: list[DatasetResult] = []

        for file_path in files:
            asset = file_path.stem.upper()

            try:
                dataframe = self.loader.load(file_path)

                results.append(
                    DatasetResult(
                        asset=asset,
                        path=file_path,
                        dataframe=dataframe,
                        error=None,
                    )
                )

            except DataLoadError as exc:
                results.append(
                    DatasetResult(
                        asset=asset,
                        path=file_path,
                        dataframe=None,
                        error=str(exc),
                    )
                )

        return results

    @staticmethod
    def successful(
        results: list[DatasetResult],
    ) -> list[DatasetResult]:
        """Return successfully loaded datasets."""
        return [
            result
            for result in results
            if result.success
        ]

    @staticmethod
    def failed(
        results: list[DatasetResult],
    ) -> list[DatasetResult]:
        """Return datasets that failed to load."""
        return [
            result
            for result in results
            if not result.success
        ]