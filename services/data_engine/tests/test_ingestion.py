from pathlib import Path

import pytest



from data_engine.ingestion import (
    DatasetResult,
    MarketDatasetIngestion,
)
from data_engine.loader import DataLoadError


RAW_DATA_DIR = Path("ml/datasets/raw")


def test_load_real_datasets() -> None:
    ingestion = MarketDatasetIngestion()

    results = ingestion.load_directory(RAW_DATA_DIR)

    assert len(results) == 5

    assets = {
        result.asset
        for result in results
    }

    assert assets == {
        "AAPL",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
    }


def test_valid_real_datasets_are_loaded() -> None:
    ingestion = MarketDatasetIngestion()

    results = ingestion.load_directory(RAW_DATA_DIR)

    successful = ingestion.successful(results)

    assets = {
        result.asset
        for result in successful
    }

    assert assets == {
        "AAPL",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
    }

    for result in successful:
        assert result.dataframe is not None
        assert not result.dataframe.empty


def test_invalid_real_dataset_is_reported() -> None:
    ingestion = MarketDatasetIngestion()

    results = ingestion.load_directory(RAW_DATA_DIR)

    failed = ingestion.failed(results)

    assert len(failed) == 0


def test_missing_directory_is_rejected(
    tmp_path: Path,
) -> None:
    ingestion = MarketDatasetIngestion()

    missing_directory = (
        tmp_path / "does-not-exist"
    )

    with pytest.raises(
        DataLoadError,
        match="directory not found",
    ):
        ingestion.load_directory(missing_directory)


def test_empty_directory_is_rejected(
    tmp_path: Path,
) -> None:
    ingestion = MarketDatasetIngestion()

    with pytest.raises(
        DataLoadError,
        match="No CSV datasets found",
    ):
        ingestion.load_directory(tmp_path)