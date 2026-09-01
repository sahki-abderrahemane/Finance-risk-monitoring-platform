from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from services.data_engine.src.data_engine.loader import (
    CSVMarketDataLoader,
    DataLoadError,
)
from services.data_engine.src.data_engine.schemas import MarketDataPoint


def create_csv(
    tmp_path: Path,
    content: str,
) -> Path:
    path = tmp_path / "market_data.csv"
    path.write_text(content)
    return path


# ============================================================
# Schema tests
# ============================================================


def test_valid_market_data_point() -> None:
    point = MarketDataPoint(
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1_000_000.0,
    )

    assert point.close == 105.0


def test_rejects_invalid_high() -> None:
    with pytest.raises(ValidationError):
        MarketDataPoint(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=90.0,
            low=85.0,
            close=105.0,
            volume=1_000_000.0,
        )


def test_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        MarketDataPoint(
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=-1.0,
        )


# ============================================================
# Flat CSV tests
# ============================================================


def test_load_valid_flat_csv(tmp_path: Path) -> None:
    path = create_csv(
        tmp_path,
        """date,open,high,low,close,volume
2025-01-01,100,110,95,105,1000000
2025-01-02,105,115,100,112,1200000
""",
    )

    loader = CSVMarketDataLoader()

    dataframe = loader.load(path)

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2

    assert list(dataframe.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert dataframe.iloc[0]["open"] == 100
    assert dataframe.iloc[0]["close"] == 105
    assert dataframe["timestamp"].dt.tz is not None


def test_load_canonical_timestamp_csv(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume
2025-01-01,100,110,95,105,1000000
""",
    )

    loader = CSVMarketDataLoader()

    dataframe = loader.load(path)

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["timestamp"].year == 2025


def test_rejects_missing_column(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """timestamp,open,high,low,close
2025-01-01,100,110,95,105
""",
    )

    loader = CSVMarketDataLoader()

    with pytest.raises(
        DataLoadError,
        match="volume",
    ):
        loader.load(path)


# ============================================================
# yfinance CSV tests
# ============================================================


def test_load_yfinance_format(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """Price,Close,High,Low,Open,Volume
Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
Date,,,,,
2025-01-01,105,110,95,100,1000000
2025-01-02,112,115,100,105,1200000
""",
    )

    loader = CSVMarketDataLoader()

    dataframe = loader.load(path)

    assert len(dataframe) == 2

    assert list(dataframe.columns) == [
        "timestamp",
        "close",
        "high",
        "low",
        "open",
        "volume",
    ]

    assert dataframe.iloc[0]["open"] == 100
    assert dataframe.iloc[0]["close"] == 105


# ============================================================
# Error handling tests
# ============================================================


def test_rejects_invalid_ohlc(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume
2025-01-01,100,90,95,105,1000000
""",
    )

    loader = CSVMarketDataLoader()

    with pytest.raises(DataLoadError):
        loader.load(path)


def test_rejects_negative_volume(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume
2025-01-01,100,110,95,105,-100
""",
    )

    loader = CSVMarketDataLoader()

    with pytest.raises(DataLoadError):
        loader.load(path)


def test_rejects_missing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "does_not_exist.csv"

    loader = CSVMarketDataLoader()

    with pytest.raises(
        DataLoadError,
        match="Dataset not found",
    ):
        loader.load(path)


def test_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.csv"
    path.touch()

    loader = CSVMarketDataLoader()

    with pytest.raises(
        DataLoadError,
        match="Dataset is empty",
    ):
        loader.load(path)


def test_rejects_unsupported_format(
    tmp_path: Path,
) -> None:
    path = create_csv(
        tmp_path,
        """foo,bar,baz
1,2,3
""",
    )

    loader = CSVMarketDataLoader()

    with pytest.raises(
        DataLoadError,
        match="Unsupported CSV format",
    ):
        loader.load(path)