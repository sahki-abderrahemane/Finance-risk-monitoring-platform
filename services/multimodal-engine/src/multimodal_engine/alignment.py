"""Alignment utilities for Sentinel-AI multimodal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class AlignmentError(ValueError):
    """Raised when multimodal alignment cannot be completed."""


MARKET_COLUMNS = (
    "timestamp",
    "ticker",
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
    "target_return_1d",
    "target_direction_1d",
)

SENTIMENT_COLUMNS = (
    "timestamp",
    "ticker",
    "sentiment_score_mean",
    "sentiment_score_std",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "probability_negative_mean",
    "probability_neutral_mean",
    "probability_positive_mean",
    "confidence_mean",
)

VISION_COLUMNS = (
    "timestamp",
    "ticker",
    "sample_id",
    "image_path",
    "label",
    "future_return",
    "window_size",
)


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration controlling multimodal temporal alignment."""

    max_sentiment_gap_days: int = 1
    require_sentiment: bool = True
    require_vision: bool = True
    drop_missing_market: bool = True

    allowed_tickers: tuple[str, ...] = (
        "AAPL",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
    )


@dataclass(frozen=True)
class AlignmentReport:
    """Statistics describing the alignment process."""

    market_rows: int
    sentiment_rows: int
    vision_rows: int
    aligned_rows: int
    dropped_market_rows: int
    matched_sentiment: int
    matched_vision: int
    dropped_sentiment_rows: int
    dropped_vision_rows: int
    ticker_count: int

    def to_dict(self) -> dict[str, int]:
        """Convert the report into a serializable dictionary."""

        return {
            "market_rows": self.market_rows,
            "sentiment_rows": self.sentiment_rows,
            "vision_rows": self.vision_rows,
            "aligned_rows": self.aligned_rows,
            "dropped_market_rows": self.dropped_market_rows,
            "matched_sentiment": self.matched_sentiment,
            "matched_vision": self.matched_vision,
            "dropped_sentiment_rows": self.dropped_sentiment_rows,
            "dropped_vision_rows": self.dropped_vision_rows,
            "ticker_count": self.ticker_count,
        }


def _validate_columns(
    dataframe: pd.DataFrame,
    required: tuple[str, ...],
    dataset_name: str,
) -> None:
    """Validate required columns."""

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise AlignmentError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _canonicalize_timestamp(
    series: pd.Series,
    dataset_name: str,
) -> pd.Series:
    """
    Convert timestamps into canonical UTC-naive calendar dates.

    Multimodal alignment is performed at the trading-date level.
    Timezone representation must therefore not prevent an otherwise
    valid market/chart match.
    """

    timestamps = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if timestamps.isna().any():
        invalid_count = int(timestamps.isna().sum())

        raise AlignmentError(
            f"{dataset_name} contains {invalid_count} invalid timestamps."
        )

    return timestamps.dt.normalize().dt.tz_localize(None)


def _prepare_market(
    dataframe: pd.DataFrame,
    config: AlignmentConfig,
) -> pd.DataFrame:
    """Validate and prepare market features."""

    _validate_columns(
        dataframe,
        MARKET_COLUMNS,
        "Market dataset",
    )

    frame = dataframe.copy()

    frame["timestamp"] = _canonicalize_timestamp(
        frame["timestamp"],
        "Market dataset",
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame = frame[
        frame["ticker"].isin(config.allowed_tickers)
    ].copy()

    if frame.empty:
        raise AlignmentError(
            "No allowed tickers remain in market dataset."
        )

    duplicate_mask = frame.duplicated(
        subset=["ticker", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        raise AlignmentError(
            "Market dataset contains duplicate "
            "(ticker, timestamp) keys."
        )

    numeric_columns = [
        column
        for column in MARKET_COLUMNS
        if column not in {"timestamp", "ticker"}
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[numeric_columns].isna().any().any():
        raise AlignmentError(
            "Market dataset contains missing/non-numeric "
            "feature or target values."
        )

    return frame.sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)


def _prepare_sentiment(
    dataframe: pd.DataFrame,
    config: AlignmentConfig,
) -> pd.DataFrame:
    """Validate and prepare sentiment features."""

    _validate_columns(
        dataframe,
        SENTIMENT_COLUMNS,
        "Sentiment dataset",
    )

    frame = dataframe.copy()

    frame["timestamp"] = _canonicalize_timestamp(
        frame["timestamp"],
        "Sentiment dataset",
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame = frame[
        frame["ticker"].isin(config.allowed_tickers)
    ].copy()

    if frame.empty:
        return frame

    duplicate_mask = frame.duplicated(
        subset=["ticker", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        raise AlignmentError(
            "Sentiment dataset contains duplicate "
            "(ticker, timestamp) keys."
        )

    numeric_columns = [
        column
        for column in SENTIMENT_COLUMNS
        if column not in {"timestamp", "ticker"}
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[numeric_columns].isna().any().any():
        raise AlignmentError(
            "Sentiment dataset contains missing/non-numeric values."
        )

    return frame.sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)


def _prepare_vision(
    dataframe: pd.DataFrame,
    config: AlignmentConfig,
) -> pd.DataFrame:
    """Validate and prepare chart metadata."""

    _validate_columns(
        dataframe,
        VISION_COLUMNS,
        "Vision dataset",
    )

    frame = dataframe.copy()

    frame["timestamp"] = _canonicalize_timestamp(
        frame["timestamp"],
        "Vision dataset",
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame = frame[
        frame["ticker"].isin(config.allowed_tickers)
    ].copy()

    if frame.empty:
        return frame

    duplicate_mask = frame.duplicated(
        subset=["ticker", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        raise AlignmentError(
            "Vision dataset contains duplicate "
            "(ticker, timestamp) keys."
        )

    frame["future_return"] = pd.to_numeric(
        frame["future_return"],
        errors="coerce",
    )

    frame["window_size"] = pd.to_numeric(
        frame["window_size"],
        errors="coerce",
    )

    if frame[
        ["future_return", "window_size"]
    ].isna().any().any():
        raise AlignmentError(
            "Vision dataset contains invalid numeric metadata."
        )

    return frame.sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)


def _match_sentiment(
    market: pd.DataFrame,
    sentiment: pd.DataFrame,
    config: AlignmentConfig,
) -> pd.DataFrame:
    """Match nearest sentiment observation within the allowed gap."""

    if sentiment.empty:
        result = market.copy()

        for column in SENTIMENT_COLUMNS:
            if column not in {"timestamp", "ticker"}:
                result[column] = pd.NA

        return result

    left = market.sort_values(
        ["timestamp", "ticker"]
    ).copy()

    right = sentiment.sort_values(
        ["timestamp", "ticker"]
    ).copy()

    left["_market_timestamp"] = left["timestamp"]

    result = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="timestamp",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(
            days=config.max_sentiment_gap_days
        ),
        suffixes=("", "_sentiment"),
    )

    if "timestamp_sentiment" in result.columns:
        result = result.drop(
            columns=["timestamp_sentiment"]
        )

    return result


def align_multimodal_data(
    *,
    market: pd.DataFrame,
    sentiment: pd.DataFrame,
    vision: pd.DataFrame,
    config: AlignmentConfig | None = None,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """
    Align market, sentiment, and vision observations.

    Alignment is performed by canonical ticker + trading date.
    """

    cfg = config or AlignmentConfig()

    prepared_market = _prepare_market(
        market,
        cfg,
    )

    prepared_sentiment = _prepare_sentiment(
        sentiment,
        cfg,
    )

    prepared_vision = _prepare_vision(
        vision,
        cfg,
    )

    market_rows = len(prepared_market)
    sentiment_rows = len(prepared_sentiment)
    vision_rows = len(prepared_vision)

    sentiment_matched = _match_sentiment(
        prepared_market,
        prepared_sentiment,
        cfg,
    )

    sentiment_feature_columns = [
        column
        for column in SENTIMENT_COLUMNS
        if column not in {"timestamp", "ticker"}
    ]

    if cfg.require_sentiment:
        sentiment_mask = (
            sentiment_matched[
                sentiment_feature_columns
            ]
            .notna()
            .all(axis=1)
        )

        matched_sentiment = int(
            sentiment_mask.sum()
        )

        sentiment_matched = sentiment_matched[
            sentiment_mask
        ].copy()
    else:
        matched_sentiment = int(
            sentiment_matched[
                sentiment_feature_columns
            ]
            .notna()
            .all(axis=1)
            .sum()
        )

    if prepared_vision.empty:
        vision_matched = sentiment_matched.copy()

        vision_matched["sample_id"] = pd.NA
        vision_matched["image_path"] = pd.NA
        vision_matched["label"] = pd.NA
        vision_matched["future_return"] = pd.NA
        vision_matched["window_size"] = pd.NA

        matched_vision = 0

    else:
        vision_columns = [
            "timestamp",
            "ticker",
            "sample_id",
            "image_path",
            "label",
            "future_return",
            "window_size",
        ]

        vision_matched = sentiment_matched.merge(
            prepared_vision[vision_columns],
            on=["ticker", "timestamp"],
            how=(
                "inner"
                if cfg.require_vision
                else "left"
            ),
            validate="one_to_one",
        )

        matched_vision = int(
            vision_matched[
                "sample_id"
            ].notna().sum()
        )

    if cfg.require_vision and vision_matched.empty:
        aligned = vision_matched.copy()
    else:
        aligned = vision_matched.copy()

    if cfg.require_sentiment:
        aligned = aligned.dropna(
            subset=sentiment_feature_columns
        )

    if cfg.require_vision:
        aligned = aligned.dropna(
            subset=[
                "sample_id",
                "image_path",
                "future_return",
                "window_size",
            ]
        )

    aligned = aligned.sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)

    aligned_rows = len(aligned)

    dropped_market_rows = max(
        0,
        market_rows - aligned_rows,
    )

    dropped_sentiment_rows = max(
        0,
        sentiment_rows - matched_sentiment,
    )

    dropped_vision_rows = max(
        0,
        vision_rows - matched_vision,
    )

    report = AlignmentReport(
        market_rows=market_rows,
        sentiment_rows=sentiment_rows,
        vision_rows=vision_rows,
        aligned_rows=aligned_rows,
        dropped_market_rows=dropped_market_rows,
        matched_sentiment=matched_sentiment,
        matched_vision=matched_vision,
        dropped_sentiment_rows=dropped_sentiment_rows,
        dropped_vision_rows=dropped_vision_rows,
        ticker_count=aligned["ticker"].nunique()
        if not aligned.empty
        else 0,
    )

    return aligned, report


def load_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Load a CSV file."""

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {source}"
        )

    return pd.read_csv(source)


def save_aligned_dataset(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save aligned multimodal data."""

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        destination,
        index=False,
    )

    return destination


def build_sample_from_row(
    row: pd.Series,
) -> dict[str, object]:
    """Convert an aligned dataframe row into a model-ready record."""

    return {
        "timestamp": row["timestamp"],
        "ticker": row["ticker"],
        "market": {
            column: row[column]
            for column in (
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
        },
        "sentiment": {
            column: row[column]
            for column in (
                "sentiment_score_mean",
                "sentiment_score_std",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
                "probability_negative_mean",
                "probability_neutral_mean",
                "probability_positive_mean",
                "confidence_mean",
            )
        },
        "vision": {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "window_size": row["window_size"],
            "future_return": row["future_return"],
            "label": row.get("label"),
        },
        "target_return_1d": row[
            "target_return_1d"
        ],
        "target_direction_1d": row[
            "target_direction_1d"
        ],
    }