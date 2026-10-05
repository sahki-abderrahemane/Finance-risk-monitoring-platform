from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

INPUT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
    / "finbert_test_predictions.csv"
)

OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
)

TICKER_OUTPUT_PATH: Final[Path] = (
    OUTPUT_DIR / "sentiment_by_ticker.csv"
)

DAILY_OUTPUT_PATH: Final[Path] = (
    OUTPUT_DIR / "sentiment_daily.csv"
)


class SentimentAggregationError(RuntimeError):
    """Raised when sentiment aggregation cannot be completed."""


def load_predictions(
    path: Path = INPUT_PATH,
) -> pd.DataFrame:
    """Load FinBERT prediction artifacts."""

    if not path.exists():
        raise SentimentAggregationError(
            f"Prediction file does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise SentimentAggregationError(
            f"Prediction file is empty: {path}"
        )

    return dataframe


def _validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the FinBERT prediction artifact contract."""

    required_columns = {
        "timestamp",
        "ticker",
        "predicted_sentiment",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "confidence",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise SentimentAggregationError(
            "FinBERT prediction artifact is missing "
            f"required columns: {sorted(missing_columns)}. "
            f"Available columns: {list(dataframe.columns)}"
        )


def _sentiment_score(
    sentiment: str,
) -> float:
    """
    Convert sentiment into a numerical score.

    positive -> +1
    neutral  ->  0
    negative -> -1
    """

    mapping = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
    }

    return mapping.get(
        sentiment.strip().lower(),
        0.0,
    )


def prepare_predictions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and enrich FinBERT predictions."""

    _validate_required_columns(dataframe)

    result = dataframe.copy()

    result["ticker"] = (
        result["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result["predicted_sentiment"] = (
        result["predicted_sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
        utc=True,
    )

    if result["timestamp"].isna().all():
        raise SentimentAggregationError(
            "No valid timestamps were found."
        )

    result = result.dropna(
        subset=["timestamp"]
    ).copy()

    probability_columns = [
        "probability_negative",
        "probability_neutral",
        "probability_positive",
    ]

    for column in probability_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result["confidence"] = pd.to_numeric(
        result["confidence"],
        errors="coerce",
    )

    result["date"] = (
        result["timestamp"]
        .dt.floor("D")
    )

    result["sentiment_score"] = (
        result["predicted_sentiment"]
        .map(_sentiment_score)
    )

    result["positive_probability"] = (
        result["probability_positive"]
    )

    result["neutral_probability"] = (
        result["probability_neutral"]
    )

    result["negative_probability"] = (
        result["probability_negative"]
    )

    result["positive_flag"] = (
        result["predicted_sentiment"] == "positive"
    ).astype(int)

    result["neutral_flag"] = (
        result["predicted_sentiment"] == "neutral"
    ).astype(int)

    result["negative_flag"] = (
        result["predicted_sentiment"] == "negative"
    ).astype(int)

    return result


def aggregate_by_ticker(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sentiment and probability features by ticker.
    """

    grouped = (
        dataframe
        .groupby("ticker", as_index=False)
        .agg(
            article_count=(
                "predicted_sentiment",
                "size",
            ),
            sentiment_score_mean=(
                "sentiment_score",
                "mean",
            ),
            sentiment_score_std=(
                "sentiment_score",
                "std",
            ),
            positive_ratio=(
                "positive_flag",
                "mean",
            ),
            neutral_ratio=(
                "neutral_flag",
                "mean",
            ),
            negative_ratio=(
                "negative_flag",
                "mean",
            ),
            probability_positive_mean=(
                "positive_probability",
                "mean",
            ),
            probability_neutral_mean=(
                "neutral_probability",
                "mean",
            ),
            probability_negative_mean=(
                "negative_probability",
                "mean",
            ),
            confidence_mean=(
                "confidence",
                "mean",
            ),
        )
    )

    grouped["sentiment_score_std"] = (
        grouped["sentiment_score_std"]
        .fillna(0.0)
    )

    return (
        grouped
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def aggregate_daily(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sentiment and probability features
    by ticker and day.
    """

    grouped = (
        dataframe
        .groupby(
            ["ticker", "date"],
            as_index=False,
        )
        .agg(
            article_count=(
                "predicted_sentiment",
                "size",
            ),
            sentiment_score_mean=(
                "sentiment_score",
                "mean",
            ),
            sentiment_score_std=(
                "sentiment_score",
                "std",
            ),
            positive_ratio=(
                "positive_flag",
                "mean",
            ),
            neutral_ratio=(
                "neutral_flag",
                "mean",
            ),
            negative_ratio=(
                "negative_flag",
                "mean",
            ),
            probability_positive_mean=(
                "positive_probability",
                "mean",
            ),
            probability_neutral_mean=(
                "neutral_probability",
                "mean",
            ),
            probability_negative_mean=(
                "negative_probability",
                "mean",
            ),
            confidence_mean=(
                "confidence",
                "mean",
            ),
        )
    )

    grouped["sentiment_score_std"] = (
        grouped["sentiment_score_std"]
        .fillna(0.0)
    )

    return (
        grouped
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    """Run sentiment aggregation."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions()

    prepared = prepare_predictions(
        predictions
    )

    ticker_features = aggregate_by_ticker(
        prepared
    )

    daily_features = aggregate_daily(
        prepared
    )

    ticker_features.to_csv(
        TICKER_OUTPUT_PATH,
        index=False,
    )

    daily_features.to_csv(
        DAILY_OUTPUT_PATH,
        index=False,
    )

    print("Sentiment aggregation completed.")
    print(
        f"Articles: {len(prepared)}"
    )
    print(
        f"Tickers:  {prepared['ticker'].nunique()}"
    )
    print(
        f"Ticker features: {TICKER_OUTPUT_PATH}"
    )
    print(
        f"Daily features:  {DAILY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()