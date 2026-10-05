from __future__ import annotations

import pandas as pd

from nlp_engine.aggregate_sentiment import (
    aggregate_by_ticker,
    aggregate_daily,
    prepare_predictions,
)


def _sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2024-01-02T09:00:00Z",
                "2024-01-02T10:00:00Z",
                "2024-01-03T09:00:00Z",
                "2024-01-02T11:00:00Z",
            ],
            "ticker": [
                "aapl",
                "AAPL",
                "AAPL",
                "MSFT",
            ],
            "predicted_label": [
                "positive",
                "negative",
                "neutral",
                "positive",
            ],
        }
    )


def test_prepare_predictions_normalizes_tickers() -> None:
    result = prepare_predictions(_sample_predictions())

    assert result["ticker"].tolist() == [
        "AAPL",
        "AAPL",
        "AAPL",
        "MSFT",
    ]


def test_prepare_predictions_creates_sentiment_scores() -> None:
    result = prepare_predictions(_sample_predictions())

    assert result["sentiment_score"].tolist() == [
        1.0,
        -1.0,
        0.0,
        1.0,
    ]


def test_prepare_predictions_creates_date() -> None:
    result = prepare_predictions(_sample_predictions())

    assert "date" in result.columns
    assert result["date"].notna().all()


def test_ticker_aggregation() -> None:
    prepared = prepare_predictions(_sample_predictions())
    result = aggregate_by_ticker(prepared)

    assert result["ticker"].tolist() == ["AAPL", "MSFT"]

    aapl = result[result["ticker"] == "AAPL"].iloc[0]

    assert aapl["article_count"] == 3
    assert aapl["positive_ratio"] == 1 / 3
    assert aapl["neutral_ratio"] == 1 / 3
    assert aapl["negative_ratio"] == 1 / 3
    assert aapl["sentiment_score_mean"] == 0.0


def test_daily_aggregation() -> None:
    prepared = prepare_predictions(_sample_predictions())
    result = aggregate_daily(prepared)

    assert len(result) == 3

    aapl_day = result[
        (result["ticker"] == "AAPL")
        & (result["date"] == pd.Timestamp("2024-01-02"))
    ].iloc[0]

    assert aapl_day["article_count"] == 2
    assert aapl_day["sentiment_score_mean"] == 0.0


def test_aggregation_contains_required_features() -> None:
    prepared = prepare_predictions(_sample_predictions())
    result = aggregate_by_ticker(prepared)

    required_columns = {
        "ticker",
        "article_count",
        "sentiment_score_mean",
        "sentiment_score_std",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
    }

    assert required_columns.issubset(result.columns)
    