from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from nlp_engine.schemas import (
    NewsArticle,
    NewsDataset,
    SentimentLabel,
)


def make_article(
    ticker: str = "AAPL",
) -> NewsArticle:
    """Create a valid test article."""

    return NewsArticle(
        timestamp=datetime(
            2025,
            1,
            2,
            10,
            30,
        ),
        ticker=ticker,
        headline="Apple reports stable demand",
        text=(
            "The company reported stable operating "
            "conditions during the period."
        ),
        source="Synthetic Financial Desk",
        sentiment_label=SentimentLabel.NEUTRAL,
    )


def test_valid_article() -> None:
    """A valid article should pass validation."""

    article = make_article()

    assert article.ticker == "AAPL"
    assert (
        article.sentiment_label
        == SentimentLabel.NEUTRAL
    )


def test_ticker_is_normalized() -> None:
    """Ticker symbols should be normalized."""

    article = NewsArticle(
        timestamp=datetime(
            2025,
            1,
            2,
        ),
        ticker=" aapl ",
        headline="Headline",
        text="Financial text",
        source="Synthetic Source",
        sentiment_label=SentimentLabel.POSITIVE,
    )

    assert article.ticker == "AAPL"


def test_text_is_normalized() -> None:
    """Text fields should collapse surrounding whitespace."""

    article = NewsArticle(
        timestamp=datetime(
            2025,
            1,
            2,
        ),
        ticker="AAPL",
        headline="  Example   headline  ",
        text="  Example   article text  ",
        source="  Synthetic Source  ",
        sentiment_label=SentimentLabel.POSITIVE,
    )

    assert article.headline == "Example headline"
    assert article.text == "Example article text"
    assert article.source == "Synthetic Source"


def test_empty_ticker_is_rejected() -> None:
    """An empty ticker should fail validation."""

    with pytest.raises(ValidationError):
        NewsArticle(
            timestamp=datetime(
                2025,
                1,
                2,
            ),
            ticker="   ",
            headline="Headline",
            text="Text",
            source="Source",
            sentiment_label=SentimentLabel.NEUTRAL,
        )


def test_empty_headline_is_rejected() -> None:
    """An empty headline should fail validation."""

    with pytest.raises(ValidationError):
        NewsArticle(
            timestamp=datetime(
                2025,
                1,
                2,
            ),
            ticker="AAPL",
            headline="   ",
            text="Text",
            source="Source",
            sentiment_label=SentimentLabel.NEUTRAL,
        )


def test_invalid_sentiment_is_rejected() -> None:
    """Only the three defined sentiment classes are accepted."""

    with pytest.raises(ValidationError):
        NewsArticle(
            timestamp=datetime(
                2025,
                1,
                2,
            ),
            ticker="AAPL",
            headline="Headline",
            text="Text",
            source="Source",
            sentiment_label="bullish",
        )


def test_extra_fields_are_rejected() -> None:
    """Unexpected fields should fail the contract."""

    with pytest.raises(ValidationError):
        NewsArticle(
            timestamp=datetime(
                2025,
                1,
                2,
            ),
            ticker="AAPL",
            headline="Headline",
            text="Text",
            source="Source",
            sentiment_label=SentimentLabel.NEUTRAL,
            unexpected_field="invalid",
        )


def test_dataset_size() -> None:
    """Dataset size should reflect contained articles."""

    dataset = NewsDataset(
        articles=[
            make_article("AAPL"),
            make_article("MSFT"),
        ]
    )

    assert dataset.size == 2


def test_dataset_tickers() -> None:
    """Dataset tickers should be unique and sorted."""

    dataset = NewsDataset(
        articles=[
            make_article("NVDA"),
            make_article("AAPL"),
            make_article("NVDA"),
            make_article("MSFT"),
        ]
    )

    assert dataset.tickers == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]


def test_dataset_by_ticker() -> None:
    """Dataset filtering should be case-insensitive."""

    dataset = NewsDataset(
        articles=[
            make_article("AAPL"),
            make_article("MSFT"),
            make_article("AAPL"),
        ]
    )

    articles = dataset.by_ticker(
        "aapl"
    )

    assert len(articles) == 2

    assert all(
        article.ticker == "AAPL"
        for article in articles
    )


def test_empty_dataset_is_rejected() -> None:
    """An empty dataset should fail validation."""

    with pytest.raises(ValidationError):
        NewsDataset(
            articles=[]
        )