from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nlp_engine.schemas import (
    NewsArticle,
    SentimentLabel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

FEATURES_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "raw"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "financial_news.csv"
)

TICKERS = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
)

SOURCES = (
    "Synthetic Financial Desk",
    "Synthetic Market Journal",
    "Synthetic Economic Wire",
    "Synthetic Research Bulletin",
)


@dataclass(frozen=True)
class NewsTemplate:
    """Template used to generate synthetic financial news."""

    label: SentimentLabel
    headlines: tuple[str, ...]
    bodies: tuple[str, ...]


TEMPLATES = (
    NewsTemplate(
        label=SentimentLabel.POSITIVE,
        headlines=(
            "{ticker} reports stronger-than-expected quarterly demand",
            "{ticker} expands investment following improved business conditions",
            "{ticker} sees positive momentum across key operating segments",
            "{ticker} announces progress in strategic growth initiative",
        ),
        bodies=(
            (
                "{ticker} reported stronger operating activity during the "
                "period, with management highlighting improving demand "
                "across several business areas."
            ),
            (
                "Management at {ticker} described recent operating trends "
                "as constructive and pointed to continued investment in "
                "long-term growth opportunities."
            ),
            (
                "Recent company updates indicate improving business "
                "conditions for {ticker}, supported by stronger activity "
                "in important operating segments."
            ),
        ),
    ),
    NewsTemplate(
        label=SentimentLabel.NEUTRAL,
        headlines=(
            "{ticker} maintains current operating outlook",
            "{ticker} provides routine quarterly business update",
            "{ticker} continues previously announced investment program",
            "{ticker} reports mixed conditions across operating segments",
        ),
        bodies=(
            (
                "{ticker} provided a routine business update, with "
                "management maintaining its existing operating outlook "
                "and previously announced plans."
            ),
            (
                "The latest update from {ticker} contained a mixture of "
                "stable operating indicators and incremental changes "
                "across business segments."
            ),
            (
                "{ticker} continues its existing operating initiatives "
                "while monitoring market conditions and changes in "
                "customer demand."
            ),
        ),
    ),
    NewsTemplate(
        label=SentimentLabel.NEGATIVE,
        headlines=(
            "{ticker} faces weaker-than-expected demand conditions",
            "{ticker} warns of pressure across key operating segments",
            "{ticker} reduces near-term outlook amid challenging conditions",
            "{ticker} reports increased operating uncertainty",
        ),
        bodies=(
            (
                "{ticker} reported weaker operating conditions, with "
                "management highlighting softer demand and increased "
                "uncertainty across several business areas."
            ),
            (
                "Recent updates from {ticker} indicate pressure on "
                "operating performance as market conditions become "
                "more challenging."
            ),
            (
                "{ticker} management warned that near-term business "
                "conditions remain difficult and that several operating "
                "segments are facing additional pressure."
            ),
        ),
    ),
)


def load_market_dates() -> dict[str, list[datetime]]:
    """
    Load actual trading dates from Phase 1 feature datasets.

    Only timestamp/date information is used. Market targets and feature
    values are deliberately ignored so synthetic news generation cannot
    leak future market information into the NLP dataset.
    """

    ticker_dates: dict[str, list[datetime]] = {}

    for ticker in TICKERS:
        feature_file = (
            FEATURES_DIR
            / f"{ticker}_features.csv"
        )

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Feature file not found for {ticker}: "
                f"{feature_file}"
            )

        frame = pd.read_csv(
            feature_file,
            usecols=["timestamp"],
        )

        if frame.empty:
            raise ValueError(
                f"Feature file contains no rows: {feature_file}"
            )

        timestamps = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                f"Invalid timestamp found in: {feature_file}"
            )

        dates = (
            timestamps
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        ticker_dates[ticker] = [
            timestamp.to_pydatetime().replace(
                tzinfo=None
            )
            for timestamp in dates
        ]

        if not ticker_dates[ticker]:
            raise ValueError(
                f"No valid trading dates found for {ticker}"
            )

    return ticker_dates


def generate_articles(
    *,
    seed: int = 42,
    articles_per_trading_day: int = 1,
) -> list[NewsArticle]:
    """
    Generate deterministic synthetic financial-news records.

    News dates are derived from the actual Phase 1 market trading dates.

    IMPORTANT:
    - Market feature values are never read.
    - target_return_1d is never read.
    - target_direction_1d is never read.
    - Synthetic sentiment is generated independently of market targets.
    """

    if articles_per_trading_day <= 0:
        raise ValueError(
            "articles_per_trading_day must be greater than zero"
        )

    rng = random.Random(seed)

    ticker_dates = load_market_dates()

    articles: list[NewsArticle] = []

    for ticker in TICKERS:
        dates = ticker_dates[ticker]

        for trading_date in dates:
            for article_index in range(
                articles_per_trading_day
            ):
                template = rng.choice(
                    TEMPLATES
                )

                headline_template = rng.choice(
                    template.headlines
                )

                body_template = rng.choice(
                    template.bodies
                )

                hour = rng.randint(
                    9,
                    16,
                )

                minute = rng.randint(
                    0,
                    59,
                )

                second = rng.randint(
                    0,
                    59,
                )

                timestamp = datetime(
                    trading_date.year,
                    trading_date.month,
                    trading_date.day,
                    hour,
                    minute,
                    second,
                )

                article = NewsArticle(
                    timestamp=timestamp,
                    ticker=ticker,
                    headline=headline_template.format(
                        ticker=ticker
                    ),
                    text=body_template.format(
                        ticker=ticker
                    ),
                    source=rng.choice(
                        SOURCES
                    ),
                    sentiment_label=(
                        template.label
                    ),
                )

                articles.append(article)

    articles.sort(
        key=lambda article: (
            article.timestamp,
            article.ticker,
        )
    )

    return articles


def write_csv(
    articles: list[NewsArticle],
    output_file: Path,
) -> None:
    """Write validated news records to CSV."""

    if not articles:
        raise ValueError(
            "Cannot write an empty article collection"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "timestamp",
        "ticker",
        "headline",
        "text",
        "source",
        "sentiment_label",
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for article in articles:
            writer.writerow(
                {
                    "timestamp": (
                        article.timestamp.isoformat()
                    ),
                    "ticker": article.ticker,
                    "headline": article.headline,
                    "text": article.text,
                    "source": article.source,
                    "sentiment_label": (
                        article.sentiment_label.value
                    ),
                }
            )


def main() -> None:
    """Generate the Phase 2 synthetic news dataset."""

    articles = generate_articles(
        seed=42,
        articles_per_trading_day=1,
    )

    write_csv(
        articles,
        OUTPUT_FILE,
    )

    labels = {
        label: 0
        for label in SentimentLabel
    }

    tickers = {
        ticker: 0
        for ticker in TICKERS
    }

    dates_by_ticker: dict[str, set[str]] = {
        ticker: set()
        for ticker in TICKERS
    }

    for article in articles:
        labels[
            article.sentiment_label
        ] += 1

        tickers[
            article.ticker
        ] += 1

        dates_by_ticker[
            article.ticker
        ].add(
            article.timestamp.date().isoformat()
        )

    print(
        "Synthetic financial-news dataset created."
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(articles)}"
    )

    print(
        "Tickers:"
    )

    for ticker, count in tickers.items():
        print(
            f"  {ticker}: {count} articles "
            f"across {len(dates_by_ticker[ticker])} dates"
        )

    print(
        "Sentiment distribution:"
    )

    for label, count in labels.items():
        print(
            f"  {label.value}: {count}"
        )


if __name__ == "__main__":
    main()