from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

FEATURES_DIR: Final[Path] = (
    PROJECT_ROOT / "ml" / "datasets" / "features"
)

VISION_METADATA_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "charts"
    / "chart_metadata.csv"
)

SENTIMENT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
    / "sentiment_daily.csv"
)

OUTPUT_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "evaluation"
    / "alignment_diagnostic.json"
)

TICKERS: Final[tuple[str, ...]] = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
)


class DiagnosticError(RuntimeError):
    """Raised when the alignment diagnostic cannot be completed."""


def canonicalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Convert timestamps/dates into canonical timezone-naive daily timestamps.
    """
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())

        raise DiagnosticError(
            f"Found {invalid_count} invalid timestamp values."
        )

    return parsed.dt.normalize().dt.tz_localize(None)


def load_market_features() -> pd.DataFrame:
    """
    Load Phase 1 feature datasets.

    Ticker identity is derived from the filename because the feature
    CSVs do not contain a ticker column.

    Example:
        AAPL_features.csv -> AAPL
    """
    frames: list[pd.DataFrame] = []

    for ticker in TICKERS:
        path = FEATURES_DIR / f"{ticker}_features.csv"

        if not path.exists():
            raise DiagnosticError(
                f"Missing market feature dataset: {path}"
            )

        frame = pd.read_csv(path)

        if "timestamp" not in frame.columns:
            raise DiagnosticError(
                f"{path} is missing required column: ['timestamp']"
            )

        frame = frame.copy()

        frame["ticker"] = ticker

        frame["timestamp"] = canonicalize_timestamp(
            frame["timestamp"]
        )

        frame = frame[
            [
                "ticker",
                "timestamp",
            ]
        ]

        frame = frame.drop_duplicates(
            subset=[
                "ticker",
                "timestamp",
            ]
        )

        frames.append(frame)

    if not frames:
        raise DiagnosticError(
            "No market feature datasets were loaded."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        result
        .drop_duplicates(
            subset=[
                "ticker",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "ticker",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def load_vision_metadata() -> pd.DataFrame:
    """Load canonical chart metadata."""
    if not VISION_METADATA_PATH.exists():
        raise DiagnosticError(
            f"Missing vision metadata: "
            f"{VISION_METADATA_PATH}"
        )

    frame = pd.read_csv(
        VISION_METADATA_PATH
    )

    required = {
        "sample_id",
        "ticker",
        "timestamp",
        "image_path",
        "label",
        "future_return",
        "window_size",
    }

    missing = required - set(frame.columns)

    if missing:
        raise DiagnosticError(
            f"{VISION_METADATA_PATH} is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame["timestamp"] = canonicalize_timestamp(
        frame["timestamp"]
    )

    frame = frame.loc[
        frame["ticker"].isin(TICKERS)
    ].copy()

    return (
        frame
        .drop_duplicates(
            subset=[
                "ticker",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "ticker",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def load_sentiment() -> pd.DataFrame:
    """
    Load daily aggregated sentiment.

    The NLP aggregation schema uses `date`, not `timestamp`.
    We canonicalize `date` into the common `timestamp` column.
    """
    if not SENTIMENT_PATH.exists():
        raise DiagnosticError(
            f"Missing sentiment dataset: "
            f"{SENTIMENT_PATH}"
        )

    frame = pd.read_csv(
        SENTIMENT_PATH
    )

    required = {
        "ticker",
        "date",
        "sentiment_score_mean",
        "sentiment_score_std",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "probability_positive_mean",
        "probability_neutral_mean",
        "probability_negative_mean",
        "confidence_mean",
    }

    missing = required - set(frame.columns)

    if missing:
        raise DiagnosticError(
            f"{SENTIMENT_PATH} is missing required columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame["timestamp"] = canonicalize_timestamp(
        frame["date"]
    )

    frame = frame.loc[
        frame["ticker"].isin(TICKERS)
    ].copy()

    return (
        frame
        .drop_duplicates(
            subset=[
                "ticker",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "ticker",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def make_keys(
    frame: pd.DataFrame,
) -> set[tuple[str, pd.Timestamp]]:
    """Build canonical ticker/date keys."""
    return set(
        zip(
            frame["ticker"].astype(str),
            frame["timestamp"],
        )
    )


def date_coverage(
    frame: pd.DataFrame,
) -> dict[str, str | None]:
    """Return minimum and maximum canonical dates."""
    if frame.empty:
        return {
            "min": None,
            "max": None,
        }

    return {
        "min": frame["timestamp"].min().isoformat(),
        "max": frame["timestamp"].max().isoformat(),
    }


def intersection_by_ticker(
    market: pd.DataFrame,
    sentiment: pd.DataFrame,
    vision: pd.DataFrame,
) -> list[dict[str, int | str]]:
    """Calculate exact intersections per ticker."""
    results: list[dict[str, int | str]] = []

    for ticker in TICKERS:
        market_keys = make_keys(
            market.loc[
                market["ticker"] == ticker
            ]
        )

        sentiment_keys = make_keys(
            sentiment.loc[
                sentiment["ticker"] == ticker
            ]
        )

        vision_keys = make_keys(
            vision.loc[
                vision["ticker"] == ticker
            ]
        )

        market_sentiment = (
            market_keys
            & sentiment_keys
        )

        market_vision = (
            market_keys
            & vision_keys
        )

        all_three = (
            market_keys
            & sentiment_keys
            & vision_keys
        )

        results.append(
            {
                "ticker": ticker,
                "market_rows": len(
                    market_keys
                ),
                "sentiment_rows": len(
                    sentiment_keys
                ),
                "vision_rows": len(
                    vision_keys
                ),
                "market_sentiment": len(
                    market_sentiment
                ),
                "market_vision": len(
                    market_vision
                ),
                "all_three": len(
                    all_three
                ),
            }
        )

    return results


def diagnose() -> dict:
    """Run the complete alignment diagnostic."""
    market = load_market_features()
    sentiment = load_sentiment()
    vision = load_vision_metadata()

    market_keys = make_keys(
        market
    )

    sentiment_keys = make_keys(
        sentiment
    )

    vision_keys = make_keys(
        vision
    )

    market_sentiment = (
        market_keys
        & sentiment_keys
    )

    market_vision = (
        market_keys
        & vision_keys
    )

    all_three = (
        market_keys
        & sentiment_keys
        & vision_keys
    )

    return {
        "project_root": str(
            PROJECT_ROOT
        ),
        "dataset_paths": {
            "market": str(
                FEATURES_DIR
            ),
            "vision": str(
                VISION_METADATA_PATH
            ),
            "sentiment": str(
                SENTIMENT_PATH
            ),
        },
        "dataset_sizes": {
            "market": len(
                market
            ),
            "sentiment": len(
                sentiment
            ),
            "vision": len(
                vision
            ),
        },
        "global_date_coverage": {
            "market": date_coverage(
                market
            ),
            "sentiment": date_coverage(
                sentiment
            ),
            "vision": date_coverage(
                vision
            ),
        },
        "exact_intersections": {
            "market_keys": len(
                market_keys
            ),
            "sentiment_keys": len(
                sentiment_keys
            ),
            "vision_keys": len(
                vision_keys
            ),
            "market_sentiment": len(
                market_sentiment
            ),
            "market_vision": len(
                market_vision
            ),
            "all_three": len(
                all_three
            ),
        },
        "intersection_by_ticker": (
            intersection_by_ticker(
                market=market,
                sentiment=sentiment,
                vision=vision,
            )
        ),
        "diagnosis": {
            "vision_alignment_ok": (
                len(market_vision) > 0
            ),
            "nlp_alignment_available": (
                len(market_sentiment) > 0
            ),
            "full_multimodal_alignment_available": (
                len(all_three) > 0
            ),
        },
    }


def print_report(
    report: dict,
) -> None:
    """Print the diagnostic report."""
    sizes = report[
        "dataset_sizes"
    ]

    coverage = report[
        "global_date_coverage"
    ]

    intersections = report[
        "exact_intersections"
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DATASET SIZES"
    )

    print(
        "=" * 72
    )

    print(
        f"Market:    "
        f"{sizes['market']:,} rows"
    )

    print(
        f"Sentiment: "
        f"{sizes['sentiment']:,} rows"
    )

    print(
        f"Vision:    "
        f"{sizes['vision']:,} rows"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GLOBAL DATE COVERAGE"
    )

    print(
        "=" * 72
    )

    print(
        "Market:    "
        f"{coverage['market']['min']} "
        f"→ "
        f"{coverage['market']['max']}"
    )

    print(
        "Sentiment: "
        f"{coverage['sentiment']['min']} "
        f"→ "
        f"{coverage['sentiment']['max']}"
    )

    print(
        "Vision:    "
        f"{coverage['vision']['min']} "
        f"→ "
        f"{coverage['vision']['max']}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "EXACT TICKER/DATE INTERSECTIONS"
    )

    print(
        "=" * 72
    )

    print(
        f"Market keys:             "
        f"{intersections['market_keys']:,}"
    )

    print(
        f"Sentiment keys:          "
        f"{intersections['sentiment_keys']:,}"
    )

    print(
        f"Vision keys:             "
        f"{intersections['vision_keys']:,}"
    )

    print(
        f"Market + Sentiment:      "
        f"{intersections['market_sentiment']:,}"
    )

    print(
        f"Market + Vision:         "
        f"{intersections['market_vision']:,}"
    )

    print(
        f"All three:               "
        f"{intersections['all_three']:,}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "INTERSECTION BY TICKER"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Ticker':<10}"
        f"{'M+NLP':>10}"
        f"{'M+Vision':>12}"
        f"{'All 3':>10}"
    )

    print(
        "-" * 42
    )

    for row in report[
        "intersection_by_ticker"
    ]:
        print(
            f"{row['ticker']:<10}"
            f"{row['market_sentiment']:>10}"
            f"{row['market_vision']:>12}"
            f"{row['all_three']:>10}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESULT"
    )

    print(
        "=" * 72
    )

    if intersections[
        "market_vision"
    ] > 0:
        print(
            "OK: Market + Vision alignment is working."
        )
    else:
        print(
            "ERROR: Market + Vision intersection "
            "is still 0 rows."
        )

    if intersections[
        "market_sentiment"
    ] > 0:
        print(
            "OK: Market + NLP has overlapping "
            "ticker/date keys."
        )
    else:
        print(
            "WARNING: Market + NLP intersection "
            "is 0 rows."
        )

    if intersections[
        "all_three"
    ] > 0:
        print(
            "OK: Full multimodal intersection "
            "is available."
        )
    else:
        print(
            "WARNING: Full multimodal intersection "
            "is 0 rows."
        )

        print(
            "Do not train the fusion model yet."
        )

    print(
        f"\nDiagnostic written to: "
        f"{OUTPUT_PATH}"
    )


def main() -> None:
    """CLI entry point."""
    try:
        report = diagnose()

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        print_report(
            report
        )

    except Exception as exc:
        raise SystemExit(
            f"\nDiagnostic failed: {exc}"
        ) from exc


if __name__ == "__main__":
    main()