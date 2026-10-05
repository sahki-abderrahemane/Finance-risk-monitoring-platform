"""Build the canonical Phase 2 multimodal fusion dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from multimodal_engine.alignment import (
    AlignmentConfig,
    AlignmentError,
    align_multimodal_data,
)
from multimodal_engine.fusion_dataset import (
    FusionDatasetError,
    validate_fusion_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_FEATURES_DIR = PROJECT_ROOT / "ml" / "datasets" / "features"
DEFAULT_SENTIMENT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
    / "sentiment_daily.csv"
)
DEFAULT_VISION_PATH = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "charts"
    / "chart_metadata.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "aligned"
    / "multimodal_dataset.csv"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "evaluation"
    / "multimodal_dataset_summary.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "evaluation"
    / "alignment_report.json"
)


TICKERS = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
)


class DatasetBuildError(RuntimeError):
    """Raised when multimodal dataset construction fails."""


def _find_market_feature_files(
    features_dir: Path,
) -> list[Path]:
    """Find Phase 1 feature CSV files."""

    if not features_dir.exists():
        raise DatasetBuildError(
            f"Feature directory does not exist: {features_dir}"
        )

    candidates = sorted(
        path
        for path in features_dir.glob("*.csv")
        if path.is_file()
    )

    selected: list[Path] = []

    for ticker in TICKERS:
        matches = [
            path
            for path in candidates
            if path.stem.upper().startswith(ticker)
            and "feature" in path.stem.lower()
        ]

        if not matches:
            raise DatasetBuildError(
                f"No feature CSV found for ticker '{ticker}' "
                f"under {features_dir}"
            )

        selected.append(matches[0])

    return selected


def _normalize_timestamp_column(
    dataframe: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Normalize common timestamp/date column names.

    This keeps the alignment layer strict while allowing upstream
    datasets to use either `timestamp` or `date`.
    """

    frame = dataframe.copy()

    if "timestamp" in frame.columns:
        column = "timestamp"
    elif "date" in frame.columns:
        column = "date"
    else:
        raise DatasetBuildError(
            f"{source_name} must contain 'timestamp' or 'date'."
        )

    frame["timestamp"] = pd.to_datetime(
        frame[column],
        errors="coerce",
    )

    if frame["timestamp"].isna().any():
        raise DatasetBuildError(
            f"{source_name} contains invalid timestamps."
        )

    if column != "timestamp":
        frame = frame.drop(columns=[column])

    return frame


def _normalize_market_features(
    feature_files: list[Path],
) -> pd.DataFrame:
    """Load and combine ticker-level Phase 1 feature datasets."""

    frames: list[pd.DataFrame] = []

    for path in feature_files:
        dataframe = pd.read_csv(path)

        if "ticker" not in dataframe.columns:
            ticker = path.stem.split("_")[0].upper()
            dataframe["ticker"] = ticker

        dataframe = _normalize_timestamp_column(
            dataframe,
            path.name,
        )

        frames.append(dataframe)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def _normalize_sentiment(
    sentiment_path: Path,
) -> pd.DataFrame:
    """Load daily sentiment aggregation."""

    if not sentiment_path.exists():
        raise DatasetBuildError(
            f"Sentiment dataset does not exist: {sentiment_path}"
        )

    dataframe = pd.read_csv(sentiment_path)

    if "ticker" not in dataframe.columns:
        raise DatasetBuildError(
            "Sentiment dataset must contain 'ticker'."
        )

    return _normalize_timestamp_column(
        dataframe,
        sentiment_path.name,
    )


def _normalize_vision(
    vision_path: Path,
) -> pd.DataFrame:
    """Load chart metadata."""

    if not vision_path.exists():
        raise DatasetBuildError(
            f"Vision metadata does not exist: {vision_path}"
        )

    dataframe = pd.read_csv(vision_path)

    if "ticker" not in dataframe.columns:
        raise DatasetBuildError(
            "Vision metadata must contain 'ticker'."
        )

    return _normalize_timestamp_column(
        dataframe,
        vision_path.name,
    )


def _build_summary(
    aligned: pd.DataFrame,
    feature_files: list[Path],
    sentiment_path: Path,
    vision_path: Path,
) -> dict[str, object]:
    """Create a reproducibility summary."""

    ticker_counts = (
        aligned["ticker"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    date_min = aligned["timestamp"].min()
    date_max = aligned["timestamp"].max()

    return {
        "dataset": "sentinel-ai-phase2-multimodal",
        "rows": int(len(aligned)),
        "columns": int(len(aligned.columns)),
        "tickers": sorted(aligned["ticker"].unique().tolist()),
        "rows_by_ticker": ticker_counts,
        "timestamp_min": date_min.isoformat(),
        "timestamp_max": date_max.isoformat(),
        "market_feature_files": [
            str(path)
            for path in feature_files
        ],
        "sentiment_file": str(sentiment_path),
        "vision_metadata_file": str(vision_path),
        "target_columns": [
            "target_return_1d",
            "target_direction_1d",
        ],
        "vision_future_return_used_as_feature": False,
        "vision_label_used_as_feature": False,
        "data_is_historical_or_synthetic": True,
        "real_time_trading_enabled": False,
    }


def build_dataset(
    *,
    features_dir: Path = DEFAULT_FEATURES_DIR,
    sentiment_path: Path = DEFAULT_SENTIMENT_PATH,
    vision_path: Path = DEFAULT_VISION_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> pd.DataFrame:
    """Build and persist the canonical multimodal dataset."""

    feature_files = _find_market_feature_files(
        features_dir
    )

    market = _normalize_market_features(
        feature_files
    )

    sentiment = _normalize_sentiment(
        sentiment_path
    )

    vision = _normalize_vision(
        vision_path
    )

    config = AlignmentConfig(
        max_sentiment_gap_days=1,
        require_sentiment=True,
        require_vision=True,
        drop_missing_market=True,
        allowed_tickers=TICKERS,
    )

    try:
        aligned, report = align_multimodal_data(
            market=market,
            sentiment=sentiment,
            vision=vision,
            config=config,
        )
    except AlignmentError as exc:
        raise DatasetBuildError(
            f"Multimodal alignment failed: {exc}"
        ) from exc

    if aligned.empty:
        raise DatasetBuildError(
            "Multimodal alignment produced zero rows."
        )

    try:
        validated = validate_fusion_dataset(
            aligned,
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        validated.to_csv(
            output_path,
            index=False,
        )
        output = output_path
    except FusionDatasetError as exc:
        raise DatasetBuildError(
            f"Fusion dataset validation failed: {exc}"
        ) from exc

    summary = _build_summary(
        aligned,
        feature_files,
        sentiment_path,
        vision_path,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_dict = {
        "market_rows": report.market_rows,
        "sentiment_rows": report.sentiment_rows,
        "vision_rows": report.vision_rows,
        "aligned_rows": report.aligned_rows,
        "dropped_market_rows": report.dropped_market_rows,
        "matched_sentiment": report.matched_sentiment,
        "matched_vision": report.matched_vision,
        "dropped_sentiment_rows": report.dropped_sentiment_rows,
        "dropped_vision_rows": report.dropped_vision_rows,
        "ticker_count": report.ticker_count,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report_dict,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Multimodal dataset written to: {output}"
    )
    print(
        f"Rows: {len(aligned)}"
    )
    print(
        f"Columns: {len(aligned.columns)}"
    )
    print(
        f"Tickers: {sorted(aligned['ticker'].unique())}"
    )
    print()
    print("Alignment report:")
    print(f"  Market rows:         {report.market_rows:>8,}")
    print(f"  Sentiment rows:      {report.sentiment_rows:>8,}")
    print(f"  Vision rows:         {report.vision_rows:>8,}")
    print()
    print(f"  Market rows considered: {report.market_rows:>8,}")
    print(f"  Matched sentiment:      {report.matched_sentiment:>8,}")
    print(f"  Matched vision:         {report.matched_vision:>8,}")
    print(f"  Final aligned rows:     {report.aligned_rows:>8,}")
    print()
    print(f"  Dropped (market):    {report.dropped_market_rows:>8,}")
    print(f"  Dropped (sentiment): {report.dropped_sentiment_rows:>8,}")
    print(f"  Dropped (vision):    {report.dropped_vision_rows:>8,}")
    print(f"  Ticker count:        {report.ticker_count:>8,}")
    print()
    print(f"Summary written to:   {summary_path}")
    print(f"Report written to:    {report_path}")

    return aligned


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build Sentinel-AI Phase 2 multimodal "
            "alignment dataset."
        )
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
    )

    parser.add_argument(
        "--sentiment",
        type=Path,
        default=DEFAULT_SENTIMENT_PATH,
    )

    parser.add_argument(
        "--vision",
        type=Path,
        default=DEFAULT_VISION_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()

    build_dataset(
        features_dir=args.features_dir,
        sentiment_path=args.sentiment,
        vision_path=args.vision,
        output_path=args.output,
        summary_path=args.summary,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()