from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vision_engine.schemas import ChartLabel


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DATA_DIR = PROJECT_ROOT / "ml" / "datasets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "datasets" / "charts" / "processed"

METADATA_PATH = PROJECT_ROOT / "ml" / "datasets" / "charts" / "chart_metadata.csv"
SUMMARY_PATH = PROJECT_ROOT / "ml" / "datasets" / "charts" / "dataset_summary.json"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

WINDOW_SIZE = 30
FUTURE_HORIZON = 5

UP_THRESHOLD = 0.01
DOWN_THRESHOLD = -0.01


class ChartGenerationError(RuntimeError):
    """Raised when the chart dataset cannot be generated."""


@dataclass(frozen=True)
class ChartGenerationConfig:
    """Configuration for chart image generation."""

    window_size: int = WINDOW_SIZE
    future_horizon: int = FUTURE_HORIZON
    up_threshold: float = UP_THRESHOLD
    down_threshold: float = DOWN_THRESHOLD
    image_width: float = 6.0
    image_height: float = 4.0
    dpi: int = 100

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        if self.up_threshold <= self.down_threshold:
            raise ValueError(
                "up_threshold must be greater than down_threshold"
            )


def _canonicalize_column_name(column: object) -> str:
    """
    Convert arbitrary dataframe column names into normalized identifiers.

    Examples:
        'Close' -> 'close'
        'Close_AAPL' -> 'close_aapl'
        'Price_Ticker' -> 'price_ticker'
    """
    value = str(column).strip().lower()

    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe column names."""
    dataframe = dataframe.copy()

    dataframe.columns = [
        _canonicalize_column_name(column)
        for column in dataframe.columns
    ]

    return dataframe


def _extract_ohlcv_columns(
    dataframe: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Convert flattened yfinance columns into canonical OHLCV columns.

    Handles formats such as:

        close_aapl
        high_aapl
        low_aapl
        open_aapl
        volume_aapl

    as well as already-canonical:

        close
        high
        low
        open
        volume
    """
    dataframe = dataframe.copy()

    normalized = {
        _canonicalize_column_name(column): column
        for column in dataframe.columns
    }

    required = ["open", "high", "low", "close", "volume"]

    selected_columns: dict[str, str] = {}

    for field in required:
        # Exact canonical column first.
        if field in normalized:
            selected_columns[field] = normalized[field]
            continue

        # Look for columns such as:
        # open_aapl
        # aapl_open
        # open_msft
        candidates = [
            column
            for column in normalized
            if (
                column.startswith(f"{field}_")
                or column.endswith(f"_{field}")
                or f"_{field}_" in column
            )
        ]

        # Prefer columns containing the current ticker.
        ticker_lower = ticker.lower()

        ticker_candidates = [
            column
            for column in candidates
            if ticker_lower in column
        ]

        if ticker_candidates:
            selected_columns[field] = normalized[ticker_candidates[0]]
        elif candidates:
            selected_columns[field] = normalized[candidates[0]]

    missing = [
        field
        for field in required
        if field not in selected_columns
    ]

    if missing:
        raise ChartGenerationError(
            f"{ticker} is missing OHLCV columns: {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    result = dataframe[
        [selected_columns[field] for field in required]
    ].copy()

    result.columns = required

    return result


def _load_flat_csv(path: Path, ticker: str) -> pd.DataFrame:
    """
    Load a conventional flat OHLCV CSV.

    Expected examples:

        timestamp,open,high,low,close,volume

    or:

        Date,Open,High,Low,Close,Volume
    """
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise ChartGenerationError(
            f"Could not read CSV {path}: {exc}"
        ) from exc

    dataframe = _normalize_columns(dataframe)

    timestamp_candidates = [
        "timestamp",
        "date",
        "datetime",
    ]

    timestamp_column = next(
        (
            column
            for column in timestamp_candidates
            if column in dataframe.columns
        ),
        None,
    )

    if timestamp_column is None:
        return pd.DataFrame()

    ohlcv = _extract_ohlcv_columns(
        dataframe,
        ticker,
    )

    result = pd.concat(
        [
            dataframe[timestamp_column].rename("timestamp"),
            ohlcv,
        ],
        axis=1,
    )

    return result


def _load_yfinance_csv(path: Path, ticker: str) -> pd.DataFrame:
    """
    Load a yfinance-style CSV.

    Sentinel's raw files may contain multi-level headers that become
    flattened into columns such as:

        price_ticker
        close_aapl
        high_aapl
        low_aapl
        open_aapl
        volume_aapl

    This loader explicitly recognizes that representation.
    """

    try:
        raw = pd.read_csv(path, header=None)
    except Exception as exc:
        raise ChartGenerationError(
            f"Could not read yfinance CSV {path}: {exc}"
        ) from exc

    if raw.empty or raw.shape[0] < 2:
        raise ChartGenerationError(
            f"yfinance CSV {path} does not contain enough rows."
        )

    # ------------------------------------------------------------------
    # First attempt:
    # The file may already be represented as a flat dataframe with
    # columns such as close_aapl.
    # ------------------------------------------------------------------
    flat = pd.read_csv(path)

    flat = _normalize_columns(flat)

    timestamp_column = next(
        (
            column
            for column in ["timestamp", "date", "datetime"]
            if column in flat.columns
        ),
        None,
    )

    if timestamp_column is not None:
        try:
            ohlcv = _extract_ohlcv_columns(flat, ticker)

            result = pd.concat(
                [
                    flat[timestamp_column].rename("timestamp"),
                    ohlcv,
                ],
                axis=1,
            )

            return result
        except ChartGenerationError:
            pass

    # ------------------------------------------------------------------
    # Second attempt:
    # Parse the original multi-row yfinance structure.
    #
    # Typical structure:
    #
    # Date       Price  Close  High  Low  Open  Volume
    # Ticker     Ticker AAPL   AAPL  AAPL AAPL  AAPL
    # 2020-...   ...
    # ------------------------------------------------------------------

    first_row = [
        _canonicalize_column_name(value)
        for value in raw.iloc[0].tolist()
    ]

    second_row = [
        _canonicalize_column_name(value)
        for value in raw.iloc[1].tolist()
    ]

    # Find the timestamp/date column.
    date_index: int | None = None

    for index, value in enumerate(first_row):
        if value in {"date", "timestamp", "datetime"}:
            date_index = index
            break

    if date_index is None:
        for index, value in enumerate(second_row):
            if value in {"date", "timestamp", "datetime"}:
                date_index = index
                break

    # If the date column was not explicitly named, assume the first
    # column is the timestamp column.
    if date_index is None:
        date_index = 0

    # Identify OHLCV columns.
    field_indices: dict[str, int] = {}

    for field in ["open", "high", "low", "close", "volume"]:
        for index, (level_one, level_two) in enumerate(
            zip(first_row, second_row)
        ):
            if level_one == field or level_two == field:
                field_indices[field] = index
                break

    missing = [
        field
        for field in ["open", "high", "low", "close", "volume"]
        if field not in field_indices
    ]

    if missing:
        raise ChartGenerationError(
            f"Could not parse yfinance OHLCV columns in {path}. "
            f"Missing: {missing}. "
            f"Available columns: {list(flat.columns)}"
        )

    data_start = 2

    records = raw.iloc[data_start:].copy()

    result = pd.DataFrame(
        {
            "timestamp": records.iloc[:, date_index],
            "open": records.iloc[:, field_indices["open"]],
            "high": records.iloc[:, field_indices["high"]],
            "low": records.iloc[:, field_indices["low"]],
            "close": records.iloc[:, field_indices["close"]],
            "volume": records.iloc[:, field_indices["volume"]],
        }
    )

    return result


def load_market_data(path: Path, ticker: str) -> pd.DataFrame:
    """
    Load and normalize market data into the canonical OHLCV schema.
    """

    if not path.exists():
        raise ChartGenerationError(
            f"Market data file does not exist: {path}"
        )

    try:
        dataframe = _load_flat_csv(path, ticker)

        if not dataframe.empty:
            return _clean_market_data(dataframe, ticker)

    except ChartGenerationError:
        pass

    dataframe = _load_yfinance_csv(path, ticker)

    return _clean_market_data(dataframe, ticker)


def _clean_market_data(
    dataframe: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Normalize timestamps and numeric OHLCV values."""

    dataframe = dataframe.copy()

    if "timestamp" not in dataframe.columns:
        raise ChartGenerationError(
            f"{ticker} is missing timestamp column."
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=["timestamp"] + numeric_columns
    )

    dataframe = dataframe.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    dataframe = dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if dataframe.empty:
        raise ChartGenerationError(
            f"{ticker} contains no valid OHLCV rows after cleaning."
        )

    return dataframe


def classify_future_return(
    future_return: float,
    config: ChartGenerationConfig,
) -> ChartLabel:
    """Convert future return into the CNN classification label."""

    if future_return >= config.up_threshold:
        return ChartLabel.UP

    if future_return <= config.down_threshold:
        return ChartLabel.DOWN

    return ChartLabel.FLAT


def create_chart(
    window: pd.DataFrame,
    output_path: Path,
    config: ChartGenerationConfig,
) -> None:
    """Create a single historical price chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(
            config.image_width,
            config.image_height,
        )
    )

    axis = figure.add_subplot(111)

    axis.plot(
        range(len(window)),
        window["close"].to_numpy(),
        linewidth=2.0,
    )

    axis.set_xticks([])
    axis.set_yticks([])

    axis.set_xlabel("")
    axis.set_ylabel("")

    figure.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    figure.savefig(
        output_path,
        dpi=config.dpi,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(figure)


def generate_ticker_charts(
    ticker: str,
    config: ChartGenerationConfig,
) -> list[dict[str, object]]:
    """Generate chart samples for one ticker."""

    input_path = RAW_DATA_DIR / f"{ticker}.csv"

    print(f"Generating charts for {ticker}...")

    dataframe = load_market_data(
        input_path,
        ticker,
    )

    minimum_required_rows = (
        config.window_size + config.future_horizon
    )

    if len(dataframe) < minimum_required_rows:
        raise ChartGenerationError(
            f"{ticker} has only {len(dataframe)} valid rows. "
            f"At least {minimum_required_rows} rows are required."
        )

    ticker_output_dir = OUTPUT_DIR / ticker
    ticker_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples: list[dict[str, object]] = []

    max_start = (
        len(dataframe)
        - config.window_size
        - config.future_horizon
        + 1
    )

    for start_index in range(max_start):
        window_end = (
            start_index
            + config.window_size
        )

        future_end = (
            window_end
            + config.future_horizon
        )

        window = dataframe.iloc[
            start_index:window_end
        ].copy()

        future_window = dataframe.iloc[
            window_end:future_end
        ]

        if future_window.empty:
            continue

        current_close = float(
            window.iloc[-1]["close"]
        )

        future_close = float(
            future_window.iloc[-1]["close"]
        )

        if current_close == 0:
            continue

        future_return = (
            future_close / current_close
        ) - 1.0

        label = classify_future_return(
            future_return,
            config,
        )

        timestamp = pd.Timestamp(
            window.iloc[-1]["timestamp"]
        )

        timestamp_string = timestamp.strftime(
            "%Y%m%d"
        )

        sample_id = (
            f"{ticker}_"
            f"{timestamp_string}_"
            f"{start_index:05d}"
        )

        image_path = (
            ticker_output_dir
            / f"{sample_id}.png"
        )

        create_chart(
            window,
            image_path,
            config,
        )

        samples.append(
            {
                "sample_id": sample_id,
                "ticker": ticker,
                "timestamp": timestamp.isoformat(),
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "label": label.value,
                "future_return": float(future_return),
                "window_size": config.window_size,
            }
        )

    print(
        f"  Generated {len(samples)} charts for {ticker}."
    )

    return samples


def write_metadata(
    samples: list[dict[str, object]],
) -> None:
    """Write chart metadata CSV."""

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(samples)

    dataframe.to_csv(
        METADATA_PATH,
        index=False,
    )


def write_summary(
    samples: list[dict[str, object]],
) -> None:
    """Write dataset summary JSON."""

    dataframe = pd.DataFrame(samples)

    if dataframe.empty:
        summary = {
            "total_samples": 0,
            "tickers": {},
            "labels": {},
        }
    else:
        ticker_counts = (
            dataframe["ticker"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        label_counts = (
            dataframe["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        summary = {
            "total_samples": int(len(dataframe)),
            "tickers": {
                str(key): int(value)
                for key, value in ticker_counts.items()
            },
            "labels": {
                str(key): int(value)
                for key, value in label_counts.items()
            },
            "window_size": WINDOW_SIZE,
            "future_horizon": FUTURE_HORIZON,
            "up_threshold": UP_THRESHOLD,
            "down_threshold": DOWN_THRESHOLD,
        }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


def main() -> None:
    """Generate the complete CNN chart dataset."""

    print(
        "Starting Sentinel-AI chart dataset generation..."
    )

    config = ChartGenerationConfig()

    all_samples: list[dict[str, object]] = []

    for ticker in TICKERS:
        samples = generate_ticker_charts(
            ticker,
            config,
        )

        all_samples.extend(samples)

    if not all_samples:
        raise ChartGenerationError(
            "No chart samples were generated."
        )

    write_metadata(all_samples)
    write_summary(all_samples)

    print()
    print(
        f"Chart generation completed successfully."
    )
    print(
        f"Total samples: {len(all_samples)}"
    )
    print(
        f"Metadata: {METADATA_PATH}"
    )
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()