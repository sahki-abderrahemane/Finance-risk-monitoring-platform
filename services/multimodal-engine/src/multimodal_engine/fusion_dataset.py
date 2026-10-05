from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]


class FusionDatasetError(ValueError):
    """Raised when the multimodal fusion dataset is invalid."""


MARKET_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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

SENTIMENT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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

VISION_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "image_path",
    "window_size",
)

TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "target_return_1d",
    "target_direction_1d",
)

FORBIDDEN_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        # Direct market targets.
        "target_return_1d",
        "target_direction_1d",

        # Vision target-derived metadata.
        "future_return",
        "vision_future_return",
        "vision_label",

        # Any alternative target naming that might accidentally
        # appear in a future version of the dataset.
        "future_return_1d",
        "future_direction_1d",
        "future_direction",
        "label",
        "target",
    }
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "ticker",
    "timestamp",
    *MARKET_FEATURE_COLUMNS,
    *SENTIMENT_FEATURE_COLUMNS,
    *VISION_METADATA_COLUMNS,
    *TARGET_COLUMNS,
)

BINARY_TARGET_VALUES: Final[frozenset[int]] = frozenset(
    {0, 1}
)


@dataclass(frozen=True)
class FusionSplit:
    """Chronological train/validation/test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FusionFeatureGroups:
    """
    Explicit feature groups used by the fusion model.

    Vision is represented by image paths at this stage. The CNN embedding
    will be extracted in the model-training layer and must never expose
    future-derived chart metadata.
    """

    market: tuple[str, ...]
    sentiment: tuple[str, ...]
    vision: tuple[str, ...]


@dataclass(frozen=True)
class FusionDataset:
    """Validated, leakage-safe multimodal fusion dataset."""

    frame: pd.DataFrame
    feature_groups: FusionFeatureGroups

    @property
    def size(self) -> int:
        """Return the number of aligned samples."""
        return len(self.frame)

    @property
    def tickers(self) -> tuple[str, ...]:
        """Return sorted ticker symbols."""
        return tuple(
            sorted(
                self.frame["ticker"]
                .astype(str)
                .unique()
                .tolist()
            )
        )

    @property
    def input_columns(self) -> tuple[str, ...]:
        """
        Return all columns permitted to participate in model input.

        Target and future-derived metadata are explicitly excluded.
        """

        return (
            *self.feature_groups.market,
            *self.feature_groups.sentiment,
            *self.feature_groups.vision,
        )


def _validate_required_columns(
    frame: pd.DataFrame,
) -> None:
    """Validate the required fusion dataset schema."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise FusionDatasetError(
            "Fusion dataset is missing required columns: "
            + ", ".join(missing)
        )


def _validate_forbidden_columns(
    frame: pd.DataFrame,
) -> None:
    """
    Validate that forbidden target-derived columns cannot enter
    the model feature groups.
    """

    input_candidates = (
        *MARKET_FEATURE_COLUMNS,
        *SENTIMENT_FEATURE_COLUMNS,
        *VISION_METADATA_COLUMNS,
    )

    leakage_columns = [
        column
        for column in input_candidates
        if column in FORBIDDEN_INPUT_COLUMNS
    ]

    if leakage_columns:
        raise FusionDatasetError(
            "Target-derived columns detected in model inputs: "
            + ", ".join(leakage_columns)
        )


def _validate_timestamps(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize timestamps to UTC-naive daily timestamps."""

    timestamps = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        invalid_count = int(timestamps.isna().sum())

        raise FusionDatasetError(
            f"Found {invalid_count} invalid timestamp values."
        )

    normalized = (
        timestamps
        .dt.normalize()
        .dt.tz_localize(None)
    )

    result = frame.copy()
    result["timestamp"] = normalized

    return result


def _validate_tickers(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize and validate ticker symbols."""

    result = frame.copy()

    result["ticker"] = (
        result["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (result["ticker"] == "").any():
        raise FusionDatasetError(
            "Ticker column contains empty values."
        )

    return result


def _validate_unique_keys(
    frame: pd.DataFrame,
) -> None:
    """Reject duplicate ticker/date fusion keys."""

    duplicate_mask = frame.duplicated(
        subset=["ticker", "timestamp"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            frame.loc[
                duplicate_mask,
                ["ticker", "timestamp"],
            ]
            .sort_values(
                ["ticker", "timestamp"]
            )
            .head(10)
        )

        raise FusionDatasetError(
            "Duplicate ticker/timestamp keys detected. "
            f"Examples:\n{duplicates.to_string(index=False)}"
        )


def _validate_numeric_columns(
    frame: pd.DataFrame,
) -> None:
    """Validate numeric feature and target columns."""

    numeric_columns = (
        *MARKET_FEATURE_COLUMNS,
        *SENTIMENT_FEATURE_COLUMNS,
        "target_return_1d",
        "target_direction_1d",
    )

    for column in numeric_columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.isna().any():
            invalid_count = int(values.isna().sum())

            raise FusionDatasetError(
                f"Column '{column}' contains "
                f"{invalid_count} non-numeric/NaN values."
            )

        if not np.isfinite(values.to_numpy()).all():
            raise FusionDatasetError(
                f"Column '{column}' contains non-finite values."
            )


def _validate_sentiment_features(
    frame: pd.DataFrame,
) -> None:
    """Validate sentiment ratios, probabilities, and confidence."""

    bounded_columns = (
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "probability_negative_mean",
        "probability_neutral_mean",
        "probability_positive_mean",
        "confidence_mean",
    )

    for column in bounded_columns:
        values = frame[column].to_numpy(
            dtype=np.float64
        )

        if np.any(values < 0.0) or np.any(values > 1.0):
            raise FusionDatasetError(
                f"Sentiment column '{column}' contains "
                "values outside [0, 1]."
            )

    ratio_sum = (
        frame["positive_ratio"]
        + frame["neutral_ratio"]
        + frame["negative_ratio"]
    )

    if not np.allclose(
        ratio_sum.to_numpy(dtype=np.float64),
        1.0,
        atol=1e-5,
    ):
        raise FusionDatasetError(
            "Sentiment class ratios must sum to approximately 1.0."
        )


def _validate_vision_metadata(
    frame: pd.DataFrame,
) -> None:
    """Validate vision metadata without allowing target leakage."""

    if (
        frame["sample_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise FusionDatasetError(
            "Vision sample_id contains empty values."
        )

    if (
        frame["image_path"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise FusionDatasetError(
            "Vision image_path contains empty values."
        )

    window_sizes = pd.to_numeric(
        frame["window_size"],
        errors="coerce",
    )

    if window_sizes.isna().any():
        raise FusionDatasetError(
            "Vision window_size contains invalid values."
        )

    if (window_sizes <= 0).any():
        raise FusionDatasetError(
            "Vision window_size must be greater than zero."
        )


def _validate_target(
    frame: pd.DataFrame,
) -> None:
    """Validate the supervised target."""

    target = pd.to_numeric(
        frame["target_direction_1d"],
        errors="coerce",
    )

    if target.isna().any():
        raise FusionDatasetError(
            "target_direction_1d contains invalid values."
        )

    target_values = set(
        target.astype(int).unique().tolist()
    )

    if not target_values.issubset(
        BINARY_TARGET_VALUES
    ):
        raise FusionDatasetError(
            "target_direction_1d must contain only "
            "binary values {0, 1}."
        )

    if not np.allclose(
        target.to_numpy(dtype=np.float64),
        target.astype(int).to_numpy(dtype=np.float64),
    ):
        raise FusionDatasetError(
            "target_direction_1d must contain integer-valued "
            "binary labels."
        )


def _validate_feature_groups(
    feature_groups: FusionFeatureGroups,
) -> None:
    """Ensure feature groups contain no forbidden columns."""

    all_inputs = (
        *feature_groups.market,
        *feature_groups.sentiment,
        *feature_groups.vision,
    )

    duplicates = {
        column
        for column in all_inputs
        if all_inputs.count(column) > 1
    }

    if duplicates:
        raise FusionDatasetError(
            "Feature appears in multiple modality groups: "
            + ", ".join(sorted(duplicates))
        )

    leakage = (
        set(all_inputs)
        & FORBIDDEN_INPUT_COLUMNS
    )

    if leakage:
        raise FusionDatasetError(
            "Forbidden target-derived feature(s) detected: "
            + ", ".join(sorted(leakage))
        )


def validate_fusion_dataset(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize the complete aligned fusion dataset.

    This function deliberately preserves target-derived metadata in the
    dataframe for auditing, but those columns are never returned through
    the feature groups.
    """

    if frame.empty:
        raise FusionDatasetError(
            "Fusion dataset is empty."
        )

    _validate_required_columns(frame)
    _validate_forbidden_columns(frame)

    result = frame.copy()

    result = _validate_timestamps(result)
    result = _validate_tickers(result)

    _validate_unique_keys(result)
    _validate_numeric_columns(result)
    _validate_sentiment_features(result)
    _validate_vision_metadata(result)
    _validate_target(result)

    result = result.sort_values(
        ["timestamp", "ticker", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)

    return result


def build_feature_groups() -> FusionFeatureGroups:
    """Build the canonical Sentinel-AI fusion feature groups."""

    groups = FusionFeatureGroups(
        market=MARKET_FEATURE_COLUMNS,
        sentiment=SENTIMENT_FEATURE_COLUMNS,
        vision=VISION_METADATA_COLUMNS,
    )

    _validate_feature_groups(groups)

    return groups


def load_fusion_dataset(
    input_file: Path,
) -> FusionDataset:
    """Load and validate the aligned multimodal dataset."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Fusion dataset not found: {input_file}"
        )

    frame = pd.read_csv(
        input_file
    )

    validated = validate_fusion_dataset(
        frame
    )

    feature_groups = build_feature_groups()

    return FusionDataset(
        frame=validated,
        feature_groups=feature_groups,
    )


def chronological_split(
    dataset: FusionDataset,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> FusionSplit:
    """
    Split the fusion dataset chronologically.

    No random shuffling is performed. This prevents future observations
    from entering training through random splitting.
    """

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    if any(
        ratio <= 0.0
        for ratio in ratios
    ):
        raise ValueError(
            "All split ratios must be greater than zero."
        )

    if not np.isclose(
        sum(ratios),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio "
            "must equal 1.0."
        )

    frame = dataset.frame.sort_values(
        ["timestamp", "ticker", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)

    unique_dates = (
        frame["timestamp"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    if len(unique_dates) < 3:
        raise FusionDatasetError(
            "At least three unique dates are required "
            "for chronological splitting."
        )

    n_dates = len(unique_dates)

    train_end = max(
        1,
        int(
            np.floor(
                n_dates * train_ratio
            )
        ),
    )

    validation_end = max(
        train_end + 1,
        int(
            np.floor(
                n_dates
                * (train_ratio + validation_ratio)
            )
        ),
    )

    validation_end = min(
        validation_end,
        n_dates - 1,
    )

    train_dates = set(
        unique_dates[:train_end]
    )

    validation_dates = set(
        unique_dates[
            train_end:validation_end
        ]
    )

    test_dates = set(
        unique_dates[
            validation_end:
        ]
    )

    train = frame[
        frame["timestamp"].isin(
            train_dates
        )
    ].copy()

    validation = frame[
        frame["timestamp"].isin(
            validation_dates
        )
    ].copy()

    test = frame[
        frame["timestamp"].isin(
            test_dates
        )
    ].copy()

    if train.empty:
        raise FusionDatasetError(
            "Chronological train split is empty."
        )

    if validation.empty:
        raise FusionDatasetError(
            "Chronological validation split is empty."
        )

    if test.empty:
        raise FusionDatasetError(
            "Chronological test split is empty."
        )

    train_max = train["timestamp"].max()
    validation_min = validation["timestamp"].min()
    validation_max = validation["timestamp"].max()
    test_min = test["timestamp"].min()

    if not train_max < validation_min:
        raise FusionDatasetError(
            "Temporal leakage detected between train and validation."
        )

    if not validation_max < test_min:
        raise FusionDatasetError(
            "Temporal leakage detected between validation and test."
        )

    return FusionSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def extract_numeric_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """
    Extract numeric model features from a dataframe.

    This function refuses forbidden target-derived columns.
    """

    leakage = (
        set(feature_columns)
        & FORBIDDEN_INPUT_COLUMNS
    )

    if leakage:
        raise FusionDatasetError(
            "Cannot extract forbidden feature columns: "
            + ", ".join(sorted(leakage))
        )

    missing = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing:
        raise FusionDatasetError(
            "Missing feature columns: "
            + ", ".join(missing)
        )

    values = frame.loc[
        :,
        list(feature_columns),
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():
        raise FusionDatasetError(
            "Extracted numeric features contain non-finite values."
        )

    return values


def extract_market_features(
    dataset: FusionDataset,
) -> np.ndarray:
    """Extract only the Phase 1 market feature representation."""

    return extract_numeric_features(
        dataset.frame,
        dataset.feature_groups.market,
    )


def extract_sentiment_features(
    dataset: FusionDataset,
) -> np.ndarray:
    """Extract only the aggregated NLP sentiment representation."""

    return extract_numeric_features(
        dataset.frame,
        dataset.feature_groups.sentiment,
    )


def extract_vision_paths(
    dataset: FusionDataset,
) -> tuple[str, ...]:
    """
    Extract vision image paths.

    Only image_path is a model input. future_return and vision_label,
    when present in the dataframe, are intentionally ignored.
    """

    paths = (
        dataset.frame["image_path"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if any(
        not path
        for path in paths
    ):
        raise FusionDatasetError(
            "Vision image paths cannot be empty."
        )

    return tuple(paths)


def extract_targets(
    dataset: FusionDataset,
) -> np.ndarray:
    """Extract the supervised binary direction target."""

    return dataset.frame[
        "target_direction_1d"
    ].to_numpy(
        dtype=np.int64
    )


def get_split_summary(
    split: FusionSplit,
) -> dict[str, object]:
    """Return a compact chronological split summary."""

    def summarize(
        frame: pd.DataFrame,
    ) -> dict[str, object]:
        return {
            "rows": int(len(frame)),
            "tickers": sorted(
                frame["ticker"]
                .astype(str)
                .unique()
                .tolist()
            ),
            "start": (
                frame["timestamp"]
                .min()
                .isoformat()
            ),
            "end": (
                frame["timestamp"]
                .max()
                .isoformat()
            ),
            "positive_ratio": float(
                frame["target_direction_1d"]
                .mean()
            ),
        }

    return {
        "train": summarize(split.train),
        "validation": summarize(split.validation),
        "test": summarize(split.test),
    }


def run_leakage_audit(
    dataset: FusionDataset,
) -> dict[str, object]:
    """
    Run an explicit leakage audit.

    The audit reports forbidden columns separately from the permitted
    feature groups so that future changes to the dataset schema cannot
    silently introduce future information.
    """

    input_columns = set(
        dataset.input_columns
    )

    forbidden_present = sorted(
        input_columns
        & FORBIDDEN_INPUT_COLUMNS
    )

    target_columns_present = sorted(
        set(TARGET_COLUMNS)
        & input_columns
    )

    future_metadata_present = sorted(
        {
            "future_return",
            "vision_future_return",
            "vision_label",
        }
        & set(dataset.frame.columns)
    )

    return {
        "passed": (
            not forbidden_present
            and not target_columns_present
        ),
        "input_columns": sorted(
            input_columns
        ),
        "forbidden_input_columns": forbidden_present,
        "target_columns_in_inputs": target_columns_present,
        "future_metadata_present": future_metadata_present,
        "target_columns_retained_for_supervision": [
            column
            for column in TARGET_COLUMNS
            if column in dataset.frame.columns
        ],
    }


def main() -> None:
    """Validate and audit the Phase 2 multimodal fusion dataset."""

    input_file = (
        PROJECT_ROOT
        / "ml"
        / "datasets"
        / "multimodal"
        / "aligned"
        / "multimodal_dataset.csv"
    )

    dataset = load_fusion_dataset(
        input_file
    )

    split = chronological_split(
        dataset
    )

    market_features = extract_market_features(
        dataset
    )

    sentiment_features = extract_sentiment_features(
        dataset
    )

    vision_paths = extract_vision_paths(
        dataset
    )

    targets = extract_targets(
        dataset
    )

    leakage_audit = run_leakage_audit(
        dataset
    )

    if not leakage_audit["passed"]:
        raise FusionDatasetError(
            "Fusion leakage audit failed: "
            f"{leakage_audit}"
        )

    summary = get_split_summary(
        split
    )

    print(
        "=" * 72
    )
    print(
        "FUSION DATASET VALIDATION"
    )
    print(
        "=" * 72
    )

    print(
        f"Rows:              {dataset.size}"
    )

    print(
        f"Tickers:            {list(dataset.tickers)}"
    )

    print(
        f"Market features:    {len(dataset.feature_groups.market)}"
    )

    print(
        f"Sentiment features: {len(dataset.feature_groups.sentiment)}"
    )

    print(
        f"Vision inputs:      {len(dataset.feature_groups.vision)}"
    )

    print(
        f"Market matrix:      {market_features.shape}"
    )

    print(
        f"Sentiment matrix:   {sentiment_features.shape}"
    )

    print(
        f"Vision paths:       {len(vision_paths)}"
    )

    print(
        f"Targets:            {targets.shape}"
    )

    print()
    print(
        "=" * 72
    )
    print(
        "CHRONOLOGICAL SPLIT"
    )
    print(
        "=" * 72
    )

    for split_name, values in summary.items():
        print(
            f"{split_name.capitalize():12s}: "
            f"{values['rows']} rows | "
            f"{values['start']} → {values['end']} | "
            f"positive={values['positive_ratio']:.4f}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "LEAKAGE AUDIT"
    )
    print(
        "=" * 72
    )

    print(
        f"Passed: {leakage_audit['passed']}"
    )

    print(
        "Forbidden columns in inputs: "
        f"{leakage_audit['forbidden_input_columns']}"
    )

    print(
        "Targets in inputs: "
        f"{leakage_audit['target_columns_in_inputs']}"
    )

    print(
        "Future metadata retained only for audit: "
        f"{leakage_audit['future_metadata_present']}"
    )

    print()
    print(
        "Fusion dataset is ready for modality-specific embedding extraction."
    )


if __name__ == "__main__":
    main()