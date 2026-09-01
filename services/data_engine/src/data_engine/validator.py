from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_engine.loader import REQUIRED_COLUMNS


@dataclass(frozen=True)
class ValidationIssue:
    """A single data-quality issue."""

    rule: str
    message: str
    count: int


@dataclass
class ValidationReport:
    """Complete validation result for one dataset."""

    valid: bool
    rows: int
    issues: list[ValidationIssue] = field(
        default_factory=list
    )

    @property
    def issue_count(self) -> int:
        """Return the total number of validation issues."""
        return len(self.issues)


class MarketDataValidator:
    """Validate canonical market-data DataFrames."""

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []

        self._check_required_columns(
            dataframe,
            issues,
        )

        if not REQUIRED_COLUMNS.issubset(
            dataframe.columns
        ):
            return ValidationReport(
                valid=False,
                rows=len(dataframe),
                issues=issues,
            )

        self._check_missing_values(
            dataframe,
            issues,
        )

        self._check_timestamps(
            dataframe,
            issues,
        )

        self._check_duplicates(
            dataframe,
            issues,
        )

        self._check_chronological_order(
            dataframe,
            issues,
        )

        self._check_ohlc(
            dataframe,
            issues,
        )

        self._check_prices(
            dataframe,
            issues,
        )

        self._check_volume(
            dataframe,
            issues,
        )

        return ValidationReport(
            valid=len(issues) == 0,
            rows=len(dataframe),
            issues=issues,
        )

    @staticmethod
    def _check_required_columns(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        missing = REQUIRED_COLUMNS - set(
            dataframe.columns
        )

        if missing:
            issues.append(
                ValidationIssue(
                    rule="required_columns",
                    message=(
                        "Missing required columns: "
                        f"{sorted(missing)}"
                    ),
                    count=len(missing),
                )
            )

    @staticmethod
    def _check_missing_values(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        missing = dataframe.isna().sum()

        for column, count in missing.items():
            if count > 0:
                issues.append(
                    ValidationIssue(
                        rule="missing_values",
                        message=(
                            f"Column '{column}' "
                            f"contains {count} missing values"
                        ),
                        count=int(count),
                    )
                )

    @staticmethod
    def _check_timestamps(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        timestamps = dataframe["timestamp"]

        invalid = timestamps.isna()

        count = int(invalid.sum())

        if count > 0:
            issues.append(
                ValidationIssue(
                    rule="invalid_timestamps",
                    message=(
                        f"Found {count} invalid timestamps"
                    ),
                    count=count,
                )
            )

    @staticmethod
    def _check_duplicates(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        count = int(
            dataframe["timestamp"]
            .duplicated()
            .sum()
        )

        if count > 0:
            issues.append(
                ValidationIssue(
                    rule="duplicate_timestamps",
                    message=(
                        f"Found {count} duplicate timestamps"
                    ),
                    count=count,
                )
            )

    @staticmethod
    def _check_chronological_order(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        if not dataframe[
            "timestamp"
        ].is_monotonic_increasing:
            issues.append(
                ValidationIssue(
                    rule="chronological_order",
                    message=(
                        "Timestamps are not in "
                        "chronological order"
                    ),
                    count=1,
                )
            )

    @staticmethod
    def _check_ohlc(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        invalid = (
            (dataframe["high"] < dataframe["low"])
            | (dataframe["high"] < dataframe["open"])
            | (dataframe["high"] < dataframe["close"])
            | (dataframe["low"] > dataframe["open"])
            | (dataframe["low"] > dataframe["close"])
        )

        count = int(invalid.sum())

        if count > 0:
            issues.append(
                ValidationIssue(
                    rule="ohlc_consistency",
                    message=(
                        f"Found {count} rows with "
                        "invalid OHLC relationships"
                    ),
                    count=count,
                )
            )

    @staticmethod
    def _check_prices(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        price_columns = [
            "open",
            "high",
            "low",
            "close",
        ]

        negative = (
            dataframe[price_columns] < 0
        ).any(axis=1)

        negative_count = int(
            negative.sum()
        )

        if negative_count > 0:
            issues.append(
                ValidationIssue(
                    rule="negative_prices",
                    message=(
                        f"Found {negative_count} rows "
                        "with negative prices"
                    ),
                    count=negative_count,
                )
            )

        zero = (
            dataframe[price_columns] == 0
        ).any(axis=1)

        zero_count = int(
            zero.sum()
        )

        if zero_count > 0:
            issues.append(
                ValidationIssue(
                    rule="zero_prices",
                    message=(
                        f"Found {zero_count} rows "
                        "with zero prices"
                    ),
                    count=zero_count,
                )
            )

    @staticmethod
    def _check_volume(
        dataframe: pd.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        negative = (
            dataframe["volume"] < 0
        )

        negative_count = int(
            negative.sum()
        )

        if negative_count > 0:
            issues.append(
                ValidationIssue(
                    rule="negative_volume",
                    message=(
                        f"Found {negative_count} rows "
                        "with negative volume"
                    ),
                    count=negative_count,
                )
            )

        zero = (
            dataframe["volume"] == 0
        )

        zero_count = int(
            zero.sum()
        )

        if zero_count > 0:
            issues.append(
                ValidationIssue(
                    rule="zero_volume",
                    message=(
                        f"Found {zero_count} rows "
                        "with zero volume"
                    ),
                    count=zero_count,
                )
            )