from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
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

TARGET_COLUMNS = (
    "target_return_1d",
    "target_direction_1d",
)

IDENTIFIER_COLUMNS = (
    "timestamp",
)


@dataclass(frozen=True)
class PCAConfig:
    """Configuration for PCA."""

    n_components: int | float | None = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.n_components, int):
            if self.n_components <= 0:
                raise ValueError(
                    "n_components must be greater than zero."
                )

        if isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError(
                    "Float n_components must be in (0, 1]."
                )


@dataclass(frozen=True)
class PCAReport:
    """Summary of PCA fitting."""

    input_rows: int
    input_features: int
    output_components: int
    explained_variance_ratio: tuple[float, ...]
    cumulative_explained_variance: tuple[float, ...]


class MarketPCA:
    """
    Standardize market features and reduce dimensionality with PCA.

    The target columns are never passed to PCA.

    The scaler and PCA model are fitted only through ``fit`` and can
    subsequently be reused through ``transform`` without refitting.
    """

    def __init__(
        self,
        config: PCAConfig | None = None,
    ) -> None:
        self.config = config or PCAConfig()

        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._fitted: bool = False

    @property
    def scaler(self) -> StandardScaler:
        """Return the fitted scaler."""
        if self._scaler is None:
            raise RuntimeError(
                "PCA pipeline has not been fitted."
            )

        return self._scaler

    @property
    def pca(self) -> PCA:
        """Return the fitted PCA model."""
        if self._pca is None:
            raise RuntimeError(
                "PCA pipeline has not been fitted."
            )

        return self._pca

    @property
    def is_fitted(self) -> bool:
        """Whether the PCA pipeline has been fitted."""
        return self._fitted

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> PCAReport:
        """
        Fit standardization and PCA on feature columns.

        Targets are explicitly excluded.
        """
        self._validate_input(dataframe)

        X = dataframe.loc[:, FEATURE_COLUMNS]

        self._scaler = StandardScaler()

        X_scaled = self._scaler.fit_transform(X)

        self._pca = PCA(
            n_components=self.config.n_components
        )

        self._pca.fit(X_scaled)

        self._fitted = True

        return self._build_report(
            input_rows=len(dataframe),
            input_features=len(FEATURE_COLUMNS),
        )

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using an already-fitted scaler and PCA model.

        The original identifiers and targets are preserved.
        """
        self._validate_input(dataframe)

        if not self._fitted:
            raise RuntimeError(
                "PCA pipeline must be fitted before transform."
            )

        X = dataframe.loc[:, FEATURE_COLUMNS]

        X_scaled = self.scaler.transform(X)

        components = self.pca.transform(X_scaled)

        component_columns = [
            f"pc_{index + 1}"
            for index in range(components.shape[1])
        ]

        result = pd.DataFrame(
            components,
            columns=component_columns,
            index=dataframe.index,
        )

        for column in IDENTIFIER_COLUMNS:
            if column in dataframe.columns:
                result[column] = dataframe[column]

        for column in TARGET_COLUMNS:
            result[column] = dataframe[column].values

        ordered_columns = [
            *IDENTIFIER_COLUMNS,
            *component_columns,
            *TARGET_COLUMNS,
        ]

        ordered_columns = [
            column
            for column in ordered_columns
            if column in result.columns
        ]

        return result.loc[:, ordered_columns]

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, PCAReport]:
        """Fit PCA and transform the same dataset."""
        report = self.fit(dataframe)

        transformed = self.transform(dataframe)

        return transformed, report

    def component_loadings(self) -> pd.DataFrame:
        """
        Return feature loadings for each principal component.

        Rows are original features and columns are PCs.
        """
        if not self._fitted:
            raise RuntimeError(
                "PCA pipeline must be fitted before "
                "component_loadings."
            )

        return pd.DataFrame(
            self.pca.components_.T,
            index=FEATURE_COLUMNS,
            columns=[
                f"pc_{index + 1}"
                for index in range(
                    self.pca.n_components_
                )
            ],
        )

    def explained_variance(self) -> pd.DataFrame:
        """Return explained and cumulative variance."""
        if not self._fitted:
            raise RuntimeError(
                "PCA pipeline must be fitted before "
                "explained_variance."
            )

        ratios = self.pca.explained_variance_ratio_

        cumulative = np.cumsum(ratios)

        return pd.DataFrame(
            {
                "component": [
                    f"pc_{index + 1}"
                    for index in range(len(ratios))
                ],
                "explained_variance_ratio": ratios,
                "cumulative_explained_variance": cumulative,
            }
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Persist the fitted scaler and PCA model."""
        if not self._fitted:
            raise RuntimeError(
                "Cannot save an unfitted PCA pipeline."
            )

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "config": self.config,
                "scaler": self.scaler,
                "pca": self.pca,
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "MarketPCA":
        """Load a previously fitted PCA pipeline."""
        source = Path(path)

        if not source.exists():
            raise FileNotFoundError(
                f"PCA model does not exist: {source}"
            )

        payload = joblib.load(source)

        required_keys = {
            "config",
            "scaler",
            "pca",
        }

        if not required_keys.issubset(payload):
            raise ValueError(
                "Invalid PCA model artifact."
            )

        instance = cls(
            config=payload["config"]
        )

        instance._scaler = payload["scaler"]
        instance._pca = payload["pca"]
        instance._fitted = True

        return instance

    def _build_report(
        self,
        input_rows: int,
        input_features: int,
    ) -> PCAReport:
        return PCAReport(
            input_rows=input_rows,
            input_features=input_features,
            output_components=self.pca.n_components_,
            explained_variance_ratio=tuple(
                float(value)
                for value
                in self.pca.explained_variance_ratio_
            ),
            cumulative_explained_variance=tuple(
                float(value)
                for value
                in np.cumsum(
                    self.pca.explained_variance_ratio_
                )
            ),
        )

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame,
    ) -> None:
        if dataframe.empty:
            raise ValueError(
                "Cannot run PCA on an empty dataframe."
            )

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Missing PCA feature columns: "
                + ", ".join(missing)
            )

        for column in FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise TypeError(
                    f"PCA feature '{column}' must be numeric."
                )