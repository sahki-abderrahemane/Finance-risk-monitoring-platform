from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "target_direction_1d"
TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Chronological train/validation/test split configuration."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("All split ratios must be between 0 and 1.")

        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1."
            )


@dataclass(frozen=True)
class GradientBoostingConfig:
    """Configuration for the Phase 1 gradient boosting classifier."""

    n_components: int | float | None = 0.95
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    subsample: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.n_components, int):
            if self.n_components <= 0:
                raise ValueError("n_components must be greater than zero.")

        if isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError(
                    "Float n_components must be in (0, 1]."
                )

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")

        if self.min_samples_split <= 1:
            raise ValueError(
                "min_samples_split must be greater than one."
            )

        if self.min_samples_leaf <= 0:
            raise ValueError(
                "min_samples_leaf must be greater than zero."
            )

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1].")


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological dataset split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ModelMetrics:
    """Classification metrics for one dataset partition."""

    accuracy: float
    roc_auc: float
    log_loss: float


@dataclass(frozen=True)
class GradientBoostingReport:
    """Summary produced after model training."""

    ticker: str
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    pca_components: int
    train_metrics: ModelMetrics
    validation_metrics: ModelMetrics
    test_metrics: ModelMetrics


class GradientBoostingRiskModel:
    """
    Leakage-safe Phase 1 Gradient Boosting model.

    The preprocessing pipeline follows strict temporal isolation:

    1. Split data chronologically.
    2. Fit StandardScaler on training data only.
    3. Fit PCA on training data only.
    4. Transform validation and test data using fitted preprocessing.
    5. Train GradientBoostingClassifier on training data.

    The model predicts next-day direction from the engineered market
    features. This is a research/simulation model and must never be
    interpreted as financial advice or used for live trading.
    """

    def __init__(
        self,
        config: GradientBoostingConfig | None = None,
        split_config: DatasetSplitConfig | None = None,
    ) -> None:
        self.config = config or GradientBoostingConfig()
        self.split_config = split_config or DatasetSplitConfig()

        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._model: GradientBoostingClassifier | None = None
        self._fitted: bool = False

    @property
    def scaler(self) -> StandardScaler:
        """Return the fitted feature scaler."""
        if self._scaler is None:
            raise RuntimeError("Model has not been fitted.")

        return self._scaler

    @property
    def pca(self) -> PCA:
        """Return the PCA transformer fitted on training data."""
        if self._pca is None:
            raise RuntimeError("Model has not been fitted.")

        return self._pca

    @property
    def model(self) -> GradientBoostingClassifier:
        """Return the fitted Gradient Boosting model."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")

        return self._model

    @property
    def is_fitted(self) -> bool:
        """Whether the complete model pipeline has been fitted."""
        return self._fitted

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> DatasetSplit:
        """
        Split a time-series dataset chronologically.

        No shuffling is performed.
        """
        self._validate_dataframe(dataframe)

        dataframe = dataframe.sort_values(
            TIMESTAMP_COLUMN
        ).reset_index(drop=True)

        total_rows = len(dataframe)

        train_end = int(
            total_rows * self.split_config.train_ratio
        )

        validation_end = train_end + int(
            total_rows * self.split_config.validation_ratio
        )

        if train_end <= 0:
            raise ValueError("Training split contains no rows.")

        if validation_end <= train_end:
            raise ValueError("Validation split contains no rows.")

        if validation_end >= total_rows:
            raise ValueError("Test split contains no rows.")

        return DatasetSplit(
            train=dataframe.iloc[:train_end].copy(),
            validation=dataframe.iloc[
                train_end:validation_end
            ].copy(),
            test=dataframe.iloc[validation_end:].copy(),
        )

    def fit(
        self,
        train_dataframe: pd.DataFrame,
    ) -> None:
        """
        Fit preprocessing and Gradient Boosting on training data only.
        """
        self._validate_dataframe(train_dataframe)

        X_train = train_dataframe.loc[:, FEATURE_COLUMNS]
        y_train = train_dataframe.loc[:, TARGET_COLUMN].astype(int)

        self._scaler = StandardScaler()

        X_train_scaled = self._scaler.fit_transform(X_train)

        self._pca = PCA(
            n_components=self.config.n_components
        )

        X_train_pca = self._pca.fit_transform(
            X_train_scaled
        )

        self._model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

        self._model.fit(
            X_train_pca,
            y_train,
        )

        self._fitted = True

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Transform feature data using train-fitted preprocessing."""
        self._validate_dataframe(dataframe)

        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before transform."
            )

        X = dataframe.loc[:, FEATURE_COLUMNS]

        X_scaled = self.scaler.transform(X)

        return self.pca.transform(X_scaled)

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Predict the next-day direction class."""
        X_pca = self.transform(dataframe)

        return self.model.predict(X_pca)

    def predict_proba(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Return class probabilities."""
        X_pca = self.transform(dataframe)

        return self.model.predict_proba(X_pca)

    def evaluate(
        self,
        dataframe: pd.DataFrame,
    ) -> ModelMetrics:
        """Evaluate the model on a dataset partition."""
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before evaluation."
            )

        y_true = dataframe[TARGET_COLUMN].astype(int).to_numpy()

        probabilities = self.predict_proba(dataframe)

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        positive_probability = probabilities[:, 1]

        return ModelMetrics(
            accuracy=float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            roc_auc=float(
                roc_auc_score(
                    y_true,
                    positive_probability,
                )
            ),
            log_loss=float(
                log_loss(
                    y_true,
                    probabilities,
                )
            ),
        )

    def fit_split(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[DatasetSplit, GradientBoostingReport]:
        """
        Perform chronological splitting, training, and evaluation.

        The returned report contains metrics for all three partitions.
        """
        split = self.split(dataframe)

        self.fit(split.train)

        train_metrics = self.evaluate(split.train)
        validation_metrics = self.evaluate(split.validation)
        test_metrics = self.evaluate(split.test)

        ticker = self._infer_ticker(dataframe)

        report = GradientBoostingReport(
            ticker=ticker,
            total_rows=len(dataframe),
            train_rows=len(split.train),
            validation_rows=len(split.validation),
            test_rows=len(split.test),
            pca_components=self.pca.n_components_,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

        return split, report

    def component_importance(self) -> pd.DataFrame:
        """
        Return Gradient Boosting importance for PCA components.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before component_importance."
            )

        return pd.DataFrame(
            {
                "component": [
                    f"pc_{index + 1}"
                    for index in range(
                        len(self.model.feature_importances_)
                    )
                ],
                "importance": self.model.feature_importances_,
            }
        )

    def explained_variance(self) -> pd.DataFrame:
        """Return train-fitted PCA explained variance."""
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before explained_variance."
            )

        ratios = self.pca.explained_variance_ratio_

        return pd.DataFrame(
            {
                "component": [
                    f"pc_{index + 1}"
                    for index in range(len(ratios))
                ],
                "explained_variance_ratio": ratios,
                "cumulative_explained_variance": np.cumsum(
                    ratios
                ),
            }
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Persist scaler, PCA, model, and configuration."""
        if not self._fitted:
            raise RuntimeError(
                "Cannot save an unfitted model."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "config": self.config,
                "split_config": self.split_config,
                "scaler": self.scaler,
                "pca": self.pca,
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "GradientBoostingRiskModel":
        """Load a previously saved model."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact does not exist: {path}"
            )

        artifact = joblib.load(path)

        instance = cls(
            config=artifact["config"],
            split_config=artifact["split_config"],
        )

        instance._scaler = artifact["scaler"]
        instance._pca = artifact["pca"]
        instance._model = artifact["model"]
        instance._fitted = True

        return instance

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        required_columns = (
            TIMESTAMP_COLUMN,
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        if dataframe.empty:
            raise ValueError("Dataset cannot be empty.")

        if dataframe[FEATURE_COLUMNS].isnull().any().any():
            raise ValueError(
                "Feature columns cannot contain null values."
            )

        if dataframe[TARGET_COLUMN].isnull().any():
            raise ValueError(
                "Target column cannot contain null values."
            )

    @staticmethod
    def _infer_ticker(
        dataframe: pd.DataFrame,
    ) -> str:
        """
        Infer ticker from an optional ticker column.

        Falls back to 'unknown' because the current feature datasets
        identify the asset through their filenames.
        """
        if "ticker" not in dataframe.columns:
            return "unknown"

        values = dataframe["ticker"].dropna().astype(str).unique()

        if len(values) == 0:
            return "unknown"

        if len(values) > 1:
            raise ValueError(
                "A model dataset must contain exactly one ticker."
            )

        return values[0]