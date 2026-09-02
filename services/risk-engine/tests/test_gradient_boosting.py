from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_engine.gradient_boosting import (
    DatasetSplitConfig,
    GradientBoostingConfig,
    GradientBoostingRiskModel,
)


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


def make_dataset(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
        }
    )

    for index, column in enumerate(FEATURE_COLUMNS):
        dataframe[column] = rng.normal(
            loc=float(index),
            scale=1.0,
            size=rows,
        )

    signal = (
        dataframe["return_1d"]
        + 0.5 * dataframe["volatility_20"]
    )

    dataframe["target_direction_1d"] = (
        signal > signal.median()
    ).astype(int)

    return dataframe


def test_chronological_split() -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel()

    split = model.split(dataframe)

    assert len(split.train) == 210
    assert len(split.validation) == 45
    assert len(split.test) == 45

    assert (
        split.train["timestamp"].max()
        < split.validation["timestamp"].min()
    )

    assert (
        split.validation["timestamp"].max()
        < split.test["timestamp"].min()
    )


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        DatasetSplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.2,
        )


def test_model_fits_and_predicts() -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel(
        config=GradientBoostingConfig(
            n_components=0.95,
            n_estimators=50,
            random_state=42,
        )
    )

    split, report = model.fit_split(dataframe)

    assert model.is_fitted
    assert model.pca.n_components_ <= len(FEATURE_COLUMNS)

    predictions = model.predict(split.test)

    assert len(predictions) == len(split.test)
    assert set(predictions).issubset({0, 1})

    probabilities = model.predict_proba(split.test)

    assert probabilities.shape == (
        len(split.test),
        2,
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )

    assert report.total_rows == len(dataframe)


def test_train_fitted_pca_has_expected_components() -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel(
        config=GradientBoostingConfig(
            n_components=0.95,
            n_estimators=25,
        )
    )

    split = model.split(dataframe)

    model.fit(split.train)

    assert model.pca.n_components_ >= 1
    assert model.pca.n_components_ <= len(FEATURE_COLUMNS)

    explained = model.explained_variance()

    assert not explained.empty

    cumulative = explained[
        "cumulative_explained_variance"
    ]

    assert cumulative.iloc[-1] >= 0.95


def test_component_importance_matches_pca_dimension() -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel(
        config=GradientBoostingConfig(
            n_components=0.95,
            n_estimators=25,
        )
    )

    model.fit(dataframe.iloc[:210])

    importance = model.component_importance()

    assert len(importance) == model.pca.n_components_
    assert np.isclose(
        importance["importance"].sum(),
        1.0,
    )


def test_save_and_load(tmp_path: Path) -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel(
        config=GradientBoostingConfig(
            n_components=0.95,
            n_estimators=25,
        )
    )

    split = model.split(dataframe)

    model.fit(split.train)

    path = tmp_path / "model.joblib"

    model.save(path)

    loaded = GradientBoostingRiskModel.load(path)

    original = model.predict_proba(split.test)
    restored = loaded.predict_proba(split.test)

    assert np.allclose(
        original,
        restored,
    )


def test_transform_requires_fit() -> None:
    dataframe = make_dataset()

    model = GradientBoostingRiskModel()

    with pytest.raises(RuntimeError):
        model.transform(dataframe)


def test_missing_columns_are_rejected() -> None:
    dataframe = make_dataset().drop(
        columns=["volatility_20"]
    )

    model = GradientBoostingRiskModel()

    with pytest.raises(ValueError):
        model.split(dataframe)


def test_empty_dataframe_is_rejected() -> None:
    dataframe = make_dataset().iloc[0:0]

    model = GradientBoostingRiskModel()

    with pytest.raises(ValueError):
        model.split(dataframe)