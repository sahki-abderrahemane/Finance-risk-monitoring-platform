from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engine.features import MarketFeatureEngineer
from data_engine.pca import (
    FEATURE_COLUMNS,
    PCAConfig,
    MarketPCA,
)


def make_market_dataframe(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
        tz="UTC",
    )

    returns = rng.normal(
        loc=0.0005,
        scale=0.02,
        size=rows,
    )

    close = 100.0 * np.cumprod(
        1.0 + returns
    )

    open_prices = close * (
        1.0 + rng.normal(
            0.0,
            0.005,
            rows,
        )
    )

    high = np.maximum(
        open_prices,
        close,
    ) * (
        1.0 + rng.uniform(
            0.0,
            0.01,
            rows,
        )
    )

    low = np.minimum(
        open_prices,
        close,
    ) * (
        1.0 - rng.uniform(
            0.0,
            0.01,
            rows,
        )
    )

    volume = rng.integers(
        1_000_000,
        5_000_000,
        size=rows,
    )

    raw = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    engineer = MarketFeatureEngineer()

    result, _ = engineer.transform(raw)

    return result.reset_index(drop=True)


def test_pca_uses_all_expected_features() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=0.95
        )
    )

    report = pca.fit(dataframe)

    assert report.input_features == len(
        FEATURE_COLUMNS
    )


def test_pca_reduces_or_preserves_dimensions() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=0.95
        )
    )

    transformed, report = pca.fit_transform(
        dataframe
    )

    assert (
        1
        <= report.output_components
        <= len(FEATURE_COLUMNS)
    )

    component_columns = [
        column
        for column in transformed.columns
        if column.startswith("pc_")
    ]

    assert len(component_columns) == (
        report.output_components
    )


def test_explained_variance_is_monotonic() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=None
        )
    )

    pca.fit(dataframe)

    variance = pca.explained_variance()

    cumulative = (
        variance[
            "cumulative_explained_variance"
        ]
        .to_numpy()
    )

    assert np.all(
        np.diff(cumulative) >= -1e-12
    )

    assert cumulative[-1] == pytest.approx(
        1.0
    )


def test_explained_variance_ratios_sum_to_one() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=None
        )
    )

    pca.fit(dataframe)

    ratios = (
        pca.explained_variance()[
            "explained_variance_ratio"
        ]
        .to_numpy()
    )

    assert ratios.sum() == pytest.approx(
        1.0
    )


def test_loadings_have_expected_shape() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=5
        )
    )

    pca.fit(dataframe)

    loadings = pca.component_loadings()

    assert loadings.shape == (
        len(FEATURE_COLUMNS),
        5,
    )

    assert list(loadings.index) == list(
        FEATURE_COLUMNS
    )


def test_targets_are_preserved_but_not_used_as_pca_features() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=5
        )
    )

    transformed, _ = pca.fit_transform(
        dataframe
    )

    assert "target_return_1d" in transformed.columns
    assert "target_direction_1d" in transformed.columns

    assert (
        "target_return_1d"
        not in FEATURE_COLUMNS
    )

    assert (
        "target_direction_1d"
        not in FEATURE_COLUMNS
    )


def test_timestamp_is_preserved() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=5
        )
    )

    transformed, _ = pca.fit_transform(
        dataframe
    )

    pd.testing.assert_series_equal(
        transformed["timestamp"],
        dataframe["timestamp"],
        check_names=False,
    )


def test_transformed_components_are_finite() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=5
        )
    )

    transformed, _ = pca.fit_transform(
        dataframe
    )

    components = transformed.filter(
        regex=r"^pc_\d+$"
    )

    assert np.isfinite(
        components.to_numpy()
    ).all()


def test_transform_requires_fit() -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA()

    with pytest.raises(RuntimeError):
        pca.transform(dataframe)


def test_loadings_requires_fit() -> None:
    pca = MarketPCA()

    with pytest.raises(RuntimeError):
        pca.component_loadings()


def test_explained_variance_requires_fit() -> None:
    pca = MarketPCA()

    with pytest.raises(RuntimeError):
        pca.explained_variance()


def test_save_and_load_preserve_transform(
    tmp_path: Path,
) -> None:
    dataframe = make_market_dataframe()

    pca = MarketPCA(
        PCAConfig(
            n_components=5
        )
    )

    original, _ = pca.fit_transform(
        dataframe
    )

    artifact = tmp_path / "pca.joblib"

    pca.save(artifact)

    loaded = MarketPCA.load(artifact)

    restored = loaded.transform(
        dataframe
    )

    pd.testing.assert_frame_equal(
        original,
        restored,
    )


def test_invalid_component_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        PCAConfig(
            n_components=0
        )


def test_invalid_variance_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        PCAConfig(
            n_components=1.5
        )


def test_missing_feature_is_rejected() -> None:
    dataframe = make_market_dataframe()

    dataframe = dataframe.drop(
        columns=["sma_20"]
    )

    pca = MarketPCA()

    with pytest.raises(ValueError):
        pca.fit(dataframe)