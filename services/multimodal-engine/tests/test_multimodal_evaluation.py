from __future__ import annotations

import numpy as np
import pytest

from multimodal_engine.multimodal_evaluation import (
    EXPECTED_FULL_DIM,
    MARKET_DIM,
    TEXT_DIM,
    VISION_DIM,
    calculate_metrics,
    expected_calibration_error,
    feature_indices,
    probability_to_risk_score,
    select_features,
    validate_feature_layout,
)


def test_feature_dimensions_sum_to_full_representation() -> None:
    assert (
        MARKET_DIM
        + TEXT_DIM
        + VISION_DIM
        == EXPECTED_FULL_DIM
    )


def test_feature_layout_accepts_901_dimensions() -> None:
    features = np.zeros(
        (4, EXPECTED_FULL_DIM),
        dtype=np.float32,
    )

    validate_feature_layout(features)


def test_feature_layout_rejects_wrong_dimension() -> None:
    features = np.zeros(
        (4, EXPECTED_FULL_DIM - 1),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_feature_layout(features)


def test_market_indices() -> None:
    indices = feature_indices(
        ("market",)
    )

    assert len(indices) == MARKET_DIM
    assert indices.tolist() == list(
        range(MARKET_DIM)
    )


def test_text_indices() -> None:
    indices = feature_indices(
        ("text",)
    )

    expected = list(
        range(
            MARKET_DIM,
            MARKET_DIM + TEXT_DIM,
        )
    )

    assert indices.tolist() == expected


def test_vision_indices() -> None:
    indices = feature_indices(
        ("vision",)
    )

    start = MARKET_DIM + TEXT_DIM

    expected = list(
        range(
            start,
            start + VISION_DIM,
        )
    )

    assert indices.tolist() == expected


def test_full_indices_cover_all_features() -> None:
    indices = feature_indices(
        ("market", "text", "vision")
    )

    assert len(indices) == EXPECTED_FULL_DIM

    assert np.array_equal(
        indices,
        np.arange(
            EXPECTED_FULL_DIM
        ),
    )


def test_select_market_features() -> None:
    features = np.arange(
        2 * EXPECTED_FULL_DIM,
        dtype=np.float32,
    ).reshape(
        2,
        EXPECTED_FULL_DIM,
    )

    selected = select_features(
        features,
        ("market",),
    )

    assert selected.shape == (
        2,
        MARKET_DIM,
    )


def test_select_market_text_features() -> None:
    features = np.zeros(
        (3, EXPECTED_FULL_DIM),
        dtype=np.float32,
    )

    selected = select_features(
        features,
        ("market", "text"),
    )

    assert selected.shape == (
        3,
        MARKET_DIM + TEXT_DIM,
    )


def test_select_rejects_empty_groups() -> None:
    features = np.zeros(
        (2, EXPECTED_FULL_DIM),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        select_features(
            features,
            (),
        )


def test_ece_is_zero_for_perfectly_calibrated_simple_case() -> None:
    y_true = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.01, 0.01, 0.99, 0.99],
        dtype=np.float64,
    )

    ece = expected_calibration_error(
        y_true,
        probabilities,
    )

    assert ece == pytest.approx(
        0.01,
        abs=1e-6,
    )


def test_ece_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.5]),
        )


def test_calculate_metrics_returns_required_metrics() -> None:
    y_true = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9],
        dtype=np.float64,
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )

    required = {
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "macro_f1",
        "precision",
        "recall",
        "log_loss",
        "brier_score",
        "ece",
        "risk_score_mean",
        "risk_score_median",
    }

    assert required.issubset(
        metrics.keys()
    )


def test_calculate_metrics_ranges() -> None:
    y_true = np.array(
        [0, 1, 0, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.1, 0.9, 0.2, 0.8],
        dtype=np.float64,
    )

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert metrics["brier_score"] >= 0.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["risk_score_mean"] <= 50.0
    assert 0.0 <= metrics["risk_score_median"] <= 50.0


def test_calculate_metrics_rejects_single_class_target() -> None:
    y_true = np.array(
        [1, 1, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.7, 0.8, 0.9],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        calculate_metrics(
            y_true,
            probabilities,
        )