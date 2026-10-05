from __future__ import annotations

import numpy as np
import pytest

from vision_engine.compare_cnn import (
    CNNComparisonError,
    calculate_majority_baseline,
    calculate_model_result,
)


def test_calculate_majority_baseline() -> None:
    targets = np.asarray(
        [2, 2, 2, 1, 0],
        dtype=np.int64,
    )

    result = calculate_majority_baseline(targets)

    assert result == pytest.approx(0.6)


def test_calculate_majority_baseline_rejects_empty_targets() -> None:
    targets = np.asarray([], dtype=np.int64)

    with pytest.raises(
        CNNComparisonError,
        match="empty targets",
    ):
        calculate_majority_baseline(targets)


def test_calculate_model_result() -> None:
    targets = np.asarray(
        [0, 1, 2, 0, 1, 2],
        dtype=np.int64,
    )

    predictions = np.asarray(
        [0, 1, 2, 0, 2, 2],
        dtype=np.int64,
    )

    result = calculate_model_result(
        model_name="test_cnn",
        checkpoint_path="/tmp/test.pt",
        targets=targets,
        predictions=predictions,
    )

    assert result.model_name == "test_cnn"
    assert result.test_samples == 6
    assert result.accuracy == pytest.approx(5.0 / 6.0)
    assert result.majority_baseline == pytest.approx(1.0 / 3.0)
    assert result.accuracy_lift == pytest.approx(0.5)


def test_calculate_model_result_rejects_empty_targets() -> None:
    targets = np.asarray([], dtype=np.int64)
    predictions = np.asarray([], dtype=np.int64)

    with pytest.raises(
        CNNComparisonError,
        match="test targets are empty",
    ):
        calculate_model_result(
            model_name="test_cnn",
            checkpoint_path="/tmp/test.pt",
            targets=targets,
            predictions=predictions,
        )


def test_calculate_model_result_rejects_length_mismatch() -> None:
    targets = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    predictions = np.asarray(
        [0, 1],
        dtype=np.int64,
    )

    with pytest.raises(
        CNNComparisonError,
        match="prediction/target length mismatch",
    ):
        calculate_model_result(
            model_name="test_cnn",
            checkpoint_path="/tmp/test.pt",
            targets=targets,
            predictions=predictions,
        )


def test_calculate_model_result_reports_all_classes() -> None:
    targets = np.asarray(
        [0, 0, 1, 1, 2, 2],
        dtype=np.int64,
    )

    predictions = np.asarray(
        [0, 1, 1, 1, 2, 0],
        dtype=np.int64,
    )

    result = calculate_model_result(
        model_name="test_cnn",
        checkpoint_path="/tmp/test.pt",
        targets=targets,
        predictions=predictions,
    )

    assert 0.0 <= result.down_precision <= 1.0
    assert 0.0 <= result.down_recall <= 1.0
    assert 0.0 <= result.down_f1 <= 1.0

    assert 0.0 <= result.flat_precision <= 1.0
    assert 0.0 <= result.flat_recall <= 1.0
    assert 0.0 <= result.flat_f1 <= 1.0

    assert 0.0 <= result.up_precision <= 1.0
    assert 0.0 <= result.up_recall <= 1.0
    assert 0.0 <= result.up_f1 <= 1.0