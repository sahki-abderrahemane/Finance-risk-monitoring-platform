from __future__ import annotations

import json

import pytest

from vision_engine.evaluation import (
    CLASS_NAMES,
    calculate_metrics,
)


def test_class_names_are_stable() -> None:
    assert CLASS_NAMES == [
        "DOWN",
        "FLAT",
        "UP",
    ]


def test_perfect_predictions() -> None:
    labels = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    predictions = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    assert metrics["test_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)


def test_majority_baseline_is_calculated() -> None:
    labels = [
        0,
        0,
        0,
        0,
        1,
        2,
    ]

    predictions = [
        0,
        0,
        0,
        1,
        1,
        2,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    assert metrics[
        "majority_baseline_accuracy"
    ] == pytest.approx(
        4 / 6
    )

    assert metrics[
        "accuracy"
    ] == pytest.approx(
        5 / 6
    )

    assert metrics[
        "accuracy_lift_over_majority"
    ] == pytest.approx(
        1 / 6
    )


def test_confusion_matrix_shape() -> None:
    labels = [
        0,
        0,
        1,
        1,
        2,
        2,
    ]

    predictions = [
        0,
        1,
        1,
        2,
        2,
        0,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    matrix = metrics[
        "confusion_matrix"
    ]

    assert matrix == [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]


def test_per_class_metrics_exist() -> None:
    labels = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    predictions = [
        0,
        1,
        2,
        1,
        1,
        0,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    per_class = metrics[
        "per_class"
    ]

    assert isinstance(
        per_class,
        dict,
    )

    for class_name in CLASS_NAMES:
        assert class_name in per_class

        class_metrics = per_class[
            class_name
        ]

        assert isinstance(
            class_metrics,
            dict,
        )

        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1" in class_metrics
        assert "support" in class_metrics


def test_class_distributions() -> None:
    labels = [
        0,
        0,
        1,
        2,
        2,
        2,
    ]

    predictions = [
        0,
        1,
        1,
        2,
        2,
        0,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    assert metrics[
        "actual_class_distribution"
    ] == {
        "DOWN": 2,
        "FLAT": 1,
        "UP": 3,
    }

    assert metrics[
        "predicted_class_distribution"
    ] == {
        "DOWN": 2,
        "FLAT": 2,
        "UP": 2,
    }


def test_empty_test_set_is_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match="empty test set",
    ):
        calculate_metrics(
            predictions=[],
            labels=[],
        )


def test_metrics_are_serializable() -> None:
    labels = [
        0,
        1,
        2,
    ]

    predictions = [
        0,
        1,
        2,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    serialized = json.dumps(
        metrics
    )

    assert serialized


def test_metrics_have_expected_top_level_fields() -> None:
    labels = [
        0,
        1,
        2,
        0,
    ]

    predictions = [
        0,
        1,
        1,
        0,
    ]

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    expected_fields = {
        "test_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "majority_baseline_accuracy",
        "accuracy_lift_over_majority",
        "actual_class_distribution",
        "predicted_class_distribution",
        "per_class",
        "classification_report",
        "confusion_matrix",
    }

    assert expected_fields.issubset(
        metrics.keys()
    )