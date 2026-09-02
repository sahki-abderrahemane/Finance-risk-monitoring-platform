from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_engine.evaluation import RiskModelEvaluator


def make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=8,
                freq="D",
            ),
            "target_direction_1d": [
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
            ],
            "predicted_direction": [
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
            ],
            "probability_down": [
                0.80,
                0.20,
                0.60,
                0.75,
                0.30,
                0.35,
                0.15,
                0.70,
            ],
            "probability_up": [
                0.20,
                0.80,
                0.40,
                0.25,
                0.70,
                0.65,
                0.85,
                0.30,
            ],
            "risk_score": [
                20.0,
                30.0,
                55.0,
                60.0,
                70.0,
                15.0,
                80.0,
                45.0,
            ],
        }
    )


def test_validation_accepts_valid_dataframe() -> None:
    evaluator = RiskModelEvaluator()

    evaluator.validate(make_dataframe())


def test_validation_rejects_missing_column() -> None:
    evaluator = RiskModelEvaluator()

    dataframe = make_dataframe().drop(
        columns=["risk_score"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required",
    ):
        evaluator.validate(dataframe)


def test_validation_rejects_empty_dataframe() -> None:
    evaluator = RiskModelEvaluator()

    dataframe = pd.DataFrame(
        columns=[
            "timestamp",
            "target_direction_1d",
            "predicted_direction",
            "probability_down",
            "probability_up",
            "risk_score",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        evaluator.evaluate(
            dataframe,
            ticker="TEST",
        )


def test_gbm_metrics() -> None:
    evaluator = RiskModelEvaluator()

    metrics = evaluator.calculate_gbm_metrics(
        make_dataframe()
    )

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert metrics.log_loss >= 0.0
    assert 0.0 <= metrics.brier_score <= 1.0


def test_baseline_metrics() -> None:
    evaluator = RiskModelEvaluator()

    metrics = evaluator.calculate_baseline_metrics(
        make_dataframe()
    )

    assert metrics.positive_class_ratio == pytest.approx(
        5 / 8
    )

    assert metrics.accuracy == pytest.approx(
        5 / 8
    )


def test_risk_statistics() -> None:
    evaluator = RiskModelEvaluator()

    (
        mean_score,
        median_score,
        high_ratio,
        low_ratio,
    ) = evaluator.calculate_risk_statistics(
        make_dataframe()
    )

    assert mean_score == pytest.approx(
        46.875
    )

    assert median_score == pytest.approx(
        50.0
    )

    assert high_ratio is not None
    assert low_ratio is not None

    assert 0.0 <= high_ratio <= 1.0
    assert 0.0 <= low_ratio <= 1.0


def test_full_evaluation() -> None:
    evaluator = RiskModelEvaluator()

    report = evaluator.evaluate(
        make_dataframe(),
        ticker="TEST",
    )

    assert report.ticker == "TEST"
    assert report.rows == 8

    assert report.accuracy_lift == pytest.approx(
        report.gbm.accuracy
        - report.baseline.accuracy
    )


def test_report_dataframe() -> None:
    evaluator = RiskModelEvaluator()

    report = evaluator.evaluate(
        make_dataframe(),
        ticker="TEST",
    )

    dataframe = evaluator.report_to_dataframe(
        report
    )

    assert len(dataframe) == 1

    expected_columns = {
        "ticker",
        "rows",
        "gbm_accuracy",
        "gbm_roc_auc",
        "gbm_log_loss",
        "gbm_brier_score",
        "baseline_accuracy",
        "positive_class_ratio",
        "accuracy_lift",
        "risk_score_mean",
        "risk_score_median",
        "high_risk_positive_ratio",
        "low_risk_positive_ratio",
    }

    assert expected_columns.issubset(
        dataframe.columns
    )


def test_invalid_prediction_class() -> None:
    evaluator = RiskModelEvaluator()

    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "predicted_direction",
    ] = 2

    with pytest.raises(
        ValueError,
        match="only 0/1",
    ):
        evaluator.validate(dataframe)


def test_invalid_probability() -> None:
    evaluator = RiskModelEvaluator()

    dataframe = make_dataframe()

    dataframe.loc[
        0,
        "probability_up",
    ] = 1.5

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluator.validate(dataframe)