from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    roc_auc: float
    log_loss: float
    brier_score: float


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    positive_class_ratio: float


@dataclass(frozen=True)
class EvaluationReport:
    ticker: str
    rows: int
    gbm: ClassificationMetrics
    baseline: BaselineMetrics
    accuracy_lift: float
    risk_score_mean: float
    risk_score_median: float
    high_risk_positive_ratio: float | None
    low_risk_positive_ratio: float | None


class RiskModelEvaluator:
    """
    Evaluate the trained GBM against a simple majority-class baseline.

    This evaluator is intended for historical research and simulation.
    It does not produce trading recommendations.
    """

    REQUIRED_COLUMNS = {
        "timestamp",
        "target_direction_1d",
        "predicted_direction",
        "probability_down",
        "probability_up",
        "risk_score",
    }

    def validate(self, dataframe: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"Missing required evaluation columns: {sorted(missing)}"
            )

        if dataframe.empty:
            raise ValueError("Cannot evaluate an empty dataframe.")

        if dataframe[list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(
                "Evaluation data contains null values in required columns."
            )

        for column in (
            "target_direction_1d",
            "predicted_direction",
        ):
            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"{column} contains non-numeric values."
                )

            if not values.isin([0, 1]).all():
                raise ValueError(
                    f"{column} must contain only 0/1 values."
                )

        for column in (
            "probability_down",
            "probability_up",
        ):
            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    f"{column} contains non-numeric values."
                )

            if ((values < 0) | (values > 1)).any():
                raise ValueError(
                    f"{column} must contain probabilities between 0 and 1."
                )

    @staticmethod
    def calculate_gbm_metrics(
        dataframe: pd.DataFrame,
    ) -> ClassificationMetrics:
        y_true = dataframe["target_direction_1d"].astype(int)
        y_pred = dataframe["predicted_direction"].astype(int)
        probability_up = dataframe["probability_up"].astype(float)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        if y_true.nunique() < 2:
            raise ValueError(
                "ROC-AUC requires both target classes."
            )

        return ClassificationMetrics(
            accuracy=float(accuracy),
            roc_auc=float(
                roc_auc_score(
                    y_true,
                    probability_up,
                )
            ),
            log_loss=float(
                log_loss(
                    y_true,
                    np.column_stack(
                        [
                            1.0 - probability_up,
                            probability_up,
                        ]
                    ),
                    labels=[0, 1],
                )
            ),
            brier_score=float(
                brier_score_loss(
                    y_true,
                    probability_up,
                )
            ),
        )

    @staticmethod
    def calculate_baseline_metrics(
        dataframe: pd.DataFrame,
    ) -> BaselineMetrics:
        y_true = dataframe["target_direction_1d"].astype(int)

        positive_ratio = float(y_true.mean())

        majority_class = int(
            positive_ratio >= 0.5
        )

        baseline_prediction = np.full(
            len(y_true),
            majority_class,
            dtype=int,
        )

        return BaselineMetrics(
            accuracy=float(
                accuracy_score(
                    y_true,
                    baseline_prediction,
                )
            ),
            positive_class_ratio=positive_ratio,
        )

    @staticmethod
    def calculate_risk_statistics(
        dataframe: pd.DataFrame,
    ) -> tuple[float, float, float | None, float | None]:
        risk_score = dataframe["risk_score"].astype(float)

        mean_score = float(risk_score.mean())
        median_score = float(risk_score.median())

        high_risk = dataframe[
            dataframe["risk_score"] >= 50.0
        ]

        low_risk = dataframe[
            dataframe["risk_score"] < 25.0
        ]

        high_ratio: float | None

        if high_risk.empty:
            high_ratio = None
        else:
            high_ratio = float(
                high_risk["target_direction_1d"].mean()
            )

        low_ratio: float | None

        if low_risk.empty:
            low_ratio = None
        else:
            low_ratio = float(
                low_risk["target_direction_1d"].mean()
            )

        return (
            mean_score,
            median_score,
            high_ratio,
            low_ratio,
        )

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        ticker: str,
    ) -> EvaluationReport:
        self.validate(dataframe)

        gbm_metrics = self.calculate_gbm_metrics(
            dataframe
        )

        baseline_metrics = self.calculate_baseline_metrics(
            dataframe
        )

        (
            mean_score,
            median_score,
            high_ratio,
            low_ratio,
        ) = self.calculate_risk_statistics(dataframe)

        return EvaluationReport(
            ticker=ticker,
            rows=len(dataframe),
            gbm=gbm_metrics,
            baseline=baseline_metrics,
            accuracy_lift=(
                gbm_metrics.accuracy
                - baseline_metrics.accuracy
            ),
            risk_score_mean=mean_score,
            risk_score_median=median_score,
            high_risk_positive_ratio=high_ratio,
            low_risk_positive_ratio=low_ratio,
        )

    @staticmethod
    def report_to_dataframe(
        report: EvaluationReport,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": report.ticker,
                    "rows": report.rows,
                    "gbm_accuracy": report.gbm.accuracy,
                    "gbm_roc_auc": report.gbm.roc_auc,
                    "gbm_log_loss": report.gbm.log_loss,
                    "gbm_brier_score": report.gbm.brier_score,
                    "baseline_accuracy": report.baseline.accuracy,
                    "positive_class_ratio": (
                        report.baseline.positive_class_ratio
                    ),
                    "accuracy_lift": report.accuracy_lift,
                    "risk_score_mean": report.risk_score_mean,
                    "risk_score_median": report.risk_score_median,
                    "high_risk_positive_ratio": (
                        report.high_risk_positive_ratio
                    ),
                    "low_risk_positive_ratio": (
                        report.low_risk_positive_ratio
                    ),
                }
            ]
        )

    @staticmethod
    def save_report(
        report: EvaluationReport,
        path: str | Path,
    ) -> None:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        RiskModelEvaluator.report_to_dataframe(
            report
        ).to_csv(
            output_path,
            index=False,
        )

    @staticmethod
    def save_evaluator(
        evaluator: "RiskModelEvaluator",
        path: str | Path,
    ) -> None:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            evaluator,
            output_path,
        )

    @staticmethod
    def load_evaluator(
        path: str | Path,
    ) -> "RiskModelEvaluator":
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Evaluator artifact does not exist: {input_path}"
            )

        evaluator = joblib.load(input_path)

        if not isinstance(
            evaluator,
            RiskModelEvaluator,
        ):
            raise TypeError(
                "Loaded artifact is not a RiskModelEvaluator."
            )

        return evaluator