from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_engine.risk_scoring import (
    ResearchRiskScorer,
    RiskScoringConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

FEATURES_DIR = PROJECT_ROOT / "ml" / "datasets" / "features"
GBM_DIR = PROJECT_ROOT / "ml" / "datasets" / "gbm"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "datasets" / "risk_scores"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "risk_scoring"


def load_predictions(ticker: str) -> pd.DataFrame:
    """
    Load GBM test predictions and join them with the corresponding
    feature dataset to recover volatility_20.
    """
    prediction_path = (
        GBM_DIR / f"{ticker}_test_predictions.csv"
    )

    feature_path = (
        FEATURES_DIR / f"{ticker}.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {feature_path}"
        )

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["timestamp"],
    )

    features = pd.read_csv(
        feature_path,
        parse_dates=["timestamp"],
    )

    required_prediction_columns = {
        "timestamp",
        "probability_down",
        "probability_up",
    }

    missing_predictions = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            f"{ticker}: missing prediction columns: "
            f"{sorted(missing_predictions)}"
        )

    if "volatility_20" not in features.columns:
        raise ValueError(
            f"{ticker}: volatility_20 is missing from "
            "the feature dataset."
        )

    volatility = features[
        ["timestamp", "volatility_20"]
    ].copy()

    result = predictions.merge(
        volatility,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    if result["volatility_20"].isna().any():
        raise ValueError(
            f"{ticker}: failed to recover volatility_20 "
            "for one or more predictions."
        )

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return result


def save_summary(
    ticker: str,
    report: object,
) -> None:
    """
    Save aggregate risk-score statistics.
    """
    summary_path = (
        OUTPUT_DIR / f"{ticker}_summary.csv"
    )

    data = {
        "ticker": ticker,
        "rows": report.rows,
        "mean_score": report.mean_score,
        "median_score": report.median_score,
        "minimum_score": report.minimum_score,
        "maximum_score": report.maximum_score,
        "low_count": report.low_count,
        "moderate_count": report.moderate_count,
        "high_count": report.high_count,
        "extreme_count": report.extreme_count,
    }

    pd.DataFrame([data]).to_csv(
        summary_path,
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scorer = ResearchRiskScorer(
        RiskScoringConfig(
            model_uncertainty_weight=0.5,
            volatility_weight=0.5,
            volatility_lookback=252,
        )
    )

    feature_files = sorted(
        FEATURES_DIR.glob("*.csv")
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No feature datasets found in {FEATURES_DIR}"
        )

    for feature_path in feature_files:
        ticker = feature_path.stem

        print(f"\nProcessing {ticker}...")

        prediction_data = load_predictions(ticker)

        scored = scorer.score(prediction_data)

        output_path = (
            OUTPUT_DIR
            / f"{ticker}_risk_scores.csv"
        )

        scored.to_csv(
            output_path,
            index=False,
        )

        report = scorer.summarize(scored)

        save_summary(
            ticker=ticker,
            report=report,
        )

        print(
            f"  Rows: {report.rows}"
        )
        print(
            f"  Mean risk score: {report.mean_score:.2f}"
        )
        print(
            f"  Median risk score: {report.median_score:.2f}"
        )
        print(
            f"  Range: "
            f"{report.minimum_score:.2f} - "
            f"{report.maximum_score:.2f}"
        )
        print(
            "  Bands: "
            f"LOW={report.low_count}, "
            f"MODERATE={report.moderate_count}, "
            f"HIGH={report.high_count}, "
            f"EXTREME={report.extreme_count}"
        )

    scorer_path = (
        MODEL_DIR / "research_risk_scorer.joblib"
    )

    ResearchRiskScorer.save(
        scorer,
        scorer_path,
    )

    print(
        f"\nSaved scorer: {scorer_path}"
    )
    print(
        f"Saved risk datasets: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()