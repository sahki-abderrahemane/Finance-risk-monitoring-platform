"""Model loading and lifecycle management."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from risk_engine.gradient_boosting import GradientBoostingRiskModel
from risk_engine.risk_scoring import ResearchRiskScorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

AVAILABLE_TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

gbm_models: dict[str, GradientBoostingRiskModel] = {}
risk_scorer: ResearchRiskScorer | None = None
models_loaded: bool = False


def load_models() -> None:
    global risk_scorer, models_loaded

    for ticker in AVAILABLE_TICKERS:
        path = MODELS_DIR / "gradient_boosting" / f"{ticker}_gradient_boosting.joblib"
        if path.exists():
            gbm_models[ticker] = GradientBoostingRiskModel.load(path)

    scorer_path = MODELS_DIR / "risk_scoring" / "research_risk_scorer.joblib"
    if scorer_path.exists():
        risk_scorer = ResearchRiskScorer.load(scorer_path)

    models_loaded = bool(gbm_models) and risk_scorer is not None


def get_gbm_model(ticker: str) -> GradientBoostingRiskModel:
    model = gbm_models.get(ticker.upper())
    if model is None:
        available = list(gbm_models.keys())
        raise ValueError(f"No model for ticker '{ticker}'. Available: {available}")
    return model


def get_risk_scorer() -> ResearchRiskScorer:
    if risk_scorer is None:
        raise RuntimeError("Risk scorer not loaded")
    return risk_scorer


def compute_features_from_ohlcv(data: list[dict[str, float]], ticker: str) -> dict[str, float]:
    """Compute the 11 ML features from raw OHLCV data."""
    df = pd.DataFrame(data)

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values("timestamp") if "timestamp" in df.columns else df

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    features = {
        "return_1d": float(close.pct_change().iloc[-1]) if len(close) > 1 else 0.0,
        "log_return_1d": float((close / close.shift(1)).apply(lambda x: pd.np.log(x) if x > 0 else 0).iloc[-1]) if len(close) > 1 else 0.0,
        "sma_5": float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else float(close.mean()),
        "sma_20": float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean()),
        "ema_5": float(close.ewm(span=5).mean().iloc[-1]),
        "ema_20": float(close.ewm(span=20).mean().iloc[-1]),
        "volatility_20": float(close.pct_change().rolling(20).std().iloc[-1]) if len(close) >= 21 else 0.0,
        "volume_sma_20": float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean()),
        "volume_ratio": float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 and volume.rolling(20).mean().iloc[-1] > 0 else 1.0,
        "high_low_spread": float((high - low).iloc[-1]),
        "open_close_spread": float(close.iloc[-1] - open_.iloc[-1]),
    }

    return features
