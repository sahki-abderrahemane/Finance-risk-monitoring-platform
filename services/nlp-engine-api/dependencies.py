"""NLP model loading and lifecycle management."""

from __future__ import annotations

from pathlib import Path

from nlp_engine.sentiment import FinancialSentimentModel, SentimentConfig
from nlp_engine.preprocessing import FinancialNewsPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "finbert"

sentiment_model: FinancialSentimentModel | None = None
preprocessor: FinancialNewsPreprocessor | None = None
model_loaded: bool = False


def load_models() -> None:
    global sentiment_model, preprocessor, model_loaded

    if MODEL_DIR.exists():
        sentiment_model = FinancialSentimentModel.load(
            model_dir=MODEL_DIR,
            device="cpu",
        )
        preprocessor = FinancialNewsPreprocessor()
        model_loaded = True


def get_sentiment_model() -> FinancialSentimentModel:
    if sentiment_model is None:
        raise RuntimeError("NLP sentiment model not loaded")
    return sentiment_model


def get_preprocessor() -> FinancialNewsPreprocessor:
    if preprocessor is None:
        raise RuntimeError("Preprocessor not initialized")
    return preprocessor
