"""Multimodal model loading and lifecycle management."""

from __future__ import annotations

import numpy as np
import torch
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "multimodal" / "unified_risk_model_best.pt"
PCA_PIPELINE_PATH = PROJECT_ROOT / "ml" / "datasets" / "multimodal" / "features" / "market_pca_pipeline.joblib"

multimodal_model = None
pca_pipeline = None
model_loaded = False
device = torch.device("cpu")


def load_models() -> None:
    global multimodal_model, pca_pipeline, model_loaded

    if not CHECKPOINT_PATH.exists():
        return

    from multimodal_engine.unified_risk_model import UnifiedRiskModel, load_checkpoint

    multimodal_model, _ = load_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    multimodal_model.eval()

    if PCA_PIPELINE_PATH.exists():
        pca_pipeline = joblib.load(PCA_PIPELINE_PATH)

    model_loaded = True


def fuse_features(
    market: np.ndarray,
    text_embedding: np.ndarray | None = None,
    vision_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse market, text, and vision features into the expected 901-dim vector.

    If text/vision embeddings are not provided, zero vectors are used.
    """
    if pca_pipeline is not None:
        market_pca = pca_pipeline.transform(market.reshape(1, -1))
    else:
        market_pca = market.reshape(1, -1)

    market_dim = market_pca.shape[1]
    text_dim = 768
    vision_dim = 128

    if text_embedding is not None:
        text_part = text_embedding.reshape(1, -1)
    else:
        text_part = np.zeros((1, text_dim), dtype=np.float32)

    if vision_embedding is not None:
        vision_part = vision_embedding.reshape(1, -1)
    else:
        vision_part = np.zeros((1, vision_dim), dtype=np.float32)

    fused = np.concatenate([market_pca, text_part, vision_part], axis=1).astype(np.float32)

    expected_dim = 901
    if fused.shape[1] < expected_dim:
        padding = np.zeros((1, expected_dim - fused.shape[1]), dtype=np.float32)
        fused = np.concatenate([fused, padding], axis=1)
    elif fused.shape[1] > expected_dim:
        fused = fused[:, :expected_dim]

    return fused


def predict_from_fused(fused_features: np.ndarray) -> tuple[int, str, float, float]:
    """Run the unified risk model on fused features.

    Returns (risk_score, risk_label, probability, uncertainty).
    """
    if multimodal_model is None:
        raise RuntimeError("Multimodal model not loaded")

    tensor = torch.tensor(fused_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        probability = float(multimodal_model.predict_probability(tensor).cpu().item())
        uncertainty = 1.0 - max(probability, 1.0 - probability)
        risk_score = int(round(100 * uncertainty))

    if risk_score < 25:
        risk_label = "LOW"
    elif risk_score < 50:
        risk_label = "MODERATE"
    else:
        risk_label = "HIGH"

    return risk_score, risk_label, probability, round(uncertainty, 4)
