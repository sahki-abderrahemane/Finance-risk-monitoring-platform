from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from multimodal_engine.unified_risk_model import (
    UnifiedRiskModel,
    UnifiedRiskModelConfig,
    load_checkpoint,
    probability_to_risk_score,
    risk_band,
    save_checkpoint,
)


def test_config_rejects_invalid_input_dimension() -> None:
    with pytest.raises(ValueError):
        UnifiedRiskModelConfig(input_dim=0)


def test_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        UnifiedRiskModelConfig(dropout=1.0)


def test_forward_output_shape() -> None:
    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(input_dim=901)
    )

    features = torch.randn(8, 901)

    logits = model(features)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_forward_rejects_wrong_feature_dimension() -> None:
    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(input_dim=901)
    )

    features = torch.randn(8, 900)

    with pytest.raises(ValueError):
        model(features)


def test_probability_is_between_zero_and_one() -> None:
    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(input_dim=901)
    )

    features = torch.randn(16, 901)

    probabilities = model.predict_probability(features)

    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_risk_score_is_between_zero_and_fifty() -> None:
    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(input_dim=901)
    )

    features = torch.randn(16, 901)

    risk_scores = model.predict_risk_score(features)

    assert torch.all(risk_scores >= 0.0)
    assert torch.all(risk_scores <= 50.0)


def test_probability_to_risk_score() -> None:
    probabilities = np.array(
        [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],
        dtype=np.float32,
    )

    risk_scores = probability_to_risk_score(
        probabilities
    )

    expected = np.array(
        [
            0.0,
            10.0,
            25.0,
            50.0,
            25.0,
            10.0,
            0.0,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        risk_scores,
        expected,
        atol=1e-6,
    )


def test_probability_to_risk_score_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        probability_to_risk_score(
            np.array([0.5, 1.2])
        )


def test_risk_bands() -> None:
    assert risk_band(0.0) == "LOW"
    assert risk_band(24.99) == "LOW"
    assert risk_band(25.0) == "MODERATE"
    assert risk_band(49.99) == "MODERATE"
    assert risk_band(50.0) == "HIGH"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = UnifiedRiskModelConfig(
        input_dim=901,
    )

    model = UnifiedRiskModel(config)

    checkpoint_path = (
        tmp_path / "unified_risk_model.pt"
    )

    save_checkpoint(
        model,
        checkpoint_path,
        epoch=7,
        best_validation_loss=0.691,
        feature_manifest="fusion_manifest.json",
    )

    loaded_model, checkpoint = load_checkpoint(
        checkpoint_path
    )

    assert loaded_model.config == config
    assert checkpoint["epoch"] == 7
    assert checkpoint["target_column"] == (
        "target_direction_1d"
    )
    assert checkpoint["feature_dim"] == 901

    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()

    for name in original_state:
        assert torch.equal(
            original_state[name],
            loaded_state[name],
        )


def test_checkpoint_contains_uncertainty_definition(
    tmp_path: Path,
) -> None:
    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(input_dim=901)
    )

    checkpoint_path = (
        tmp_path / "unified_risk_model.pt"
    )

    save_checkpoint(
        model,
        checkpoint_path,
    )

    _, checkpoint = load_checkpoint(
        checkpoint_path
    )

    definition = checkpoint["risk_definition"]

    assert definition["probability"] == (
        "sigmoid(logit)"
    )

    assert definition["uncertainty"] == (
        "1 - max(p, 1 - p)"
    )

    assert definition["risk_score"] == (
        "100 * uncertainty"
    )

    assert definition["interpretation"] == (
        "model uncertainty only"
    )