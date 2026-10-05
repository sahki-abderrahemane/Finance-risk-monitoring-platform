from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor, nn


DEFAULT_INPUT_DIM: Final[int] = 901
DEFAULT_HIDDEN_DIM_1: Final[int] = 256
DEFAULT_HIDDEN_DIM_2: Final[int] = 64
DEFAULT_DROPOUT: Final[float] = 0.25

RISK_BAND_LOW: Final[str] = "LOW"
RISK_BAND_MODERATE: Final[str] = "MODERATE"
RISK_BAND_HIGH: Final[str] = "HIGH"
RISK_BAND_EXTREME: Final[str] = "EXTREME"


@dataclass(frozen=True)
class UnifiedRiskModelConfig:
    """Configuration for the multimodal unified risk head."""

    input_dim: int = DEFAULT_INPUT_DIM
    hidden_dim_1: int = DEFAULT_HIDDEN_DIM_1
    hidden_dim_2: int = DEFAULT_HIDDEN_DIM_2
    dropout: float = DEFAULT_DROPOUT

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if self.hidden_dim_1 <= 0:
            raise ValueError("hidden_dim_1 must be greater than zero.")

        if self.hidden_dim_2 <= 0:
            raise ValueError("hidden_dim_2 must be greater than zero.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable configuration dictionary."""
        return asdict(self)


class UnifiedRiskModel(nn.Module):
    """
    Feed-forward risk head operating on the fused multimodal representation.

    Input:
        [market representation || FinBERT representation || CNN representation]

    Output:
        One logit representing the probability of target_direction_1d == 1.

    The model does not directly produce a financial recommendation.
    """

    def __init__(
        self,
        config: UnifiedRiskModelConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or UnifiedRiskModelConfig()

        self.network = nn.Sequential(
            nn.Linear(
                self.config.input_dim,
                self.config.hidden_dim_1,
            ),
            nn.LayerNorm(self.config.hidden_dim_1),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.hidden_dim_1,
                self.config.hidden_dim_2,
            ),
            nn.LayerNorm(self.config.hidden_dim_2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim_2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Compute binary classification logits.

        Args:
            features: Tensor shaped [batch_size, input_dim].

        Returns:
            Tensor shaped [batch_size].
        """

        if features.ndim != 2:
            raise ValueError(
                "features must be a 2D tensor shaped "
                "[batch_size, input_dim]."
            )

        if features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} input features, "
                f"received {features.shape[1]}."
            )

        logits = self.network(features)

        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_probability(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Convert model logits into probability of target_direction_1d == 1.
        """

        logits = self.forward(features)

        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_risk_score(
        self,
        features: Tensor,
    ) -> Tensor:
        """
        Calculate model-uncertainty risk score.

        U = 1 - max(p, 1 - p)

        risk_score = 100 * U

        A score near 0 means the classifier is confident.
        A score near 50 means maximum binary uncertainty.

        This is a model-uncertainty metric, not financial advice.
        """

        probabilities = self.predict_probability(features)

        uncertainty = 1.0 - torch.maximum(
            probabilities,
            1.0 - probabilities,
        )

        return uncertainty * 100.0


def risk_band(risk_score: float) -> str:
    """
    Convert model uncertainty into the Phase 1 risk-band convention.
    """

    if not np.isfinite(risk_score):
        raise ValueError("risk_score must be finite.")

    if not 0.0 <= risk_score <= 50.0:
        raise ValueError(
            "Binary model-uncertainty risk scores must be "
            "between 0 and 50."
        )

    if risk_score < 25.0:
        return RISK_BAND_LOW

    if risk_score < 50.0:
        return RISK_BAND_MODERATE

    return RISK_BAND_HIGH


def probability_to_risk_score(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Convert probabilities into binary model-uncertainty scores.

    U = 1 - max(p, 1-p)
    R = 100 * U
    """

    probabilities = np.asarray(probabilities, dtype=np.float32)

    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "probabilities must lie in the [0, 1] interval."
        )

    uncertainty = 1.0 - np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    return uncertainty * 100.0


def save_checkpoint(
    model: UnifiedRiskModel,
    path: str | Path,
    *,
    optimizer_state_dict: dict[str, Any] | None = None,
    epoch: int | None = None,
    best_validation_loss: float | None = None,
    training_history: list[dict[str, float]] | None = None,
    feature_manifest: str | None = None,
    target_column: str = "target_direction_1d",
) -> None:
    """Save a complete unified-risk-model checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "feature_dim": model.config.input_dim,
        "target_column": target_column,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "training_history": training_history or [],
        "feature_manifest": feature_manifest,
        "risk_definition": {
            "probability": "sigmoid(logit)",
            "uncertainty": "1 - max(p, 1 - p)",
            "risk_score": "100 * uncertainty",
            "interpretation": "model uncertainty only",
        },
    }

    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[UnifiedRiskModel, dict[str, Any]]:
    """Load a unified risk model and its metadata."""

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Unified risk model checkpoint does not exist: {source}"
        )

    checkpoint = torch.load(
        source,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid unified risk model checkpoint.")

    model_config = checkpoint.get("model_config")

    if not isinstance(model_config, dict):
        raise ValueError(
            "Checkpoint is missing model_config."
        )

    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(**model_config)
    )

    state_dict = checkpoint.get("model_state_dict")

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint is missing model_state_dict."
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint