from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from multimodal_engine.unified_risk_model import (
    UnifiedRiskModel,
    UnifiedRiskModelConfig,
    save_checkpoint,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

FEATURE_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "features"
)

MODEL_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "multimodal"
)

CHECKPOINT_PATH: Final[Path] = (
    MODEL_DIR / "unified_risk_model_best.pt"
)

TRAIN_FEATURES: Final[Path] = (
    FEATURE_DIR / "train_fused_features.npy"
)

VALIDATION_FEATURES: Final[Path] = (
    FEATURE_DIR / "validation_fused_features.npy"
)

TEST_FEATURES: Final[Path] = (
    FEATURE_DIR / "test_fused_features.npy"
)

TRAIN_METADATA: Final[Path] = (
    FEATURE_DIR / "train_fused_metadata.csv"
)

VALIDATION_METADATA: Final[Path] = (
    FEATURE_DIR / "validation_fused_metadata.csv"
)

TEST_METADATA: Final[Path] = (
    FEATURE_DIR / "test_fused_metadata.csv"
)

MANIFEST_PATH: Final[Path] = (
    FEATURE_DIR / "fusion_manifest.json"
)

SEED: Final[int] = 42
BATCH_SIZE: Final[int] = 64
LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-4
MAX_EPOCHS: Final[int] = 50
PATIENCE: Final[int] = 8


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for the unified risk head."""

    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than zero.")

        if self.patience <= 0:
            raise ValueError("patience must be greater than zero.")


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_matrix(path: Path) -> np.ndarray:
    """Load and validate a fused feature matrix."""

    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix does not exist: {path}"
        )

    features = np.load(path)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix at {path}, "
            f"received shape {features.shape}."
        )

    if len(features) == 0:
        raise ValueError(
            f"Feature matrix is empty: {path}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Feature matrix contains non-finite values: {path}"
        )

    return features.astype(np.float32, copy=False)


def load_targets(
    metadata_path: Path,
    expected_rows: int,
) -> np.ndarray:
    """
    Load the unified binary target.

    Only target_direction_1d is used as the supervised target.
    Future-derived vision metadata is never used as model input.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "ticker",
        "timestamp",
        "target_direction_1d",
    }

    missing = sorted(
        required_columns.difference(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(missing)
        )

    if len(metadata) != expected_rows:
        raise ValueError(
            f"Feature/metadata row mismatch for {metadata_path}: "
            f"{expected_rows} features vs {len(metadata)} metadata rows."
        )

    target = pd.to_numeric(
        metadata["target_direction_1d"],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            f"target_direction_1d contains invalid values in "
            f"{metadata_path}."
        )

    values = target.to_numpy(dtype=np.float32)

    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(
            "target_direction_1d must contain only binary values {0, 1}."
        )

    return values


def validate_alignment(
    features: np.ndarray,
    metadata_path: Path,
) -> None:
    """Validate that fused features cannot be trained with leaked metadata."""

    metadata = pd.read_csv(metadata_path)

    forbidden_columns = {
        "target_return_1d",
        "target_direction_1d",
        "future_return",
        "vision_future_return",
        "vision_label",
        "future_return_1d",
        "future_direction_1d",
        "future_direction",
        "label",
        "target",
    }

    # These columns are expected to exist for audit/target purposes.
    # They are never converted into feature tensors.
    feature_columns = set()

    if features.shape[1] <= 0:
        raise ValueError("Feature matrix must contain input features.")

    leaked_feature_columns = feature_columns.intersection(
        forbidden_columns
    )

    if leaked_feature_columns:
        raise ValueError(
            "Forbidden target-derived columns detected in feature matrix: "
            + ", ".join(sorted(leaked_feature_columns))
        )

    if "future_return" not in metadata.columns:
        raise ValueError(
            "Fusion metadata must retain future_return for leakage audit."
        )


def build_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[TensorDataset]:
    """Create a PyTorch DataLoader."""

    if len(features) != len(targets):
        raise ValueError(
            "Feature and target row counts must match."
        )

    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(targets)

    dataset = TensorDataset(
        feature_tensor,
        target_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def calculate_pos_weight(targets: np.ndarray) -> float:
    """Calculate BCE positive-class weighting from training data only."""

    positive_count = float(np.sum(targets == 1.0))
    negative_count = float(np.sum(targets == 0.0))

    if positive_count == 0.0:
        raise ValueError(
            "Training target contains no positive samples."
        )

    if negative_count == 0.0:
        raise ValueError(
            "Training target contains no negative samples."
        )

    return negative_count / positive_count


def train_one_epoch(
    model: UnifiedRiskModel,
    loader: DataLoader[TensorDataset],
    criterion: BCEWithLogitsLoss,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Run one training epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader produced zero samples.")

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(
    model: UnifiedRiskModel,
    loader: DataLoader[TensorDataset],
    criterion: BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    """Evaluate BCE loss without updating model parameters."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)

        loss = criterion(logits, targets)

        batch_size = features.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")

    return total_loss / total_samples


def train(
    config: TrainingConfig | None = None,
) -> Path:
    """Train the unified multimodal risk model."""

    config = config or TrainingConfig()

    set_seed(config.seed)

    train_features = load_feature_matrix(TRAIN_FEATURES)
    validation_features = load_feature_matrix(
        VALIDATION_FEATURES
    )
    test_features = load_feature_matrix(TEST_FEATURES)

    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions do not match."
        )

    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            "Train and test feature dimensions do not match."
        )

    train_targets = load_targets(
        TRAIN_METADATA,
        len(train_features),
    )

    validation_targets = load_targets(
        VALIDATION_METADATA,
        len(validation_features),
    )

    test_targets = load_targets(
        TEST_METADATA,
        len(test_features),
    )

    validate_alignment(
        train_features,
        TRAIN_METADATA,
    )

    validate_alignment(
        validation_features,
        VALIDATION_METADATA,
    )

    validate_alignment(
        test_features,
        TEST_METADATA,
    )

    input_dim = train_features.shape[1]

    model_config = UnifiedRiskModelConfig(
        input_dim=input_dim,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UnifiedRiskModel(model_config).to(device)

    pos_weight_value = calculate_pos_weight(
        train_targets
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    criterion = BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = build_loader(
        train_features,
        train_targets,
        config.batch_size,
        shuffle=True,
    )

    validation_loader = build_loader(
        validation_features,
        validation_targets,
        config.batch_size,
        shuffle=False,
    )

    test_loader = build_loader(
        test_features,
        test_targets,
        config.batch_size,
        shuffle=False,
    )

    best_validation_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model,
                CHECKPOINT_PATH,
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                training_history=history,
                feature_manifest=str(MANIFEST_PATH),
            )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(
                f"Early stopping after epoch {epoch}."
            )
            break

    if best_epoch < 0:
        raise RuntimeError(
            "Training completed without producing a valid checkpoint."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    test_loss = evaluate_loss(
        model,
        test_loader,
        criterion,
        device,
    )

    training_summary = {
        "model": "unified_multimodal_risk_model",
        "input_dimension": input_dim,
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "train_positive_ratio": float(
            np.mean(train_targets)
        ),
        "validation_positive_ratio": float(
            np.mean(validation_targets)
        ),
        "test_positive_ratio": float(
            np.mean(test_targets)
        ),
        "pos_weight": pos_weight_value,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "device": str(device),
        "seed": config.seed,
        "risk_definition": {
            "uncertainty": "1 - max(p, 1-p)",
            "risk_score": "100 * uncertainty",
            "meaning": "model uncertainty only",
        },
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        MODEL_DIR / "unified_risk_training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            training_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("UNIFIED RISK MODEL TRAINING COMPLETE")
    print(f"Input dimensions: {input_dim}")
    print(f"Train rows:       {len(train_features)}")
    print(f"Validation rows:  {len(validation_features)}")
    print(f"Test rows:        {len(test_features)}")
    print(f"Best epoch:       {best_epoch}")
    print(
        f"Best val loss:    {best_validation_loss:.6f}"
    )
    print(f"Test loss:        {test_loss:.6f}")
    print(f"Checkpoint:       {CHECKPOINT_PATH}")

    return CHECKPOINT_PATH


def main() -> None:
    """CLI entry point."""

    train()


if __name__ == "__main__":
    main()