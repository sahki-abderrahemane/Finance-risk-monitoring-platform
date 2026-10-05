from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from multimodal_engine.unified_risk_model import (
    UnifiedRiskModel,
    UnifiedRiskModelConfig,
    load_checkpoint,
    probability_to_risk_score,
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

EVALUATION_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "evaluation"
)

CHECKPOINT_PATH: Final[Path] = (
    MODEL_DIR / "unified_risk_model_best.pt"
)

TRAIN_FEATURES_PATH: Final[Path] = (
    FEATURE_DIR / "train_fused_features.npy"
)

VALIDATION_FEATURES_PATH: Final[Path] = (
    FEATURE_DIR / "validation_fused_features.npy"
)

TEST_FEATURES_PATH: Final[Path] = (
    FEATURE_DIR / "test_fused_features.npy"
)

TRAIN_METADATA_PATH: Final[Path] = (
    FEATURE_DIR / "train_fused_metadata.csv"
)

VALIDATION_METADATA_PATH: Final[Path] = (
    FEATURE_DIR / "validation_fused_metadata.csv"
)

TEST_METADATA_PATH: Final[Path] = (
    FEATURE_DIR / "test_fused_metadata.csv"
)

MANIFEST_PATH: Final[Path] = (
    FEATURE_DIR / "fusion_manifest.json"
)

RESULTS_CSV_PATH: Final[Path] = (
    EVALUATION_DIR / "multimodal_evaluation_results.csv"
)

SUMMARY_JSON_PATH: Final[Path] = (
    EVALUATION_DIR / "multimodal_evaluation_summary.json"
)

PREDICTIONS_PATH: Final[Path] = (
    EVALUATION_DIR / "multimodal_test_predictions.csv"
)

ABLATION_CHECKPOINT_DIR: Final[Path] = (
    MODEL_DIR / "ablations"
)

SEED: Final[int] = 42
BATCH_SIZE: Final[int] = 64
LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-4
MAX_EPOCHS: Final[int] = 50
PATIENCE: Final[int] = 8

MARKET_DIM: Final[int] = 5
TEXT_DIM: Final[int] = 768
VISION_DIM: Final[int] = 128
EXPECTED_FULL_DIM: Final[int] = (
    MARKET_DIM + TEXT_DIM + VISION_DIM
)

ECE_BINS: Final[int] = 10


@dataclass(frozen=True)
class AblationSpec:
    """Feature groups included in an ablation experiment."""

    name: str
    groups: tuple[str, ...]


ABLATIONS: Final[tuple[AblationSpec, ...]] = (
    AblationSpec(
        name="market_only",
        groups=("market",),
    ),
    AblationSpec(
        name="text_only",
        groups=("text",),
    ),
    AblationSpec(
        name="vision_only",
        groups=("vision",),
    ),
    AblationSpec(
        name="market_text",
        groups=("market", "text"),
    ),
    AblationSpec(
        name="market_vision",
        groups=("market", "vision"),
    ),
    AblationSpec(
        name="text_vision",
        groups=("text", "vision"),
    ),
    AblationSpec(
        name="full_multimodal",
        groups=("market", "text", "vision"),
    ),
)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for an ablation head."""

    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    seed: int = SEED


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(path: Path) -> np.ndarray:
    """Load and validate a fused feature matrix."""

    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix does not exist: {path}"
        )

    features = np.load(path)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix at {path}; "
            f"received {features.shape}."
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
    """Load target_direction_1d from fusion metadata."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata does not exist: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required = {
        "ticker",
        "timestamp",
        "target_direction_1d",
    }

    missing = sorted(
        required.difference(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Missing required metadata columns: "
            + ", ".join(missing)
        )

    if len(metadata) != expected_rows:
        raise ValueError(
            f"Feature/metadata mismatch for {metadata_path}: "
            f"{expected_rows} vs {len(metadata)}."
        )

    target = pd.to_numeric(
        metadata["target_direction_1d"],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "target_direction_1d contains invalid values."
        )

    values = target.to_numpy(dtype=np.float32)

    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(
            "target_direction_1d must contain only 0 and 1."
        )

    return values


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata used for prediction-level reporting."""

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata does not exist: {path}"
        )

    metadata = pd.read_csv(path)

    forbidden_inputs = {
        "target_return_1d",
        "target_direction_1d",
        "future_return",
        "vision_future_return",
        "vision_label",
        "future_return_1d",
        "future_direction_1d",
        "future_direction",
    }

    if "future_return" not in metadata.columns:
        raise ValueError(
            "Fusion metadata must retain future_return "
            "for leakage auditing."
        )

    # The metadata is allowed to contain future-derived fields
    # because they are audit/target fields. They are never passed
    # into the feature matrix.
    _ = forbidden_inputs

    return metadata


def validate_feature_layout(
    features: np.ndarray,
) -> None:
    """Validate the known market/text/vision fused representation."""

    if features.shape[1] != EXPECTED_FULL_DIM:
        raise ValueError(
            "Unexpected fused feature dimension. "
            f"Expected {EXPECTED_FULL_DIM} "
            f"(market={MARKET_DIM}, text={TEXT_DIM}, "
            f"vision={VISION_DIM}), received "
            f"{features.shape[1]}."
        )


def feature_indices(groups: tuple[str, ...]) -> np.ndarray:
    """Return feature indices for selected modality groups."""

    ranges: list[np.ndarray] = []

    if "market" in groups:
        ranges.append(
            np.arange(
                0,
                MARKET_DIM,
                dtype=np.int64,
            )
        )

    if "text" in groups:
        ranges.append(
            np.arange(
                MARKET_DIM,
                MARKET_DIM + TEXT_DIM,
                dtype=np.int64,
            )
        )

    if "vision" in groups:
        start = MARKET_DIM + TEXT_DIM

        ranges.append(
            np.arange(
                start,
                start + VISION_DIM,
                dtype=np.int64,
            )
        )

    if not ranges:
        raise ValueError(
            "At least one modality must be selected."
        )

    return np.concatenate(ranges)


def select_features(
    features: np.ndarray,
    groups: tuple[str, ...],
) -> np.ndarray:
    """Select only the requested modality representations."""

    validate_feature_layout(features)

    indices = feature_indices(groups)

    selected = features[:, indices]

    if selected.ndim != 2:
        raise RuntimeError(
            "Selected feature matrix is not two-dimensional."
        )

    if not np.isfinite(selected).all():
        raise ValueError(
            "Selected features contain non-finite values."
        )

    return selected.astype(
        np.float32,
        copy=False,
    )


def calculate_pos_weight(
    targets: np.ndarray,
) -> float:
    """Calculate positive-class weighting from training data only."""

    positive = float(np.sum(targets == 1.0))
    negative = float(np.sum(targets == 0.0))

    if positive == 0.0 or negative == 0.0:
        raise ValueError(
            "Both target classes must exist in training data."
        )

    return negative / positive


def build_loader(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[TensorDataset]:
    """Build a PyTorch DataLoader."""

    if len(features) != len(targets):
        raise ValueError(
            "Feature and target lengths must match."
        )

    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: UnifiedRiskModel,
    loader: DataLoader[TensorDataset],
    criterion: BCEWithLogitsLoss,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Train one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        count = features.shape[0]

        total_loss += (
            float(loss.item()) * count
        )

        total_samples += count

    if total_samples == 0:
        raise RuntimeError(
            "Training loader produced zero samples."
        )

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(
    model: UnifiedRiskModel,
    loader: DataLoader[TensorDataset],
    criterion: BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    """Evaluate BCE loss."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)

        loss = criterion(
            logits,
            targets,
        )

        count = features.shape[0]

        total_loss += (
            float(loss.item()) * count
        )

        total_samples += count

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation loader produced zero samples."
        )

    return total_loss / total_samples


def train_ablation(
    name: str,
    groups: tuple[str, ...],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> UnifiedRiskModel:
    """
    Train an ablation model using only the selected modalities.

    The test set is intentionally absent from this function.
    """

    set_seed(config.seed)

    selected_train = select_features(
        train_features,
        groups,
    )

    selected_validation = select_features(
        validation_features,
        groups,
    )

    if selected_train.shape[1] != selected_validation.shape[1]:
        raise ValueError(
            f"Ablation '{name}' has inconsistent dimensions."
        )

    model = UnifiedRiskModel(
        UnifiedRiskModelConfig(
            input_dim=selected_train.shape[1],
        )
    ).to(device)

    pos_weight_value = calculate_pos_weight(
        train_targets
    )

    criterion = BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = build_loader(
        selected_train,
        train_targets,
        batch_size=config.batch_size,
        shuffle=True,
    )

    validation_loader = build_loader(
        selected_validation,
        validation_targets,
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_validation_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    patience_counter = 0
    best_epoch = -1

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

        print(
            f"[{name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"Ablation '{name}' failed to produce a checkpoint."
        )

    model.load_state_dict(best_state)
    model.eval()

    checkpoint_path = (
        ABLATION_CHECKPOINT_DIR
        / f"{name}_best.pt"
    )

    save_checkpoint(
        model,
        checkpoint_path,
        epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        feature_manifest=str(MANIFEST_PATH),
    )

    return model


@torch.no_grad()
def predict_probabilities(
    model: UnifiedRiskModel,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Generate binary probabilities."""

    tensor = torch.from_numpy(features).to(device)

    model.eval()

    probabilities = (
        model.predict_probability(tensor)
        .detach()
        .cpu()
        .numpy()
    )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model generated non-finite probabilities."
        )

    return probabilities.astype(
        np.float64,
        copy=False,
    )


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = ECE_BINS,
) -> float:
    """
    Calculate Expected Calibration Error.

    Confidence is max(p, 1-p), while correctness is whether
    the thresholded prediction matches the observed target.
    """

    if bins <= 0:
        raise ValueError(
            "bins must be greater than zero."
        )

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Targets and probabilities must have equal length."
        )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    correctness = (
        predictions == y_true
    ).astype(np.float64)

    total = len(y_true)

    if total == 0:
        raise ValueError(
            "Cannot calculate ECE on an empty dataset."
        )

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    ece = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        accuracy = float(
            correctness[mask].mean()
        )

        mean_confidence = float(
            confidence[mask].mean()
        )

        ece += (
            count / total
        ) * abs(
            accuracy - mean_confidence
        )

    return float(ece)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate classification and calibration metrics."""

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Targets and probabilities must have equal length."
        )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        raise ValueError(
            "Both target classes are required for evaluation."
        )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    risk_scores = probability_to_risk_score(
        probabilities
    )

    metrics = {
        "rows": float(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
        ),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "risk_score_mean": float(
            np.mean(risk_scores)
        ),
        "risk_score_median": float(
            np.median(risk_scores)
        ),
        "risk_score_std": float(
            np.std(risk_scores)
        ),
        "risk_score_min": float(
            np.min(risk_scores)
        ),
        "risk_score_max": float(
            np.max(risk_scores)
        ),
    }

    return metrics


def evaluate_model(
    name: str,
    model: UnifiedRiskModel,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    test_metadata: pd.DataFrame,
    device: torch.device,
    groups: tuple[str, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate one model and produce prediction-level output."""

    selected_test = select_features(
        test_features,
        groups,
    )

    probabilities = predict_probabilities(
        model,
        selected_test,
        device,
    )

    metrics = calculate_metrics(
        test_targets,
        probabilities,
    )

    metrics["model"] = name
    metrics["modalities"] = "+".join(groups)
    metrics["feature_dimension"] = float(
        selected_test.shape[1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int64)

    risk_scores = probability_to_risk_score(
        probabilities
    )

    prediction_frame = test_metadata[
        [
            "ticker",
            "timestamp",
        ]
    ].copy()

    prediction_frame["model"] = name
    prediction_frame["probability_up"] = probabilities
    prediction_frame["prediction"] = predictions
    prediction_frame["target_direction_1d"] = (
        test_targets.astype(np.int64)
    )
    prediction_frame["risk_score"] = risk_scores

    return metrics, prediction_frame


def evaluate_pretrained_full_model(
    test_features: np.ndarray,
    test_targets: np.ndarray,
    test_metadata: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the previously trained unified full model."""

    model, checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,
    )
    model.to(device)

    if model.config.input_dim != test_features.shape[1]:
        raise ValueError(
            "Unified model input dimension does not match "
            "the test fused representation."
        )

    metrics, predictions = evaluate_model(
        "full_multimodal_pretrained",
        model,
        test_features,
        test_targets,
        test_metadata,
        device,
        ("market", "text", "vision"),
    )

    metrics["best_epoch"] = float(
        checkpoint.get("epoch", -1)
    )

    metrics["best_validation_loss"] = float(
        checkpoint.get(
            "best_validation_loss",
            math.nan,
        )
    )

    return metrics, predictions


def save_results(
    results: list[dict[str, float]],
    predictions: pd.DataFrame,
) -> None:
    """Persist evaluation artifacts."""

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_frame = pd.DataFrame(results)

    preferred_columns = [
        "model",
        "modalities",
        "feature_dimension",
        "rows",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "macro_f1",
        "precision",
        "recall",
        "log_loss",
        "brier_score",
        "ece",
        "risk_score_mean",
        "risk_score_median",
        "risk_score_std",
        "risk_score_min",
        "risk_score_max",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "best_epoch",
        "best_validation_loss",
    ]

    available = [
        column
        for column in preferred_columns
        if column in results_frame.columns
    ]

    remaining = [
        column
        for column in results_frame.columns
        if column not in available
    ]

    results_frame = results_frame[
        available + remaining
    ]

    results_frame.to_csv(
        RESULTS_CSV_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    summary = {
        "evaluation": "Phase 2 multimodal evaluation",
        "test_rows": int(
            predictions[
                predictions["model"]
                == "full_multimodal_pretrained"
            ].shape[0]
        ),
        "feature_layout": {
            "market": MARKET_DIM,
            "text": TEXT_DIM,
            "vision": VISION_DIM,
            "total": EXPECTED_FULL_DIM,
        },
        "calibration": {
            "metric": "Expected Calibration Error",
            "bins": ECE_BINS,
            "lower_is_better": True,
        },
        "risk_definition": {
            "probability": "sigmoid(logit)",
            "uncertainty": "1 - max(p, 1-p)",
            "risk_score": "100 * uncertainty",
            "maximum_binary_uncertainty_score": 50.0,
            "interpretation": "model uncertainty only",
        },
        "leakage_policy": {
            "test_used_for_model_selection": False,
            "future_derived_metadata_used_as_input": False,
            "target": "target_direction_1d",
        },
        "results": results,
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def run_evaluation() -> None:
    """Run the complete multimodal evaluation."""

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_features = load_features(
        TRAIN_FEATURES_PATH
    )

    validation_features = load_features(
        VALIDATION_FEATURES_PATH
    )

    test_features = load_features(
        TEST_FEATURES_PATH
    )

    validate_feature_layout(train_features)
    validate_feature_layout(validation_features)
    validate_feature_layout(test_features)

    train_targets = load_targets(
        TRAIN_METADATA_PATH,
        len(train_features),
    )

    validation_targets = load_targets(
        VALIDATION_METADATA_PATH,
        len(validation_features),
    )

    test_targets = load_targets(
        TEST_METADATA_PATH,
        len(test_features),
    )

    test_metadata = load_metadata(
        TEST_METADATA_PATH
    )

    if len(test_metadata) != len(test_features):
        raise ValueError(
            "Test metadata and test features are misaligned."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Fusion manifest does not exist: {MANIFEST_PATH}"
        )

    # First evaluate the already-trained production candidate.
    pretrained_metrics, pretrained_predictions = (
        evaluate_pretrained_full_model(
            test_features,
            test_targets,
            test_metadata,
            device,
        )
    )

    results: list[dict[str, float]] = [
        pretrained_metrics
    ]

    prediction_frames = [
        pretrained_predictions
    ]

    training_config = TrainingConfig()

    # Retrained ablations are independently selected using
    # validation loss only. The test set remains untouched until
    # final evaluation.
    for ablation in ABLATIONS:
        if ablation.name == "full_multimodal":
            continue

        print()
        print(
            f"TRAINING ABLATION: {ablation.name}"
        )
        print(
            f"Modalities: {', '.join(ablation.groups)}"
        )

        model = train_ablation(
            name=ablation.name,
            groups=ablation.groups,
            train_features=train_features,
            train_targets=train_targets,
            validation_features=validation_features,
            validation_targets=validation_targets,
            config=training_config,
            device=device,
        )

        metrics, predictions = evaluate_model(
            ablation.name,
            model,
            test_features,
            test_targets,
            test_metadata,
            device,
            ablation.groups,
        )

        results.append(metrics)
        prediction_frames.append(predictions)

    # Also train a fresh full-model head so the ablation comparison
    # is based on identical training logic. The original pretrained
    # model remains separately reported above.
    print()
    print("TRAINING ABLATION: full_multimodal")

    full_model = train_ablation(
        name="full_multimodal",
        groups=("market", "text", "vision"),
        train_features=train_features,
        train_targets=train_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
        config=training_config,
        device=device,
    )

    full_metrics, full_predictions = evaluate_model(
        "full_multimodal",
        full_model,
        test_features,
        test_targets,
        test_metadata,
        device,
        ("market", "text", "vision"),
    )

    results.append(full_metrics)
    prediction_frames.append(full_predictions)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    save_results(
        results,
        predictions,
    )

    results_frame = pd.DataFrame(results)

    print()
    print("MULTIMODAL EVALUATION COMPLETE")
    print()
    print(
        results_frame[
            [
                "model",
                "modalities",
                "feature_dimension",
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "macro_f1",
                "log_loss",
                "brier_score",
                "ece",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"Results:     {RESULTS_CSV_PATH}"
    )
    print(
        f"Predictions: {PREDICTIONS_PATH}"
    )
    print(
        f"Summary:     {SUMMARY_JSON_PATH}"
    )


def main() -> None:
    """CLI entry point."""

    run_evaluation()


if __name__ == "__main__":
    main()