from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vision_engine.dataset import (
    VisionDatasetConfig,
    chronological_split,
    create_dataloaders,
    get_split_summary,
    load_chart_metadata,
)
from vision_engine.model import (
    CNNConfig,
    ChartCNN,
    build_cnn,
    count_parameters,
    model_summary,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

MODEL_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "ml" / "models" / "cnn"
)

EVALUATION_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "ml" / "datasets" / "charts" / "evaluation"
)

BASELINE_CHECKPOINT_PATH: Final[Path] = (
    MODEL_OUTPUT_DIR / "chart_cnn_best.pt"
)

WEIGHTED_CHECKPOINT_PATH: Final[Path] = (
    MODEL_OUTPUT_DIR / "chart_cnn_weighted_best.pt"
)

WEIGHTED_HISTORY_PATH: Final[Path] = (
    EVALUATION_OUTPUT_DIR
    / "cnn_weighted_training_history.json"
)

WEIGHTED_SUMMARY_PATH: Final[Path] = (
    EVALUATION_OUTPUT_DIR
    / "cnn_weighted_training_summary.json"
)

CLASS_NAMES: Final[list[str]] = [
    "DOWN",
    "FLAT",
    "UP",
]


class CNNTrainingError(RuntimeError):
    """Raised when CNN training fails."""


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for weighted CNN training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    random_seed: int = 42
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.patience < 0:
            raise ValueError(
                "patience cannot be negative."
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive."
            )

        if not 0 < self.scheduler_factor < 1:
            raise ValueError(
                "scheduler_factor must be between 0 and 1."
            )

        if self.scheduler_patience < 0:
            raise ValueError(
                "scheduler_patience cannot be negative."
            )

        if self.scheduler_min_lr < 0:
            raise ValueError(
                "scheduler_min_lr cannot be negative."
            )


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics for one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    """Final weighted training result."""

    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    best_validation_macro_f1: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: str


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(
    device_name: str,
) -> torch.device:
    """Resolve requested training device."""

    normalized = device_name.strip().lower()

    if normalized == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise CNNTrainingError(
                "CUDA was explicitly requested but is unavailable."
            )

        return torch.device("cuda")

    if normalized == "cpu":
        return torch.device("cpu")

    raise CNNTrainingError(
        f"Unsupported device '{device_name}'. "
        "Use 'auto', 'cpu', or 'cuda'."
    )


def calculate_class_weights(
    labels: list[int],
    num_classes: int = 3,
) -> Tensor:
    """Calculate inverse-frequency class weights.

    The weights are calculated only from the training split.
    """

    if not labels:
        raise CNNTrainingError(
            "Cannot calculate class weights from empty labels."
        )

    counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if len(counts) != num_classes:
        raise CNNTrainingError(
            "Unexpected class-count vector."
        )

    if np.any(counts == 0):
        missing = [
            index
            for index, count in enumerate(counts)
            if count == 0
        ]

        raise CNNTrainingError(
            "Cannot calculate balanced weights because "
            f"classes are missing from the training split: "
            f"{missing}"
        )

    total = float(sum(counts))

    weights = total / (
        num_classes * counts.astype(np.float64)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def extract_labels_from_loader(
    loader: DataLoader,
) -> list[int]:
    """Extract labels from a DataLoader."""

    labels: list[int] = []

    for _, batch_labels in loader:
        labels.extend(
            batch_labels.cpu().tolist()
        )

    if not labels:
        raise CNNTrainingError(
            "DataLoader contains no labels."
        )

    return labels


def _calculate_metrics(
    predictions: list[int],
    labels: list[int],
) -> tuple[float, float]:
    """Calculate accuracy and macro-F1."""

    if not labels:
        raise CNNTrainingError(
            "Cannot calculate metrics from empty labels."
        )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    return (
        float(accuracy),
        float(macro_f1),
    )


def _validate_batch(
    images: Tensor,
    labels: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Validate and move a batch to the selected device."""

    if images.ndim != 4:
        raise CNNTrainingError(
            "Expected image batch with shape "
            "(batch, channels, height, width)."
        )

    if labels.ndim != 1:
        raise CNNTrainingError(
            "Expected label batch with shape (batch,)."
        )

    if images.shape[0] != labels.shape[0]:
        raise CNNTrainingError(
            "Image and label batch sizes do not match."
        )

    return (
        images.to(
            device,
            non_blocking=True,
        ),
        labels.to(
            device,
            non_blocking=True,
        ),
    )


def train_one_epoch(
    model: ChartCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float, float]:
    """Train for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    predictions: list[int] = []
    labels: list[int] = []

    for images, batch_labels in loader:
        images, batch_labels = _validate_batch(
            images,
            batch_labels,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            batch_labels,
        )

        if not torch.isfinite(loss):
            raise CNNTrainingError(
                "Non-finite training loss detected."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        predictions.extend(
            torch.argmax(
                logits,
                dim=1,
            )
            .detach()
            .cpu()
            .tolist()
        )

        labels.extend(
            batch_labels.detach()
            .cpu()
            .tolist()
        )

    if total_samples == 0:
        raise CNNTrainingError(
            "Training DataLoader produced zero samples."
        )

    accuracy, macro_f1 = _calculate_metrics(
        predictions,
        labels,
    )

    return (
        total_loss / total_samples,
        accuracy,
        macro_f1,
    )


@torch.no_grad()
def evaluate(
    model: ChartCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the CNN."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    predictions: list[int] = []
    labels: list[int] = []

    for images, batch_labels in loader:
        images, batch_labels = _validate_batch(
            images,
            batch_labels,
            device,
        )

        logits = model(images)

        loss = criterion(
            logits,
            batch_labels,
        )

        if not torch.isfinite(loss):
            raise CNNTrainingError(
                "Non-finite evaluation loss detected."
            )

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        predictions.extend(
            torch.argmax(
                logits,
                dim=1,
            )
            .cpu()
            .tolist()
        )

        labels.extend(
            batch_labels.cpu().tolist()
        )

    if total_samples == 0:
        raise CNNTrainingError(
            "Evaluation DataLoader produced zero samples."
        )

    accuracy, macro_f1 = _calculate_metrics(
        predictions,
        labels,
    )

    return (
        total_loss / total_samples,
        accuracy,
        macro_f1,
    )


def save_checkpoint(
    model: ChartCNN,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    validation_macro_f1: float,
    model_config: CNNConfig,
    training_config: TrainingConfig,
    dataset_config: VisionDatasetConfig,
    class_weights: Tensor,
    path: Path,
) -> None:
    """Save weighted-training checkpoint."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "dataset_config": asdict(dataset_config),
        "class_weights": class_weights.tolist(),
        "class_names": CLASS_NAMES,
        "weighted_loss": True,
    }

    torch.save(
        checkpoint,
        path,
    )


def train(
    model: ChartCNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    training_config: TrainingConfig,
    model_config: CNNConfig,
    dataset_config: VisionDatasetConfig,
    class_weights: Tensor,
    device: torch.device,
) -> tuple[TrainingResult, list[EpochMetrics]]:
    """Train using weighted cross-entropy."""

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=training_config.scheduler_min_lr,
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_validation_macro_f1 = 0.0
    best_epoch = 0

    epochs_without_improvement = 0
    stopped_early = False

    history: list[EpochMetrics] = []

    for epoch in range(
        1,
        training_config.epochs + 1,
    ):
        (
            train_loss,
            train_accuracy,
            train_macro_f1,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=(
                training_config.gradient_clip_norm
            ),
        )

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            validation_loss
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            train_macro_f1=train_macro_f1,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            validation_macro_f1=validation_macro_f1,
            learning_rate=current_lr,
        )

        history.append(
            epoch_metrics
        )

        print(
            f"Epoch {epoch:02d}/{training_config.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| train_acc={train_accuracy:.4f} "
            f"| train_f1={train_macro_f1:.4f} "
            f"| val_loss={validation_loss:.4f} "
            f"| val_acc={validation_accuracy:.4f} "
            f"| val_f1={validation_macro_f1:.4f} "
            f"| lr={current_lr:.6f}"
        )

        improved = (
            validation_loss
            < best_validation_loss
            - training_config.min_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_validation_accuracy = (
                validation_accuracy
            )
            best_validation_macro_f1 = (
                validation_macro_f1
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                validation_macro_f1=validation_macro_f1,
                model_config=model_config,
                training_config=training_config,
                dataset_config=dataset_config,
                class_weights=class_weights,
                path=WEIGHTED_CHECKPOINT_PATH,
            )

            print(
                "  ✓ New best weighted checkpoint saved."
            )

        else:
            epochs_without_improvement += 1

        if (
            training_config.patience > 0
            and epochs_without_improvement
            >= training_config.patience
        ):
            stopped_early = True

            print(
                f"  Early stopping after {epoch} epochs."
            )

            break

    if best_epoch == 0:
        raise CNNTrainingError(
            "Training completed without a valid checkpoint."
        )

    result = TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        best_validation_macro_f1=best_validation_macro_f1,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        checkpoint_path=str(
            WEIGHTED_CHECKPOINT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
    )

    return result, history


def save_history(
    history: list[EpochMetrics],
) -> None:
    """Save training history."""

    WEIGHTED_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with WEIGHTED_HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                asdict(metrics)
                for metrics in history
            ],
            file,
            indent=2,
        )


def save_summary(
    result: TrainingResult,
    model: ChartCNN,
    model_config: CNNConfig,
    training_config: TrainingConfig,
    dataset_config: VisionDatasetConfig,
    split_summary: dict[str, object],
    class_weights: Tensor,
    device: torch.device,
) -> None:
    """Save weighted training metadata."""

    WEIGHTED_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "training_result": asdict(result),
        "model": model_summary(model),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "dataset_config": asdict(dataset_config),
        "class_names": CLASS_NAMES,
        "class_weights": class_weights.tolist(),
        "weighted_loss": True,
        "device": str(device),
        "dataset": split_summary,
    }

    with WEIGHTED_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def main() -> None:
    """Run the weighted CNN experiment."""

    training_config = TrainingConfig()

    model_config = CNNConfig()

    dataset_config = VisionDatasetConfig()

    set_seed(
        training_config.random_seed
    )

    device = resolve_device(
        training_config.device
    )

    print(
        "Starting Sentinel-AI weighted CNN experiment..."
    )

    print(
        f"Device: {device}"
    )

    metadata = load_chart_metadata()

    split = chronological_split(
        metadata,
        dataset_config,
    )

    split_summary = get_split_summary(
        split
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        metadata=metadata,
        config=dataset_config,
    )

    train_labels = extract_labels_from_loader(
        train_loader
    )

    class_weights = calculate_class_weights(
        train_labels
    )

    class_counts = np.bincount(
        train_labels,
        minlength=3,
    )

    print()
    print(
        "Training class distribution:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name}: "
            f"{class_counts[index]}"
        )

    print()
    print(
        "Calculated class weights:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name}: "
            f"{class_weights[index].item():.6f}"
        )

    model = build_cnn(
        model_config
    ).to(device)

    print()
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    result, history = train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        training_config=training_config,
        model_config=model_config,
        dataset_config=dataset_config,
        class_weights=class_weights,
        device=device,
    )

    save_history(
        history
    )

    checkpoint = torch.load(
        WEIGHTED_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    save_summary(
        result=result,
        model=model,
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        split_summary=split_summary,
        class_weights=class_weights,
        device=device,
    )

    print()
    print(
        "Best weighted validation result:"
    )

    print(
        f"  Epoch: {result.best_epoch}"
    )

    print(
        f"  Loss: "
        f"{result.best_validation_loss:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{result.best_validation_accuracy:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{result.best_validation_macro_f1:.4f}"
    )

    print()
    print(
        "Weighted test result:"
    )

    print(
        f"  Loss: {test_loss:.4f}"
    )

    print(
        f"  Accuracy: {test_accuracy:.4f}"
    )

    print(
        f"  Macro-F1: {test_macro_f1:.4f}"
    )

    print()
    print(
        "Artifacts:"
    )

    print(
        f"  {WEIGHTED_CHECKPOINT_PATH}"
    )

    print(
        f"  {WEIGHTED_HISTORY_PATH}"
    )

    print(
        f"  {WEIGHTED_SUMMARY_PATH}"
    )

    print()
    print(
        "Weighted CNN experiment completed."
    )


if __name__ == "__main__":
    main()