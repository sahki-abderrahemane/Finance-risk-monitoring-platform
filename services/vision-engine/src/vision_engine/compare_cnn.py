from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import Tensor
from torch.utils.data import DataLoader

from vision_engine.dataset import VisionDatasetConfig, create_dataloaders
from vision_engine.model import CNNConfig, ChartCNN


PROJECT_ROOT = Path(__file__).resolve().parents[4]

BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "ml" / "models" / "cnn" / "chart_cnn_best.pt"
)

WEIGHTED_CHECKPOINT = (
    PROJECT_ROOT / "ml" / "models" / "cnn" / "chart_cnn_weighted_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "charts"
    / "evaluation"
)

COMPARISON_JSON = OUTPUT_DIR / "cnn_model_comparison.json"
COMPARISON_CSV = OUTPUT_DIR / "cnn_model_comparison.csv"

BASELINE_CONFUSION_MATRIX = (
    OUTPUT_DIR / "cnn_baseline_comparison_confusion_matrix.csv"
)

WEIGHTED_CONFUSION_MATRIX = (
    OUTPUT_DIR / "cnn_weighted_comparison_confusion_matrix.csv"
)

CLASS_NAMES = ["DOWN", "FLAT", "UP"]


class CNNComparisonError(RuntimeError):
    """Raised when CNN comparison cannot be completed."""


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    checkpoint: str
    test_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    majority_baseline: float
    accuracy_lift: float
    down_precision: float
    down_recall: float
    down_f1: float
    flat_precision: float
    flat_recall: float
    flat_f1: float
    up_precision: float
    up_recall: float
    up_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ChartCNN, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise CNNComparisonError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise CNNComparisonError(
            f"Failed to load checkpoint: {checkpoint_path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise CNNComparisonError(
            f"Invalid checkpoint format: {checkpoint_path}"
        )

    model_config_data = checkpoint.get("model_config")

    if isinstance(model_config_data, dict):
        allowed_fields = {
            "input_channels",
            "num_classes",
            "base_channels",
            "dropout",
            "image_size",
        }

        filtered_config = {
            key: value
            for key, value in model_config_data.items()
            if key in allowed_fields
        }

        model_config = CNNConfig(**filtered_config)
    else:
        model_config = CNNConfig()

    model = ChartCNN(model_config)

    state_dict = checkpoint.get("model_state_dict")

    if not isinstance(state_dict, dict):
        raise CNNComparisonError(
            f"Checkpoint is missing model_state_dict: {checkpoint_path}"
        )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise CNNComparisonError(
            f"Model architecture does not match checkpoint: "
            f"{checkpoint_path}"
        ) from exc

    model.to(device)
    model.eval()

    return model, checkpoint


def extract_predictions(
    model: ChartCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    targets: list[int] = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise CNNComparisonError(
                    "Expected dataloader batches to contain "
                    "(images, labels)."
                )

            images, labels = batch

            if not isinstance(images, Tensor):
                raise CNNComparisonError(
                    "Image batch must be a torch.Tensor."
                )

            if not isinstance(labels, Tensor):
                raise CNNComparisonError(
                    "Label batch must be a torch.Tensor."
                )

            images = images.to(device, non_blocking=True)

            logits = model(images)

            if logits.ndim != 2:
                raise CNNComparisonError(
                    f"Expected model output shape [B, C], got "
                    f"{tuple(logits.shape)}."
                )

            batch_predictions = torch.argmax(logits, dim=1)

            predictions.extend(
                batch_predictions.cpu().numpy().astype(int).tolist()
            )
            targets.extend(
                labels.cpu().numpy().astype(int).tolist()
            )

    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def calculate_majority_baseline(
    targets: np.ndarray,
) -> float:
    if targets.size == 0:
        raise CNNComparisonError(
            "Cannot calculate majority baseline on empty targets."
        )

    _, counts = np.unique(targets, return_counts=True)

    return float(counts.max() / targets.size)


def calculate_model_result(
    model_name: str,
    checkpoint_path: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> ModelResult:
    if targets.size == 0:
        raise CNNComparisonError(
            f"{model_name}: test targets are empty."
        )

    if predictions.size != targets.size:
        raise CNNComparisonError(
            f"{model_name}: prediction/target length mismatch."
        )

    accuracy = float(
        accuracy_score(
            targets,
            predictions,
        )
    )

    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            targets,
            predictions,
            labels=[0, 1, 2],
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        targets,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    majority_baseline = calculate_majority_baseline(targets)

    accuracy_lift = accuracy - majority_baseline

    down_metrics = report["DOWN"]
    flat_metrics = report["FLAT"]
    up_metrics = report["UP"]

    return ModelResult(
        model_name=model_name,
        checkpoint=str(checkpoint_path),
        test_samples=int(targets.size),
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        majority_baseline=majority_baseline,
        accuracy_lift=accuracy_lift,
        down_precision=float(down_metrics["precision"]),
        down_recall=float(down_metrics["recall"]),
        down_f1=float(down_metrics["f1-score"]),
        flat_precision=float(flat_metrics["precision"]),
        flat_recall=float(flat_metrics["recall"]),
        flat_f1=float(flat_metrics["f1-score"]),
        up_precision=float(up_metrics["precision"]),
        up_recall=float(up_metrics["recall"]),
        up_f1=float(up_metrics["f1-score"]),
    )


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1, 2],
    )

    lines = [
        "actual,predicted,count",
    ]

    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            lines.append(
                f"{actual_name},{predicted_name},"
                f"{int(matrix[actual_index, predicted_index])}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_comparison_csv(
    results: list[ModelResult],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = list(asdict(results[0]).keys())

    lines = [
        ",".join(field_names),
    ]

    for result in results:
        values = asdict(result)

        lines.append(
            ",".join(
                str(values[field])
                for field in field_names
            )
        )

    COMPARISON_CSV.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def save_comparison_json(
    results: list[ModelResult],
    test_class_distribution: dict[str, int],
) -> None:
    baseline = results[0]
    weighted = results[1]

    macro_f1_delta = (
        weighted.macro_f1 - baseline.macro_f1
    )

    accuracy_delta = (
        weighted.accuracy - baseline.accuracy
    )

    weighted_f1_delta = (
        weighted.weighted_f1 - baseline.weighted_f1
    )

    report = {
        "experiment": "cnn_baseline_vs_weighted_loss",
        "description": (
            "Comparison of baseline and class-weighted CNN "
            "checkpoints on the identical chronological test split."
        ),
        "class_names": CLASS_NAMES,
        "test_class_distribution": test_class_distribution,
        "models": [
            asdict(result)
            for result in results
        ],
        "deltas_weighted_minus_baseline": {
            "accuracy": accuracy_delta,
            "macro_f1": macro_f1_delta,
            "weighted_f1": weighted_f1_delta,
            "down_f1": (
                weighted.down_f1 - baseline.down_f1
            ),
            "flat_f1": (
                weighted.flat_f1 - baseline.flat_f1
            ),
            "up_f1": (
                weighted.up_f1 - baseline.up_f1
            ),
        },
        "interpretation": {
            "accuracy_improved": accuracy_delta > 0.0,
            "macro_f1_improved": macro_f1_delta > 0.0,
            "weighted_f1_improved": weighted_f1_delta > 0.0,
            "weighted_above_majority_baseline": (
                weighted.accuracy
                > weighted.majority_baseline
            ),
            "baseline_above_majority_baseline": (
                baseline.accuracy
                > baseline.majority_baseline
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_JSON.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_test_class_distribution(
    loader: DataLoader,
) -> dict[str, int]:
    counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise CNNComparisonError(
                "Expected dataloader batches to contain "
                "(images, labels)."
            )

        _, labels = batch

        if not isinstance(labels, Tensor):
            raise CNNComparisonError(
                "Labels must be torch.Tensor instances."
            )

        for label in labels.cpu().numpy().astype(int).tolist():
            if label < 0 or label >= len(CLASS_NAMES):
                raise CNNComparisonError(
                    f"Invalid class label: {label}"
                )

            counts[CLASS_NAMES[label]] += 1

    return counts


def print_result(result: ModelResult) -> None:
    print()
    print(f"{result.model_name}:")
    print(f"  Test samples:      {result.test_samples}")
    print(f"  Accuracy:          {result.accuracy:.4f}")
    print(f"  Macro-F1:          {result.macro_f1:.4f}")
    print(f"  Weighted-F1:       {result.weighted_f1:.4f}")
    print(f"  Majority baseline: {result.majority_baseline:.4f}")
    print(f"  Accuracy lift:     {result.accuracy_lift:.4f}")

    print()
    print("  Per-class F1:")
    print(f"    DOWN: {result.down_f1:.4f}")
    print(f"    FLAT: {result.flat_f1:.4f}")
    print(f"    UP:   {result.up_f1:.4f}")


def main() -> None:
    print("Starting Sentinel-AI CNN comparison...")

    set_seed(42)

    device = resolve_device()

    print(f"Device: {device}")

    if not BASELINE_CHECKPOINT.exists():
        raise CNNComparisonError(
            f"Missing baseline checkpoint: "
            f"{BASELINE_CHECKPOINT}"
        )

    if not WEIGHTED_CHECKPOINT.exists():
        raise CNNComparisonError(
            f"Missing weighted checkpoint: "
            f"{WEIGHTED_CHECKPOINT}"
        )

    dataset_config = VisionDatasetConfig()

    _, _, test_loader = create_dataloaders(
        config =dataset_config
    )

    test_class_distribution = build_test_class_distribution(
        test_loader
    )

    print()
    print("Test class distribution:")

    for class_name, count in test_class_distribution.items():
        print(f"  {class_name}: {count}")

    baseline_model, _ = load_checkpoint(
        BASELINE_CHECKPOINT,
        device,
    )

    weighted_model, _ = load_checkpoint(
        WEIGHTED_CHECKPOINT,
        device,
    )

    baseline_targets, baseline_predictions = extract_predictions(
        baseline_model,
        test_loader,
        device,
    )

    weighted_targets, weighted_predictions = extract_predictions(
        weighted_model,
        test_loader,
        device,
    )

    if not np.array_equal(
        baseline_targets,
        weighted_targets,
    ):
        raise CNNComparisonError(
            "Baseline and weighted evaluations do not use "
            "the same target ordering."
        )

    baseline_result = calculate_model_result(
        "baseline_cnn",
        BASELINE_CHECKPOINT,
        baseline_targets,
        baseline_predictions,
    )

    weighted_result = calculate_model_result(
        "weighted_cnn",
        WEIGHTED_CHECKPOINT,
        weighted_targets,
        weighted_predictions,
    )

    save_confusion_matrix(
        baseline_targets,
        baseline_predictions,
        BASELINE_CONFUSION_MATRIX,
    )

    save_confusion_matrix(
        weighted_targets,
        weighted_predictions,
        WEIGHTED_CONFUSION_MATRIX,
    )

    results = [
        baseline_result,
        weighted_result,
    ]

    save_comparison_csv(results)

    save_comparison_json(
        results,
        test_class_distribution,
    )

    print_result(baseline_result)
    print_result(weighted_result)

    print()
    print("Weighted minus baseline:")
    print(
        f"  Accuracy delta:    "
        f"{weighted_result.accuracy - baseline_result.accuracy:+.4f}"
    )
    print(
        f"  Macro-F1 delta:    "
        f"{weighted_result.macro_f1 - baseline_result.macro_f1:+.4f}"
    )
    print(
        f"  Weighted-F1 delta: "
        f"{weighted_result.weighted_f1 - baseline_result.weighted_f1:+.4f}"
    )

    print()
    print("Artifacts:")
    print(f"  {COMPARISON_JSON}")
    print(f"  {COMPARISON_CSV}")
    print(f"  {BASELINE_CONFUSION_MATRIX}")
    print(f"  {WEIGHTED_CONFUSION_MATRIX}")

    print()
    print("CNN comparison completed.")


if __name__ == "__main__":
    main()