from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from vision_engine.dataset import (
    VisionDatasetConfig,
    chronological_split,
    create_dataloaders,
    load_chart_metadata,
)
from vision_engine.model import CNNConfig, ChartCNN


PROJECT_ROOT = Path(__file__).resolve().parents[4]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "cnn"
    / "chart_cnn_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "charts"
    / "evaluation"
)

METRICS_PATH = OUTPUT_DIR / "cnn_evaluation.json"
CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "cnn_confusion_matrix.csv"
)
PREDICTIONS_PATH = (
    OUTPUT_DIR / "cnn_test_predictions.csv"
)

CLASS_NAMES = [
    "DOWN",
    "FLAT",
    "UP",
]


class VisionEvaluationError(RuntimeError):
    """Raised when CNN evaluation fails."""


def resolve_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model(
    device: torch.device,
) -> ChartCNN:
    """Load the best trained CNN checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise VisionEvaluationError(
            f"CNN checkpoint does not exist: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model_config_data = checkpoint.get(
        "model_config"
    )

    if model_config_data is None:
        raise VisionEvaluationError(
            "Checkpoint is missing model_config."
        )

    model_config = CNNConfig(
        **model_config_data
    )

    model = ChartCNN(
        model_config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


@torch.no_grad()
def predict_test_set(
    model: ChartCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Generate test predictions and confidence scores."""

    predictions: list[int] = []
    labels: list[int] = []
    confidences: list[float] = []

    for images, batch_labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        batch_labels = batch_labels.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        batch_predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        batch_confidences = torch.max(
            probabilities,
            dim=1,
        ).values

        predictions.extend(
            batch_predictions.cpu().tolist()
        )

        labels.extend(
            batch_labels.cpu().tolist()
        )

        confidences.extend(
            batch_confidences.cpu().tolist()
        )

    if not labels:
        raise VisionEvaluationError(
            "Test DataLoader produced no samples."
        )

    return (
        predictions,
        labels,
        confidences,
    )


def calculate_metrics(
    predictions: list[int],
    labels: list[int],
) -> dict[str, object]:
    """Calculate classification metrics."""

    if not labels:
        raise VisionEvaluationError(
            "Cannot evaluate an empty test set."
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

    weighted_f1 = f1_score(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2],
    )

    actual_counts = {
        CLASS_NAMES[index]: int(
            sum(label == index for label in labels)
        )
        for index in range(3)
    }

    predicted_counts = {
        CLASS_NAMES[index]: int(
            sum(
                prediction == index
                for prediction in predictions
            )
        )
        for index in range(3)
    }

    majority_class_count = max(
        actual_counts.values()
    )

    majority_baseline_accuracy = (
        majority_class_count / len(labels)
    )

    accuracy_lift = (
        float(accuracy)
        - float(majority_baseline_accuracy)
    )

    return {
        "test_samples": len(labels),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "majority_baseline_accuracy": float(
            majority_baseline_accuracy
        ),
        "accuracy_lift_over_majority": float(
            accuracy_lift
        ),
        "actual_class_distribution": actual_counts,
        "predicted_class_distribution": predicted_counts,
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    class_f1[index]
                ),
                "support": int(
                    support[index]
                ),
            }
            for index in range(3)
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(
    matrix: list[list[int]],
) -> None:
    """Save confusion matrix as CSV."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        matrix,
        index=[
            f"actual_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"predicted_{name}"
            for name in CLASS_NAMES
        ],
    )

    dataframe.to_csv(
        CONFUSION_MATRIX_PATH
    )


def save_predictions(
    metadata: pd.DataFrame,
    predictions: list[int],
    labels: list[int],
    confidences: list[float],
) -> None:
    """Save sample-level test predictions."""

    if not (
        len(metadata)
        == len(predictions)
        == len(labels)
        == len(confidences)
    ):
        raise VisionEvaluationError(
            "Prediction output lengths do not match "
            "test metadata."
        )

    output = metadata.copy()

    output["actual_label_id"] = labels
    output["predicted_label_id"] = predictions

    output["actual_label"] = [
        CLASS_NAMES[label]
        for label in labels
    ]

    output["predicted_label"] = [
        CLASS_NAMES[prediction]
        for prediction in predictions
    ]

    output["confidence"] = confidences

    output["correct"] = (
        output["actual_label_id"]
        == output["predicted_label_id"]
    )

    output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


def evaluate() -> dict[str, object]:
    """Run complete held-out CNN evaluation."""

    device = resolve_device()

    print(
        "Starting Sentinel-AI Vision evaluation..."
    )

    print(
        f"Device: {device}"
    )

    dataset_config = VisionDatasetConfig()

    metadata = load_chart_metadata()

    split = chronological_split(
        metadata,
        dataset_config,
    )

    (
        _train_loader,
        _validation_loader,
        test_loader,
    ) = create_dataloaders(
        metadata=metadata,
        config=dataset_config,
    )

    model = load_model(
        device
    )

    (
        predictions,
        labels,
        confidences,
    ) = predict_test_set(
        model=model,
        loader=test_loader,
        device=device,
    )

    metrics = calculate_metrics(
        predictions=predictions,
        labels=labels,
    )

    test_metadata = split.test.copy()

    # The DataLoader is deterministic for evaluation,
    # so its order corresponds to the chronological test metadata.
    save_predictions(
        metadata=test_metadata,
        predictions=predictions,
        labels=labels,
        confidences=confidences,
    )

    matrix = metrics[
        "confusion_matrix"
    ]

    if not isinstance(matrix, list):
        raise VisionEvaluationError(
            "Invalid confusion matrix."
        )

    save_confusion_matrix(
        matrix
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": "ChartCNN",
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "device": str(device),
        "classes": CLASS_NAMES,
        "metrics": metrics,
        "test_period": {
            "start": str(
                split.test["timestamp"].min()
            ),
            "end": str(
                split.test["timestamp"].max()
            ),
        },
        "artifacts": {
            "predictions": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "confusion_matrix": str(
                CONFUSION_MATRIX_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print()
    print(
        "CNN test evaluation:"
    )

    print(
        f"  Samples: "
        f"{metrics['test_samples']}"
    )

    print(
        f"  Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"  Weighted-F1: "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"  Majority baseline: "
        f"{metrics['majority_baseline_accuracy']:.4f}"
    )

    print(
        f"  Accuracy lift: "
        f"{metrics['accuracy_lift_over_majority']:.4f}"
    )

    print()
    print(
        "Per-class performance:"
    )

    per_class = metrics[
        "per_class"
    ]

    if isinstance(per_class, dict):
        for class_name, class_metrics in (
            per_class.items()
        ):
            if isinstance(
                class_metrics,
                dict,
            ):
                print(
                    f"  {class_name}: "
                    f"precision="
                    f"{class_metrics['precision']:.4f}, "
                    f"recall="
                    f"{class_metrics['recall']:.4f}, "
                    f"f1="
                    f"{class_metrics['f1']:.4f}"
                )

    print()
    print(
        "Artifacts:"
    )

    print(
        f"  {METRICS_PATH}"
    )

    print(
        f"  {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    print()
    print(
        "Vision evaluation completed."
    )

    return payload


def main() -> None:
    """CLI entry point."""

    evaluate()


if __name__ == "__main__":
    main()