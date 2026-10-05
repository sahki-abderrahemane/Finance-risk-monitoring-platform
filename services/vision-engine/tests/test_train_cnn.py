from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_engine.model import CNNConfig, ChartCNN
from vision_engine.train_cnn import (
    TrainingConfig,
    _calculate_metrics,
    resolve_device,
    set_seed,
)


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.epochs == 20
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.patience == 5
    assert config.random_seed == 42


def test_training_config_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            epochs=0
        )


def test_training_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            learning_rate=0
        )


def test_training_config_rejects_invalid_weight_decay() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            weight_decay=-1
        )


def test_training_config_rejects_invalid_scheduler_factor() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            scheduler_factor=1.0
        )


def test_seed_produces_reproducible_torch_values() -> None:
    set_seed(42)

    first = torch.randn(
        5
    )

    set_seed(42)

    second = torch.randn(
        5
    )

    assert torch.equal(
        first,
        second,
    )


def test_calculate_metrics_perfect_predictions() -> None:
    labels = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    predictions = [
        0,
        1,
        2,
        0,
        1,
        2,
    ]

    accuracy, macro_f1 = _calculate_metrics(
        predictions,
        labels,
    )

    assert accuracy == pytest.approx(
        1.0
    )

    assert macro_f1 == pytest.approx(
        1.0
    )


def test_calculate_metrics_handles_missing_class_predictions() -> None:
    labels = [
        0,
        1,
        2,
        2,
    ]

    predictions = [
        0,
        1,
        1,
        1,
    ]

    accuracy, macro_f1 = _calculate_metrics(
        predictions,
        labels,
    )

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_f1 <= 1.0


def test_calculate_metrics_rejects_empty_labels() -> None:
    with pytest.raises(
        RuntimeError,
        match="empty labels",
    ):
        _calculate_metrics(
            [],
            [],
        )


def test_resolve_cpu_device() -> None:
    device = resolve_device(
        "cpu"
    )

    assert device.type == "cpu"


def test_resolve_auto_device() -> None:
    device = resolve_device(
        "auto"
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_resolve_invalid_device() -> None:
    with pytest.raises(
        RuntimeError,
        match="Unsupported device",
    ):
        resolve_device(
            "invalid-device"
        )


def test_resolve_cuda_behavior() -> None:
    if torch.cuda.is_available():
        device = resolve_device(
            "cuda"
        )

        assert device.type == "cuda"

    else:
        with pytest.raises(
            RuntimeError,
            match="CUDA.*unavailable",
        ):
            resolve_device(
                "cuda"
            )


def test_small_cnn_forward_and_loss() -> None:
    config = CNNConfig(
        base_channels=8,
        image_size=64,
    )

    model = ChartCNN(
        config
    )

    images = torch.randn(
        4,
        3,
        64,
        64,
    )

    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    logits = model(
        images
    )

    loss = nn.CrossEntropyLoss()(
        logits,
        labels,
    )

    assert logits.shape == (
        4,
        3,
    )

    assert torch.isfinite(
        loss
    )


def test_model_can_run_on_cpu() -> None:
    model = ChartCNN(
        CNNConfig(
            base_channels=8,
            image_size=32,
        )
    )

    model = model.to(
        torch.device("cpu")
    )

    images = torch.randn(
        2,
        3,
        32,
        32,
    )

    outputs = model(
        images
    )

    assert outputs.shape == (
        2,
        3,
    )


def test_training_dataloader_produces_expected_shapes() -> None:
    images = torch.randn(
        6,
        3,
        32,
        32,
    )

    labels = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    batch_images, batch_labels = next(
        iter(loader)
    )

    assert batch_images.shape == (
        2,
        3,
        32,
        32,
    )

    assert batch_labels.shape == (
        2,
    )