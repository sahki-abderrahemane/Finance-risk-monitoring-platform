from __future__ import annotations

import pytest
import torch

from vision_engine.model import (
    CNNConfig,
    ChartCNN,
    VisionModelError,
    build_cnn,
    count_parameters,
    model_summary,
)


def test_default_cnn_config() -> None:
    config = CNNConfig()

    assert config.input_channels == 3
    assert config.num_classes == 3
    assert config.base_channels == 32
    assert config.image_size == 224


def test_cnn_config_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError):
        CNNConfig(
            input_channels=0
        )


def test_cnn_config_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError):
        CNNConfig(
            num_classes=1
        )


def test_cnn_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        CNNConfig(
            dropout=1.0
        )


def test_model_can_be_constructed() -> None:
    model = ChartCNN()

    assert isinstance(
        model,
        ChartCNN,
    )


def test_build_cnn_returns_chart_cnn() -> None:
    model = build_cnn()

    assert isinstance(
        model,
        ChartCNN,
    )


def test_forward_output_shape() -> None:
    config = CNNConfig(
        image_size=224
    )

    model = ChartCNN(config)

    inputs = torch.randn(
        4,
        3,
        224,
        224,
    )

    outputs = model(inputs)

    assert outputs.shape == (
        4,
        3,
    )


def test_forward_supports_different_batch_sizes() -> None:
    model = ChartCNN()

    for batch_size in [1, 2, 8]:
        inputs = torch.randn(
            batch_size,
            3,
            224,
            224,
        )

        outputs = model(inputs)

        assert outputs.shape == (
            batch_size,
            3,
        )


def test_forward_rejects_non_image_tensor() -> None:
    model = ChartCNN()

    inputs = torch.randn(
        4,
        224,
        224,
    )

    with pytest.raises(
        VisionModelError,
        match="Expected input tensor",
    ):
        model(inputs)


def test_forward_rejects_wrong_channel_count() -> None:
    model = ChartCNN()

    inputs = torch.randn(
        4,
        1,
        224,
        224,
    )

    with pytest.raises(
        VisionModelError,
        match="input channels",
    ):
        model(inputs)


def test_model_produces_finite_logits() -> None:
    model = ChartCNN()

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    outputs = model(inputs)

    assert torch.isfinite(
        outputs
    ).all()


def test_model_can_compute_cross_entropy_loss() -> None:
    model = ChartCNN()

    inputs = torch.randn(
        4,
        3,
        224,
        224,
    )

    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    outputs = model(inputs)

    loss = torch.nn.functional.cross_entropy(
        outputs,
        labels,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_model_has_trainable_parameters() -> None:
    model = ChartCNN()

    parameters = count_parameters(
        model
    )

    assert parameters > 0


def test_model_summary_contains_expected_fields() -> None:
    model = ChartCNN()

    summary = model_summary(
        model
    )

    assert summary["model_class"] == "ChartCNN"
    assert summary["input_channels"] == 3
    assert summary["num_classes"] == 3
    assert summary["trainable_parameters"] > 0


def test_model_parameters_change_after_optimizer_step() -> None:
    torch.manual_seed(42)

    model = ChartCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    labels = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    outputs = model(inputs)

    loss = torch.nn.functional.cross_entropy(
        outputs,
        labels,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new in zip(
            before,
            model.parameters(),
        )
    )

    assert changed