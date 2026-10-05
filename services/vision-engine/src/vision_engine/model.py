from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class VisionModelError(RuntimeError):
    """Raised when the vision model configuration is invalid."""


@dataclass(frozen=True)
class CNNConfig:
    """Configuration for the Sentinel-AI chart CNN."""

    input_channels: int = 3
    num_classes: int = 3

    base_channels: int = 32

    dropout: float = 0.30

    image_size: int = 224

    def __post_init__(self) -> None:
        """Validate model configuration."""

        if self.input_channels <= 0:
            raise ValueError(
                "input_channels must be positive."
            )

        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if self.base_channels <= 0:
            raise ValueError(
                "base_channels must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be positive."
            )


class ConvBlock(nn.Module):
    """Convolutional feature-extraction block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the convolutional block."""
        return self.block(x)


class ChartCNN(nn.Module):
    """
    CNN baseline for historical financial chart classification.

    Input:
        RGB chart image.

    Output:
        Raw logits for:

            class 0 -> DOWN
            class 1 -> FLAT
            class 2 -> UP

    The model intentionally returns logits rather than probabilities.
    CrossEntropyLoss expects raw logits during training.
    """

    def __init__(
        self,
        config: CNNConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = CNNConfig()

        self.config = config

        channels_1 = config.base_channels
        channels_2 = config.base_channels * 2
        channels_3 = config.base_channels * 4

        self.features = nn.Sequential(
            ConvBlock(
                config.input_channels,
                channels_1,
            ),
            ConvBlock(
                channels_1,
                channels_2,
            ),
            ConvBlock(
                channels_2,
                channels_3,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=config.dropout
            ),
            nn.Linear(
                channels_3,
                config.num_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize trainable weights."""

        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.Linear,
            ):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Run a forward pass."""

        if x.ndim != 4:
            raise VisionModelError(
                "Expected input tensor with shape "
                "(batch, channels, height, width). "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[1] != self.config.input_channels:
            raise VisionModelError(
                f"Expected {self.config.input_channels} "
                f"input channels, received {x.shape[1]}."
            )

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


def build_cnn(
    config: CNNConfig | None = None,
) -> ChartCNN:
    """Construct the Sentinel-AI chart CNN."""

    return ChartCNN(
        config=config
    )


def count_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def model_summary(
    model: ChartCNN,
) -> dict[str, object]:
    """Return a serializable model summary."""

    return {
        "model_class": model.__class__.__name__,
        "input_channels": model.config.input_channels,
        "num_classes": model.config.num_classes,
        "base_channels": model.config.base_channels,
        "dropout": model.config.dropout,
        "image_size": model.config.image_size,
        "trainable_parameters": count_parameters(
            model
        ),
    }


def main() -> None:
    """Run a basic model sanity check."""

    config = CNNConfig()

    model = build_cnn(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    dummy_input = torch.randn(
        4,
        config.input_channels,
        config.image_size,
        config.image_size,
        device=device,
    )

    logits = model(
        dummy_input
    )

    print(
        "Sentinel-AI Chart CNN"
    )
    print(
        "====================="
    )
    print(
        f"Device: {device}"
    )
    print(
        f"Input shape: "
        f"{tuple(dummy_input.shape)}"
    )
    print(
        f"Output shape: "
        f"{tuple(logits.shape)}"
    )
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    print()
    print(
        "Model summary:"
    )

    for key, value in model_summary(
        model
    ).items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "CNN model sanity check passed."
    )


if __name__ == "__main__":
    main()