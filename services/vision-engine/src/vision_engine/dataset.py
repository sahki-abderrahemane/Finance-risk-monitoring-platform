from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vision_engine.schemas import ChartLabel


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

METADATA_PATH: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "charts"
    / "chart_metadata.csv"
)


class VisionDatasetError(RuntimeError):
    """Raised when the CNN dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class VisionDatasetConfig:
    """Configuration for the CNN dataset."""

    image_size: int = 224

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    batch_size: int = 32
    num_workers: int = 0

    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate dataset configuration."""

        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0 for ratio in ratios):
            raise ValueError(
                "train_ratio, validation_ratio and test_ratio "
                "must all be positive."
            )

        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio "
                "must equal 1.0."
            )


@dataclass(frozen=True)
class VisionSplit:
    """Chronological dataset split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class ChartImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    PyTorch dataset for historical financial chart images.

    Labels:
        DOWN -> 0
        FLAT -> 1
        UP   -> 2
    """

    LABEL_TO_INDEX: Final[dict[str, int]] = {
        ChartLabel.DOWN.value: 0,
        ChartLabel.FLAT.value: 1,
        ChartLabel.UP.value: 2,
    }

    def __init__(
        self,
        metadata: pd.DataFrame,
        project_root: Path = PROJECT_ROOT,
        transform: transforms.Compose | None = None,
    ) -> None:
        if metadata.empty:
            raise VisionDatasetError(
                "Cannot create ChartImageDataset from empty metadata."
            )

        self.metadata = metadata.reset_index(drop=True).copy()
        self.project_root = project_root
        self.transform = transform

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        """Validate all metadata required by the dataset."""

        required_columns = {
            "sample_id",
            "ticker",
            "timestamp",
            "image_path",
            "label",
            "future_return",
            "window_size",
        }

        missing = required_columns.difference(
            self.metadata.columns
        )

        if missing:
            raise VisionDatasetError(
                f"Metadata is missing required columns: "
                f"{sorted(missing)}"
            )

        invalid_labels = set(
            self.metadata["label"].astype(str)
        ).difference(
            self.LABEL_TO_INDEX
        )

        if invalid_labels:
            raise VisionDatasetError(
                f"Unknown chart labels: {sorted(invalid_labels)}"
            )

        missing_images: list[str] = []

        for image_path in self.metadata["image_path"]:
            path = self._resolve_image_path(str(image_path))

            if not path.exists():
                missing_images.append(str(path))

        if missing_images:
            preview = missing_images[:5]

            raise VisionDatasetError(
                f"{len(missing_images)} chart images do not exist. "
                f"Examples: {preview}"
            )

    def _resolve_image_path(
        self,
        image_path: str,
    ) -> Path:
        """Resolve a metadata image path against the project root."""

        path = Path(image_path)

        if path.is_absolute():
            return path

        return self.project_root / path

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        """Load and transform one chart image."""

        row = self.metadata.iloc[index]

        image_path = self._resolve_image_path(
            str(row["image_path"])
        )

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise VisionDatasetError(
                f"Could not load chart image: {image_path}"
            ) from exc

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        label_name = str(row["label"])

        label_index = self.LABEL_TO_INDEX.get(
            label_name
        )

        if label_index is None:
            raise VisionDatasetError(
                f"Unknown label '{label_name}' "
                f"for sample {row['sample_id']}."
            )

        label_tensor = torch.tensor(
            label_index,
            dtype=torch.long,
        )

        return image_tensor, label_tensor


def load_chart_metadata(
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Load and validate chart metadata."""

    if not metadata_path.exists():
        raise VisionDatasetError(
            f"Chart metadata does not exist: {metadata_path}"
        )

    try:
        dataframe = pd.read_csv(
            metadata_path
        )
    except Exception as exc:
        raise VisionDatasetError(
            f"Could not read chart metadata: {metadata_path}"
        ) from exc

    if dataframe.empty:
        raise VisionDatasetError(
            f"Chart metadata is empty: {metadata_path}"
        )

    required_columns = {
        "sample_id",
        "ticker",
        "timestamp",
        "image_path",
        "label",
        "future_return",
        "window_size",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise VisionDatasetError(
            f"Chart metadata is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe["ticker"] = (
        dataframe["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if dataframe["timestamp"].isna().any():
        raise VisionDatasetError(
            "Chart metadata contains invalid timestamps."
        )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["future_return"] = pd.to_numeric(
        dataframe["future_return"],
        errors="coerce",
    )

    if dataframe["future_return"].isna().any():
        raise VisionDatasetError(
            "Chart metadata contains invalid future returns."
        )

    dataframe = dataframe.sort_values(
        ["timestamp", "ticker", "sample_id"]
    ).reset_index(drop=True)

    return dataframe


def chronological_split(
    metadata: pd.DataFrame,
    config: VisionDatasetConfig | None = None,
) -> VisionSplit:
    """
    Split chart metadata chronologically.

    No random shuffling is performed.

    This is essential because the target represents future market
    movement. Random splitting could place observations from later
    periods into training while earlier observations appear in test.
    """

    if config is None:
        config = VisionDatasetConfig()

    if metadata.empty:
        raise VisionDatasetError(
            "Cannot split empty chart metadata."
        )

    dataframe = metadata.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )

    dataframe = dataframe.sort_values(
        ["timestamp", "ticker", "sample_id"]
    ).reset_index(drop=True)

    total_rows = len(dataframe)

    if total_rows < 3:
        raise VisionDatasetError(
            "At least 3 samples are required for train/"
            "validation/test splitting."
        )

    train_end = int(
        total_rows * config.train_ratio
    )

    validation_end = train_end + int(
        total_rows * config.validation_ratio
    )

    # Guarantee that all three splits contain data.
    train_end = max(1, train_end)
    validation_end = max(
        train_end + 1,
        validation_end,
    )
    validation_end = min(
        validation_end,
        total_rows - 1,
    )

    train = dataframe.iloc[
        :train_end
    ].copy()

    validation = dataframe.iloc[
        train_end:validation_end
    ].copy()

    test = dataframe.iloc[
        validation_end:
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise VisionDatasetError(
            "Chronological split produced an empty partition."
        )

    # Explicit leakage checks.
    train_max = train["timestamp"].max()
    validation_min = validation["timestamp"].min()
    validation_max = validation["timestamp"].max()
    test_min = test["timestamp"].min()

    if train_max > validation_min:
        raise VisionDatasetError(
            "Temporal leakage detected between train "
            "and validation splits."
        )

    if validation_max > test_min:
        raise VisionDatasetError(
            "Temporal leakage detected between validation "
            "and test splits."
        )

    return VisionSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def build_train_transform(
    image_size: int,
) -> transforms.Compose:
    """
    Build training transforms.

    Augmentation is deliberately conservative because these are
    financial chart images where excessive geometric transformations
    could alter meaningful temporal structure.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.RandomHorizontalFlip(
                p=0.0
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def build_evaluation_transform(
    image_size: int,
) -> transforms.Compose:
    """Build deterministic validation/test transforms."""

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def create_dataloaders(
    metadata: pd.DataFrame | None = None,
    config: VisionDatasetConfig | None = None,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """Create chronological train/validation/test DataLoaders."""

    if config is None:
        config = VisionDatasetConfig()

    if metadata is None:
        metadata = load_chart_metadata()

    split = chronological_split(
        metadata,
        config,
    )

    train_dataset = ChartImageDataset(
        metadata=split.train,
        transform=build_train_transform(
            config.image_size
        ),
    )

    validation_dataset = ChartImageDataset(
        metadata=split.validation,
        transform=build_evaluation_transform(
            config.image_size
        ),
    )

    test_dataset = ChartImageDataset(
        metadata=split.test,
        transform=build_evaluation_transform(
            config.image_size
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


def get_split_summary(
    split: VisionSplit,
) -> dict[str, object]:
    """Return useful statistics for the three dataset partitions."""

    def summarize(
        dataframe: pd.DataFrame,
    ) -> dict[str, object]:
        labels = (
            dataframe["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        return {
            "rows": int(len(dataframe)),
            "start": dataframe["timestamp"]
            .min()
            .isoformat(),
            "end": dataframe["timestamp"]
            .max()
            .isoformat(),
            "tickers": sorted(
                dataframe["ticker"]
                .unique()
                .tolist()
            ),
            "labels": {
                str(key): int(value)
                for key, value in labels.items()
            },
        }

    return {
        "train": summarize(split.train),
        "validation": summarize(split.validation),
        "test": summarize(split.test),
    }


def main() -> None:
    """Inspect the generated CNN dataset."""

    print(
        "Loading Sentinel-AI chart dataset..."
    )

    config = VisionDatasetConfig()

    metadata = load_chart_metadata()

    split = chronological_split(
        metadata,
        config,
    )

    summary = get_split_summary(split)

    print()
    print(
        f"Total samples: {len(metadata)}"
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        information = summary[split_name]

        print()
        print(
            f"{split_name.capitalize()}:"
        )
        print(
            f"  Rows: {information['rows']}"
        )
        print(
            f"  Start: {information['start']}"
        )
        print(
            f"  End: {information['end']}"
        )
        print(
            f"  Tickers: {information['tickers']}"
        )
        print(
            f"  Labels: {information['labels']}"
        )

    train_loader, validation_loader, test_loader = (
        create_dataloaders(
            metadata,
            config,
        )
    )

    train_images, train_labels = next(
        iter(train_loader)
    )

    print()
    print("First training batch:")
    print(
        f"  Image tensor shape: "
        f"{tuple(train_images.shape)}"
    )
    print(
        f"  Label tensor shape: "
        f"{tuple(train_labels.shape)}"
    )

    print()
    print(
        f"Train batches: {len(train_loader)}"
    )
    print(
        f"Validation batches: {len(validation_loader)}"
    )
    print(
        f"Test batches: {len(test_loader)}"
    )

    print()
    print(
        "CNN dataset layer is ready."
    )


if __name__ == "__main__":
    main()