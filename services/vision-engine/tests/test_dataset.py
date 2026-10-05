from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from vision_engine.dataset import (
    ChartImageDataset,
    VisionDatasetConfig,
    VisionDatasetError,
    chronological_split,
    load_chart_metadata,
)


def _create_image(
    path: Path,
    size: tuple[int, int] = (64, 64),
) -> None:
    """Create a minimal RGB test image."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        "RGB",
        size,
    )

    image.save(path)


def _build_metadata(
    tmp_path: Path,
    rows: int = 20,
) -> pd.DataFrame:
    """Build synthetic chart metadata for unit tests."""

    records: list[dict[str, object]] = []

    for index in range(rows):
        timestamp = pd.Timestamp(
            "2024-01-01"
        ) + pd.Timedelta(
            days=index
        )

        image_path = (
            tmp_path
            / f"chart_{index}.png"
        )

        _create_image(image_path)

        records.append(
            {
                "sample_id": f"SAMPLE_{index:04d}",
                "ticker": "AAPL",
                "timestamp": timestamp.isoformat(),
                "image_path": str(
                    image_path
                ),
                "label": (
                    "UP"
                    if index % 3 == 0
                    else "FLAT"
                    if index % 3 == 1
                    else "DOWN"
                ),
                "future_return": (
                    0.02
                    if index % 3 == 0
                    else 0.0
                    if index % 3 == 1
                    else -0.02
                ),
                "window_size": 30,
            }
        )

    return pd.DataFrame(records)


def test_vision_dataset_config_accepts_valid_ratios() -> None:
    config = VisionDatasetConfig(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert config.train_ratio == 0.70
    assert config.validation_ratio == 0.15
    assert config.test_ratio == 0.15


def test_vision_dataset_config_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError):
        VisionDatasetConfig(
            train_ratio=0.80,
            validation_ratio=0.15,
            test_ratio=0.15,
        )


def test_chronological_split_has_expected_sizes(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=20,
    )

    split = chronological_split(
        metadata,
        VisionDatasetConfig(
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
        ),
    )

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3


def test_chronological_split_is_temporally_ordered(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=20,
    )

    split = chronological_split(
        metadata
    )

    assert (
        split.train["timestamp"].max()
        <= split.validation["timestamp"].min()
    )

    assert (
        split.validation["timestamp"].max()
        <= split.test["timestamp"].min()
    )


def test_chronological_split_preserves_all_samples(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=20,
    )

    split = chronological_split(
        metadata
    )

    combined = pd.concat(
        [
            split.train,
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )

    assert len(combined) == len(metadata)

    assert set(
        combined["sample_id"]
    ) == set(
        metadata["sample_id"]
    )


def test_chronological_split_does_not_overlap_samples(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=20,
    )

    split = chronological_split(
        metadata
    )

    train_ids = set(
        split.train["sample_id"]
    )

    validation_ids = set(
        split.validation["sample_id"]
    )

    test_ids = set(
        split.test["sample_id"]
    )

    assert train_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert validation_ids.isdisjoint(
        test_ids
    )


def test_chart_image_dataset_loads_image_and_label(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=1,
    )

    dataset = ChartImageDataset(
        metadata=metadata,
        project_root=Path("/"),
    )

    image, label = dataset[0]

    assert isinstance(
        image,
        torch.Tensor,
    )

    assert isinstance(
        label,
        torch.Tensor,
    )

    assert image.ndim == 3
    assert image.shape[0] == 3
    assert image.shape[1] == 64
    assert image.shape[2] == 64

    assert label.dtype == torch.long
    assert label.item() == 2


def test_chart_image_dataset_maps_labels(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=3,
    )

    dataset = ChartImageDataset(
        metadata=metadata,
        project_root=Path("/"),
    )

    _, first_label = dataset[0]
    _, second_label = dataset[1]
    _, third_label = dataset[2]

    assert first_label.item() == 2
    assert second_label.item() == 1
    assert third_label.item() == 0


def test_chart_image_dataset_applies_transform(
    tmp_path: Path,
) -> None:
    from torchvision import transforms

    metadata = _build_metadata(
        tmp_path,
        rows=1,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
        ]
    )

    dataset = ChartImageDataset(
        metadata=metadata,
        project_root=Path("/"),
        transform=transform,
    )

    image, _ = dataset[0]

    assert image.shape == (
        3,
        224,
        224,
    )


def test_chart_image_dataset_rejects_missing_image(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=1,
    )

    metadata.loc[
        0,
        "image_path",
    ] = str(
        tmp_path
        / "does_not_exist.png"
    )

    with pytest.raises(
        VisionDatasetError,
        match="chart images do not exist",
    ):
        ChartImageDataset(
            metadata=metadata,
            project_root=Path("/"),
        )


def test_chart_image_dataset_rejects_invalid_label(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=1,
    )

    metadata.loc[
        0,
        "label",
    ] = "INVALID"

    with pytest.raises(
        VisionDatasetError,
        match="Unknown chart labels",
    ):
        ChartImageDataset(
            metadata=metadata,
            project_root=Path("/"),
        )


def test_chart_image_dataset_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=1,
    )

    metadata = metadata.drop(
        columns=["label"]
    )

    with pytest.raises(
        VisionDatasetError,
        match="missing required columns",
    ):
        ChartImageDataset(
            metadata=metadata,
            project_root=Path("/"),
        )


def test_load_chart_metadata_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        VisionDatasetError,
        match="does not exist",
    ):
        load_chart_metadata(
            missing_path
        )


def test_load_chart_metadata_normalizes_ticker_and_timestamp(
    tmp_path: Path,
) -> None:
    metadata_path = (
        tmp_path
        / "chart_metadata.csv"
    )

    dataframe = _build_metadata(
        tmp_path,
        rows=2,
    )

    dataframe["ticker"] = [
        " aapl ",
        "aapl",
    ]

    dataframe.to_csv(
        metadata_path,
        index=False,
    )

    loaded = load_chart_metadata(
        metadata_path
    )

    assert loaded["ticker"].tolist() == [
        "AAPL",
        "AAPL",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        loaded["timestamp"]
    )


def test_dataset_length_matches_metadata(
    tmp_path: Path,
) -> None:
    metadata = _build_metadata(
        tmp_path,
        rows=7,
    )

    dataset = ChartImageDataset(
        metadata=metadata,
        project_root=Path("/"),
    )

    assert len(dataset) == 7