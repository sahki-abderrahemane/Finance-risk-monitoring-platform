from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from multimodal_engine.fusion_dataset import (
    FusionDataset,
    FusionSplit,
    MARKET_FEATURE_COLUMNS,
    chronological_split,
    extract_numeric_features,
)

try:
    from nlp_engine.sentiment import FinancialSentimentModel
except ImportError as exc:  # pragma: no cover - environment dependent
    FinancialSentimentModel = None  # type: ignore[assignment]
    _NLP_IMPORT_ERROR = exc
else:
    _NLP_IMPORT_ERROR = None

try:
    from vision_engine.model import ChartCNN, CNNConfig
except ImportError as exc:  # pragma: no cover - environment dependent
    ChartCNN = None  # type: ignore[assignment]
    CNNConfig = None  # type: ignore[assignment]
    _VISION_IMPORT_ERROR = exc
else:
    _VISION_IMPORT_ERROR = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[4]

DEFAULT_FUSION_DATASET: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "aligned"
    / "multimodal_dataset.csv"
)

DEFAULT_NEWS_DATASET: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "raw"
    / "financial_news.csv"
)

DEFAULT_FINBERT_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "finbert"
)

DEFAULT_CNN_CHECKPOINT: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "cnn"
    / "chart_cnn_weighted_best.pt"
)

DEFAULT_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "multimodal"
    / "features"
)


class FeatureFusionError(RuntimeError):
    """Raised when multimodal feature fusion fails."""


@dataclass(frozen=True)
class FeatureFusionConfig:
    """Configuration for leakage-safe multimodal feature fusion."""

    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    market_pca_variance: float = 0.95

    text_batch_size: int = 8
    vision_batch_size: int = 32
    num_workers: int = 0

    text_max_length: int = 256
    image_size: int = 224

    device: str = "auto"

    def __post_init__(self) -> None:
        ratios = (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0.0 for ratio in ratios):
            raise ValueError(
                "All split ratios must be positive."
            )

        if not np.isclose(
            sum(ratios),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Split ratios must sum to 1.0."
            )

        if not 0.0 < self.market_pca_variance <= 1.0:
            raise ValueError(
                "market_pca_variance must be in (0, 1]."
            )

        if (
            self.text_batch_size <= 0
            or self.vision_batch_size <= 0
        ):
            raise ValueError(
                "Batch sizes must be positive."
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        if (
            self.text_max_length <= 0
            or self.image_size <= 0
        ):
            raise ValueError(
                "text_max_length and image_size must be positive."
            )


@dataclass(frozen=True)
class FusedFeatureSet:
    """Numeric multimodal representation for one chronological split."""

    features: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame

    @property
    def rows(self) -> int:
        return int(
            self.features.shape[0]
        )

    @property
    def dimensions(self) -> int:
        return int(
            self.features.shape[1]
        )


@dataclass(frozen=True)
class FusionArtifacts:
    """All persisted artifacts produced by feature fusion."""

    train_features: Path
    validation_features: Path
    test_features: Path

    train_metadata: Path
    validation_metadata: Path
    test_metadata: Path

    preprocessor: Path
    manifest: Path


class _VisionEmbeddingDataset(
    Dataset[Tensor]
):
    """Image-only dataset used to extract CNN representations."""

    def __init__(
        self,
        image_paths: Iterable[Path],
        transform: transforms.Compose,
    ) -> None:
        self.image_paths = tuple(
            image_paths
        )
        self.transform = transform

        if not self.image_paths:
            raise FeatureFusionError(
                "Vision embedding dataset is empty."
            )

        missing = [
            str(path)
            for path in self.image_paths
            if not path.exists()
        ]

        if missing:
            raise FeatureFusionError(
                f"{len(missing)} vision images do not exist. "
                f"Examples: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        path = self.image_paths[index]

        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")

                return self.transform(
                    rgb
                )

        except Exception as exc:
            raise FeatureFusionError(
                f"Could not load chart image: {path}"
            ) from exc


class MarketRepresentation:
    """
    StandardScaler + PCA fitted exclusively on the training split.
    """

    def __init__(
        self,
        explained_variance: float = 0.95,
    ) -> None:
        self.scaler = StandardScaler()

        self.pca = PCA(
            n_components=explained_variance
        )

        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def output_dimensions(self) -> int:
        if not self._fitted:
            raise FeatureFusionError(
                "Market representation is not fitted."
            )

        return int(
            self.pca.n_components_
        )

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> MarketRepresentation:
        values = extract_numeric_features(
            frame,
            MARKET_FEATURE_COLUMNS,
        )

        self.scaler.fit(
            values
        )

        scaled = self.scaler.transform(
            values
        )

        self.pca.fit(
            scaled
        )

        self._fitted = True

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        if not self._fitted:
            raise FeatureFusionError(
                "Market representation must be fitted before transform."
            )

        values = extract_numeric_features(
            frame,
            MARKET_FEATURE_COLUMNS,
        )

        scaled = self.scaler.transform(
            values
        )

        transformed = self.pca.transform(
            scaled
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )


class FinBERTEmbeddingExtractor:
    """
    Extract mean-pooled FinBERT encoder embeddings.

    Classification logits are intentionally not used as the text
    representation.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = 8,
        max_length: int = 256,
        device: str = "auto",
    ) -> None:
        if FinancialSentimentModel is None:
            raise FeatureFusionError(
                "nlp_engine is not importable. "
                "Install/use the NLP service package."
            ) from _NLP_IMPORT_ERROR

        if not model_dir.exists():
            raise FeatureFusionError(
                f"FinBERT model directory does not exist: "
                f"{model_dir}"
            )

        self.batch_size = batch_size
        self.max_length = max_length

        self.device = self._resolve_device(
            device
        )

        self.sentiment_model = (
            FinancialSentimentModel.load(
                model_dir,
                device=str(self.device),
            )
        )

        self.sentiment_model.model.eval()

        self.encoder = (
            self.sentiment_model.model.base_model
        )

        self.encoder.eval()

    @staticmethod
    def _resolve_device(
        device_name: str,
    ) -> torch.device:
        normalized = (
            device_name
            .strip()
            .lower()
        )

        if normalized == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        device = torch.device(
            device_name
        )

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise FeatureFusionError(
                "CUDA was requested but is not available."
            )

        return device

    @property
    def embedding_dimension(self) -> int:
        return int(
            self.encoder.config.hidden_size
        )

    @staticmethod
    def _mean_pool(
        last_hidden_state: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        summed = torch.sum(
            last_hidden_state * mask,
            dim=1,
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        return summed / counts

    @torch.no_grad()
    def transform(
        self,
        texts: list[str],
    ) -> np.ndarray:
        if not texts:
            raise FeatureFusionError(
                "Cannot extract text embeddings from empty text input."
            )

        rows: list[np.ndarray] = []

        for start in range(
            0,
            len(texts),
            self.batch_size,
        ):
            batch_texts = texts[
                start : start + self.batch_size
            ]

            encoded = (
                self.sentiment_model.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            outputs = self.encoder(
                **encoded
            )

            pooled = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            rows.append(
                pooled
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return np.concatenate(
            rows,
            axis=0,
        )

    def transform_articles(
        self,
        news: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract article embeddings and aggregate them by ticker/date.
        """

        required = {
            "timestamp",
            "ticker",
            "headline",
            "text",
        }

        missing = required.difference(
            news.columns
        )

        if missing:
            raise FeatureFusionError(
                "Raw news dataset is missing columns: "
                f"{sorted(missing)}"
            )

        frame = news.copy()

        frame["ticker"] = (
            frame["ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

        if frame["timestamp"].isna().any():
            raise FeatureFusionError(
                "Raw news contains invalid timestamps."
            )

        frame["date"] = (
            frame["timestamp"]
            .dt.normalize()
            .dt.tz_localize(None)
        )

        frame["text_input"] = (
            frame["headline"]
            .fillna("")
            .astype(str)
            .str.strip()
            + " [SEP] "
            + frame["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        frame["text_input"] = (
            frame["text_input"]
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
            .str.lower()
        )

        if frame["text_input"].eq("").any():
            raise FeatureFusionError(
                "Raw news contains empty article text after preprocessing."
            )

        embeddings = self.transform(
            frame["text_input"].tolist()
        )

        embedding_columns = [
            f"text_embedding_{index}"
            for index in range(
                embeddings.shape[1]
            )
        ]

        embedding_frame = pd.DataFrame(
            embeddings,
            columns=embedding_columns,
        )

        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                embedding_frame,
            ],
            axis=1,
        )

        grouped = (
            frame
            .groupby(
                ["ticker", "date"],
                sort=True,
            )[embedding_columns]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "date": "timestamp"
                }
            )
        )

        return grouped


class CNNEmbeddingExtractor:
    """
    Extract the CNN penultimate representation.

    For the current ChartCNN architecture this is 128 dimensions.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "auto",
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        if (
            ChartCNN is None
            or CNNConfig is None
        ):
            raise FeatureFusionError(
                "vision_engine is not importable. "
                "Install/use the Vision service package."
            ) from _VISION_IMPORT_ERROR

        if not checkpoint_path.exists():
            raise FeatureFusionError(
                f"CNN checkpoint does not exist: "
                f"{checkpoint_path}"
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.project_root = project_root

        self.device = self._resolve_device(
            device
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        config_data = checkpoint.get(
            "model_config"
        )

        if config_data is None:
            raise FeatureFusionError(
                "CNN checkpoint is missing model_config."
            )

        model_config = CNNConfig(
            **config_data
        )

        self.model = (
            ChartCNN(
                model_config
            )
            .to(self.device)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.image_size = image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
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

    @staticmethod
    def _resolve_device(
        device_name: str,
    ) -> torch.device:
        normalized = (
            device_name
            .strip()
            .lower()
        )

        if normalized == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        device = torch.device(
            device_name
        )

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise FeatureFusionError(
                "CUDA was requested but is not available."
            )

        return device

    @property
    def embedding_dimension(self) -> int:
        return int(
            self.model.config.base_channels
            * 4
        )

    def _resolve_image_path(
        self,
        value: str,
    ) -> Path:
        path = Path(value)

        if path.is_absolute():
            return path

        return self.project_root / path

    @torch.no_grad()
    def transform(
        self,
        image_paths: list[str],
    ) -> np.ndarray:
        resolved = [
            self._resolve_image_path(path)
            for path in image_paths
        ]

        dataset = _VisionEmbeddingDataset(
            resolved,
            self.image_transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        rows: list[np.ndarray] = []

        for images in loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            x = self.model.features(
                images
            )

            x = self.model.pool(
                x
            )

            embedding = torch.flatten(
                x,
                start_dim=1,
            )

            rows.append(
                embedding
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        if not rows:
            raise FeatureFusionError(
                "CNN embedding extraction produced no rows."
            )

        return np.concatenate(
            rows,
            axis=0,
        )


class FeatureFusionPipeline:
    """
    Build leakage-safe market + FinBERT + CNN representations.
    """

    def __init__(
        self,
        config: FeatureFusionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FeatureFusionConfig()
        )

        self.market = MarketRepresentation(
            self.config.market_pca_variance
        )

    def split(
        self,
        dataset: FusionDataset,
    ) -> FusionSplit:
        return chronological_split(
            dataset,
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
        )

    def fit_market(
        self,
        train_frame: pd.DataFrame,
    ) -> None:
        self.market.fit(
            train_frame
        )

    def transform_market(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        return self.market.transform(
            frame
        )

    @staticmethod
    def assemble(
        market: np.ndarray,
        text: np.ndarray,
        vision: np.ndarray,
    ) -> np.ndarray:
        """Concatenate representations as [market || text || vision]."""

        arrays = (
            market,
            text,
            vision,
        )

        if any(
            array.ndim != 2
            for array in arrays
        ):
            raise FeatureFusionError(
                "All modality representations must be 2-dimensional."
            )

        rows = {
            array.shape[0]
            for array in arrays
        }

        if len(rows) != 1:
            raise FeatureFusionError(
                "All modality representations must contain "
                "the same number of rows."
            )

        fused = np.concatenate(
            arrays,
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        if not np.isfinite(
            fused
        ).all():
            raise FeatureFusionError(
                "Fused representation contains non-finite values."
            )

        return fused


def load_raw_news(
    path: Path = DEFAULT_NEWS_DATASET,
) -> pd.DataFrame:
    """Load raw news used for FinBERT representation extraction."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw news dataset not found: {path}"
        )

    frame = pd.read_csv(
        path
    )

    if frame.empty:
        raise FeatureFusionError(
            f"Raw news dataset is empty: {path}"
        )

    return frame


def build_fused_splits(
    dataset: FusionDataset,
    *,
    news_path: Path = DEFAULT_NEWS_DATASET,
    finbert_dir: Path = DEFAULT_FINBERT_DIR,
    cnn_checkpoint: Path = DEFAULT_CNN_CHECKPOINT,
    config: FeatureFusionConfig | None = None,
) -> tuple[
    FusedFeatureSet,
    FusedFeatureSet,
    FusedFeatureSet,
    dict[str, object],
    FeatureFusionPipeline,
]:
    """
    Extract all three modality representations.

    Market PCA is fitted only on the training period.
    """

    pipeline = FeatureFusionPipeline(
        config
    )

    split = pipeline.split(
        dataset
    )

    pipeline.fit_market(
        split.train
    )

    text_extractor = FinBERTEmbeddingExtractor(
        finbert_dir,
        batch_size=pipeline.config.text_batch_size,
        max_length=pipeline.config.text_max_length,
        device=pipeline.config.device,
    )

    vision_extractor = CNNEmbeddingExtractor(
        cnn_checkpoint,
        image_size=pipeline.config.image_size,
        batch_size=pipeline.config.vision_batch_size,
        num_workers=pipeline.config.num_workers,
        device=pipeline.config.device,
    )

    raw_news = load_raw_news(
        news_path
    )

    text_by_day = (
        text_extractor.transform_articles(
            raw_news
        )
    )

    def extract_split(
        frame: pd.DataFrame,
    ) -> FusedFeatureSet:
        text = text_by_day[
            text_by_day["timestamp"].isin(
                frame["timestamp"].unique()
            )
        ]

        text = text[
            text["ticker"].isin(
                frame["ticker"].unique()
            )
        ]

        aligned = frame.merge(
            text,
            on=[
                "ticker",
                "timestamp",
            ],
            how="left",
            validate="one_to_one",
            sort=False,
        )

        text_columns = [
            column
            for column in aligned.columns
            if column.startswith(
                "text_embedding_"
            )
        ]

        if not text_columns:
            raise FeatureFusionError(
                "No FinBERT embeddings were found."
            )

        if (
            aligned[text_columns]
            .isna()
            .any()
            .any()
        ):
            raise FeatureFusionError(
                "Missing FinBERT embeddings for aligned fusion rows."
            )

        vision_values = (
            vision_extractor.transform(
                aligned[
                    "image_path"
                ]
                .astype(str)
                .tolist()
            )
        )

        vision_columns = [
            f"vision_embedding_{index}"
            for index in range(
                vision_values.shape[1]
            )
        ]

        vision_frame = pd.DataFrame(
            vision_values,
            columns=vision_columns,
        )

        aligned = pd.concat(
            [
                aligned.reset_index(
                    drop=True
                ),
                vision_frame,
            ],
            axis=1,
        )

        text_values = (
            aligned[
                text_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        market_values = (
            pipeline.transform_market(
                aligned
            )
        )

        fused = pipeline.assemble(
            market_values,
            text_values,
            vision_values,
        )

        targets = (
            aligned[
                "target_direction_1d"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        metadata = aligned[
            [
                "ticker",
                "timestamp",
                "sample_id",
                "image_path",
                "future_return",
                "target_return_1d",
                "target_direction_1d",
            ]
        ].copy()

        return FusedFeatureSet(
            features=fused,
            targets=targets,
            metadata=metadata,
        )

    train = extract_split(
        split.train
    )

    validation = extract_split(
        split.validation
    )

    test = extract_split(
        split.test
    )

    manifest = {
        "feature_order": [
            "market_pca",
            "finbert_mean_pooled",
            "cnn_penultimate",
        ],
        "market_input_features": list(
            MARKET_FEATURE_COLUMNS
        ),
        "market_pca_variance": (
            pipeline.config.market_pca_variance
        ),
        "market_pca_dimensions": (
            pipeline.market.output_dimensions
        ),
        "finbert_embedding_dimensions": (
            text_extractor.embedding_dimension
        ),
        "vision_embedding_dimensions": (
            vision_extractor.embedding_dimension
        ),
        "fused_dimensions": train.dimensions,
        "train_rows": train.rows,
        "validation_rows": validation.rows,
        "test_rows": test.rows,
        "target_column": "target_direction_1d",
        "forbidden_inputs": [
            "target_return_1d",
            "target_direction_1d",
            "future_return",
            "vision_label",
            "label",
        ],
    }

    return (
        train,
        validation,
        test,
        manifest,
        pipeline,
    )


def save_fused_splits(
    train: FusedFeatureSet,
    validation: FusedFeatureSet,
    test: FusedFeatureSet,
    manifest: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    market_representation: MarketRepresentation | None = None,
) -> FusionArtifacts:
    """Persist fused arrays, metadata, and preprocessing artifacts."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = FusionArtifacts(
        train_features=(
            output_dir
            / "train_fused_features.npy"
        ),
        validation_features=(
            output_dir
            / "validation_fused_features.npy"
        ),
        test_features=(
            output_dir
            / "test_fused_features.npy"
        ),
        train_metadata=(
            output_dir
            / "train_fused_metadata.csv"
        ),
        validation_metadata=(
            output_dir
            / "validation_fused_metadata.csv"
        ),
        test_metadata=(
            output_dir
            / "test_fused_metadata.csv"
        ),
        preprocessor=(
            output_dir
            / "market_pca_pipeline.joblib"
        ),
        manifest=(
            output_dir
            / "fusion_manifest.json"
        ),
    )

    np.save(
        paths.train_features,
        train.features,
    )

    np.save(
        paths.validation_features,
        validation.features,
    )

    np.save(
        paths.test_features,
        test.features,
    )

    train.metadata.to_csv(
        paths.train_metadata,
        index=False,
    )

    validation.metadata.to_csv(
        paths.validation_metadata,
        index=False,
    )

    test.metadata.to_csv(
        paths.test_metadata,
        index=False,
    )

    if (
        market_representation is None
        or not market_representation.is_fitted
    ):
        raise FeatureFusionError(
            "A fitted market representation is required for persistence."
        )

    joblib.dump(
        market_representation,
        paths.preprocessor,
    )

    import json

    paths.manifest.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return paths