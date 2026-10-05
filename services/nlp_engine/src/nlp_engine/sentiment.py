from __future__ import annotations 

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


MODEL_NAME = "ProsusAI/finbert"

LABEL_TO_ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL: dict[int, str] = {
    value: key
    for key, value in LABEL_TO_ID.items()
}


@dataclass(frozen=True)
class SentimentConfig:
    """Configuration for the FinBERT sentiment model."""

    model_name: str = MODEL_NAME
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(
                "warmup_ratio must be in [0, 1)"
            )


class FinancialNewsDataset(Dataset[dict[str, torch.Tensor]]):
    """
    PyTorch dataset for preprocessed financial news.

    Each item contains tokenized text and its sentiment label.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                "texts and labels must have identical lengths"
            )

        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


@dataclass(frozen=True)
class SentimentPrediction:
    """Sentiment prediction for one article."""

    label: str
    probability_negative: float
    probability_neutral: float
    probability_positive: float

    @property
    def confidence(self) -> float:
        """Return the highest class probability."""

        return max(
            self.probability_negative,
            self.probability_neutral,
            self.probability_positive,
        )


class FinancialSentimentModel:
    """
    FinBERT-based financial sentiment model.

    This model produces text sentiment only.

    It does NOT:
        - predict asset prices
        - predict returns
        - recommend trades
        - execute trades
        - generate financial advice
    """

    def __init__(
        self,
        config: SentimentConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else SentimentConfig()
        )

        self.device = self._resolve_device(
            device
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.config.model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=3,
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID,
            )
        )

        self.model.to(
            self.device
        )

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> torch.device:
        """Resolve the requested or automatically selected device."""

        if requested_device is not None:
            device = torch.device(
                requested_device
            )

            if (
                device.type == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "CUDA was requested but is not available"
                )

            return device

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def tokenize(
        self,
        texts: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenize text for FinBERT."""

        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def predict(
        self,
        texts: list[str],
    ) -> list[SentimentPrediction]:
        """
        Generate sentiment predictions.

        Inference is performed without gradient tracking.
        """

        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        self.model.eval()

        encodings = self.tokenize(
            texts
        )

        encodings = {
            key: value.to(self.device)
            for key, value in encodings.items()
        }

        with torch.no_grad():
            outputs = self.model(
                **encodings
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        probability_array = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

        predictions: list[
            SentimentPrediction
        ] = []

        for row in probability_array:
            predicted_id = int(
                np.argmax(row)
            )

            predictions.append(
                SentimentPrediction(
                    label=ID_TO_LABEL[
                        predicted_id
                    ],
                    probability_negative=float(
                        row[0]
                    ),
                    probability_neutral=float(
                        row[1]
                    ),
                    probability_positive=float(
                        row[2]
                    ),
                )
            )

        return predictions

    def predict_one(
        self,
        text: str,
    ) -> SentimentPrediction:
        """Generate a prediction for one article."""

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        return self.predict(
            [text]
        )[0]

    def save(
        self,
        output_dir: str | Path,
    ) -> Path:
        """Save tokenizer and model."""

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            output_path
        )

        self.tokenizer.save_pretrained(
            output_path
        )

        return output_path

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | None = None,
    ) -> FinancialSentimentModel:
        """Load a previously saved FinBERT model."""

        model_path = Path(
            model_dir
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}"
            )

        instance = cls.__new__(
            cls
        )

        instance.config = SentimentConfig(
            model_name=str(
                model_path
            )
        )

        instance.device = (
            cls._resolve_device(
                device
            )
        )

        instance.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path
            )
        )

        instance.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
        )

        instance.model.to(
            instance.device
        )

        return instance