from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from nlp_engine.preprocessing import (
    FinancialNewsPreprocessor,
)
from nlp_engine.schemas import (
    NewsArticle,
    SentimentLabel,
)
from nlp_engine.sentiment import (
    FinancialNewsDataset,
    FinancialSentimentModel,
    SentimentConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_NEWS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "raw"
    / "financial_news.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "finbert"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "datasets"
    / "news"
    / "evaluation"
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(
    seed: int,
) -> None:
    """Make training as reproducible as practical."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def load_articles() -> list[NewsArticle]:
    """Load and validate the raw news dataset."""

    if not RAW_NEWS_FILE.exists():
        raise FileNotFoundError(
            f"News dataset not found: {RAW_NEWS_FILE}"
        )

    dataframe = pd.read_csv(
        RAW_NEWS_FILE
    )

    required_columns = {
        "timestamp",
        "ticker",
        "headline",
        "text",
        "source",
        "sentiment_label",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "News dataset is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    articles: list[NewsArticle] = []

    for row in dataframe.to_dict(
        orient="records"
    ):
        articles.append(
            NewsArticle(
                timestamp=row["timestamp"],
                ticker=row["ticker"],
                headline=row["headline"],
                text=row["text"],
                source=row["source"],
                sentiment_label=(
                    row["sentiment_label"]
                ),
            )
        )

    if not articles:
        raise ValueError(
            "News dataset is empty"
        )

    return articles


def preprocess_articles(
    articles: list[NewsArticle],
) -> pd.DataFrame:
    """Preprocess articles into a model-ready dataframe."""

    preprocessor = (
        FinancialNewsPreprocessor()
    )

    processed = (
        preprocessor.transform_many(
            articles
        )
    )

    dataframe = pd.DataFrame(
        [
            {
                "timestamp": item.timestamp,
                "ticker": item.ticker,
                "source": item.source,
                "sentiment_label": (
                    item.sentiment_label
                ),
                "text": item.text,
            }
            for item in processed
        ]
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe = dataframe.sort_values(
        [
            "timestamp",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    return dataframe


def chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split articles chronologically.

    No random shuffling is performed before the split.
    """

    if dataframe.empty:
        raise ValueError(
            "Cannot split an empty dataframe"
        )

    if not np.isclose(
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Dataset split ratios must sum to 1"
        )

    total = len(dataframe)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = (
        train_end
        + int(
            total
            * VALIDATION_RATIO
        )
    )

    if train_end <= 0:
        raise ValueError(
            "Training split is empty"
        )

    if validation_end <= train_end:
        raise ValueError(
            "Validation split is empty"
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

    if test.empty:
        raise ValueError(
            "Test split is empty"
        )

    return (
        train,
        validation,
        test,
    )


def labels_to_ids(
    labels: pd.Series,
) -> list[int]:
    """Convert sentiment labels to model class IDs."""

    mapping = {
        SentimentLabel.NEGATIVE.value: 0,
        SentimentLabel.NEUTRAL.value: 1,
        SentimentLabel.POSITIVE.value: 2,
    }

    result: list[int] = []

    for label in labels:
        normalized = str(
            label
        ).strip().lower()

        if normalized not in mapping:
            raise ValueError(
                f"Unknown sentiment label: {label}"
            )

        result.append(
            mapping[normalized]
        )

    return result


def evaluate(
    model: FinancialSentimentModel,
    dataframe: pd.DataFrame,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate the sentiment model."""

    texts = dataframe[
        "text"
    ].tolist()

    true_labels = labels_to_ids(
        dataframe[
            "sentiment_label"
        ]
    )

    predictions = model.predict(
        texts
    )

    predicted_labels = [
        {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }[
            prediction.label
        ]
        for prediction in predictions
    ]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "negative",
            "neutral",
            "positive",
        ],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "classification_report": report,
        "rows": len(dataframe),
        "batch_size": batch_size,
    }


def save_predictions(
    model: FinancialSentimentModel,
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Save model predictions alongside article metadata."""

    predictions = model.predict(
        dataframe[
            "text"
        ].tolist()
    )

    output = dataframe[
        [
            "timestamp",
            "ticker",
            "source",
            "sentiment_label",
            "text",
        ]
    ].copy()

    output[
        "predicted_sentiment"
    ] = [
        prediction.label
        for prediction in predictions
    ]

    output[
        "probability_negative"
    ] = [
        prediction.probability_negative
        for prediction in predictions
    ]

    output[
        "probability_neutral"
    ] = [
        prediction.probability_neutral
        for prediction in predictions
    ]

    output[
        "probability_positive"
    ] = [
        prediction.probability_positive
        for prediction in predictions
    ]

    output[
        "confidence"
    ] = [
        prediction.confidence
        for prediction in predictions
    ]

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_file,
        index=False,
    )


def train(
    model: FinancialSentimentModel,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> None:
    """Fine-tune FinBERT on the training split."""

    train_texts = (
        train_dataframe[
            "text"
        ].tolist()
    )

    train_labels = labels_to_ids(
        train_dataframe[
            "sentiment_label"
        ]
    )

    validation_texts = (
        validation_dataframe[
            "text"
        ].tolist()
    )

    validation_labels = labels_to_ids(
        validation_dataframe[
            "sentiment_label"
        ]
    )

    train_dataset = FinancialNewsDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=model.tokenizer,
        max_length=model.config.max_length,
    )

    validation_dataset = (
        FinancialNewsDataset(
            texts=validation_texts,
            labels=validation_labels,
            tokenizer=model.tokenizer,
            max_length=model.config.max_length,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model.config.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=model.config.batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        model.model.parameters(),
        lr=model.config.learning_rate,
        weight_decay=model.config.weight_decay,
    )

    total_steps = (
        len(train_loader)
        * model.config.epochs
    )

    warmup_steps = int(
        total_steps
        * model.config.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    best_validation_loss = float(
        "inf"
    )

    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(
        model.config.epochs
    ):
        model.model.train()

        train_loss = 0.0

        for batch in train_loader:
            batch = {
                key: value.to(
                    model.device
                )
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model.model(
                **batch
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model did not return a training loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            train_loss += float(
                loss.item()
            )

        validation_loss = 0.0

        model.model.eval()

        with torch.no_grad():
            for batch in validation_loader:
                batch = {
                    key: value.to(
                        model.device
                    )
                    for key, value in batch.items()
                }

                outputs = model.model(
                    **batch
                )

                if outputs.loss is not None:
                    validation_loss += float(
                        outputs.loss.item()
                    )

        average_train_loss = (
            train_loss
            / max(
                len(train_loader),
                1,
            )
        )

        average_validation_loss = (
            validation_loss
            / max(
                len(validation_loader),
                1,
            )
        )

        print(
            f"Epoch {epoch + 1}/"
            f"{model.config.epochs} "
            f"train_loss={average_train_loss:.4f} "
            f"validation_loss="
            f"{average_validation_loss:.4f}"
        )

        if (
            average_validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                average_validation_loss
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.model.state_dict()
                    .items()
                )
            }

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model state"
        )

    model.model.load_state_dict(
        best_state
    )


def main() -> None:
    """Train and evaluate the FinBERT sentiment model."""

    config = SentimentConfig()

    set_seed(
        config.random_state
    )

    articles = load_articles()

    dataframe = preprocess_articles(
        articles
    )

    train_dataframe, validation_dataframe, test_dataframe = (
        chronological_split(
            dataframe
        )
    )

    print(
        f"Total articles: {len(dataframe)}"
    )
    print(
        f"Train articles: {len(train_dataframe)}"
    )
    print(
        f"Validation articles: "
        f"{len(validation_dataframe)}"
    )
    print(
        f"Test articles: {len(test_dataframe)}"
    )

    print(
        f"Device: "
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    model = FinancialSentimentModel(
        config=config
    )

    print(
        "Training FinBERT..."
    )

    train(
        model=model,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
    )

    print(
        "Evaluating test split..."
    )

    test_metrics = evaluate(
        model=model,
        dataframe=test_dataframe,
        batch_size=config.batch_size,
    )

    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        MODEL_DIR
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        PROCESSED_DIR
        / "financial_news_processed.csv",
        index=False,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_predictions(
        model=model,
        dataframe=test_dataframe,
        output_file=(
            RESULTS_DIR
            / "finbert_test_predictions.csv"
        ),
    )

    metrics_file = (
        RESULTS_DIR
        / "finbert_metrics.json"
    )

    with metrics_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            test_metrics,
            handle,
            indent=2,
        )

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(
                    train_dataframe
                ),
                "start": train_dataframe[
                    "timestamp"
                ].min(),
                "end": train_dataframe[
                    "timestamp"
                ].max(),
            },
            {
                "split": "validation",
                "rows": len(
                    validation_dataframe
                ),
                "start": validation_dataframe[
                    "timestamp"
                ].min(),
                "end": validation_dataframe[
                    "timestamp"
                ].max(),
            },
            {
                "split": "test",
                "rows": len(
                    test_dataframe
                ),
                "start": test_dataframe[
                    "timestamp"
                ].min(),
                "end": test_dataframe[
                    "timestamp"
                ].max(),
            },
        ]
    )

    split_summary.to_csv(
        RESULTS_DIR
        / "finbert_split_summary.csv",
        index=False,
    )

    print(
        f"Model saved to: {MODEL_DIR}"
    )

    print(
        "NLP sentiment pipeline completed."
    )


if __name__ == "__main__":
    main()