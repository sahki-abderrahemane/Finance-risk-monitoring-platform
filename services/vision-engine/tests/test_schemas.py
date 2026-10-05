from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vision_engine.schemas import (
    ChartDataset,
    ChartLabel,
    ChartSample,
)


def make_sample() -> ChartSample:
    return ChartSample(
        sample_id="AAPL_2024-01-10",
        ticker="aapl",
        timestamp=datetime(
            2024,
            1,
            10,
            tzinfo=timezone.utc,
        ),
        image_path="processed/AAPL/sample.png",
        label=ChartLabel.UP,
        future_return=0.025,
        window_size=30,
    )


def test_chart_sample_normalizes_ticker() -> None:
    sample = make_sample()

    assert sample.ticker == "AAPL"


def test_chart_sample_accepts_valid_label() -> None:
    sample = make_sample()

    assert sample.label == ChartLabel.UP


def test_chart_sample_rejects_empty_sample_id() -> None:
    with pytest.raises(ValidationError):
        ChartSample(
            sample_id="",
            ticker="AAPL",
            timestamp=datetime.now(timezone.utc),
            image_path="sample.png",
            label=ChartLabel.UP,
            future_return=0.01,
            window_size=30,
        )


def test_chart_sample_rejects_invalid_window() -> None:
    with pytest.raises(ValidationError):
        ChartSample(
            sample_id="sample",
            ticker="AAPL",
            timestamp=datetime.now(timezone.utc),
            image_path="sample.png",
            label=ChartLabel.UP,
            future_return=0.01,
            window_size=0,
        )


def test_chart_sample_rejects_empty_image_path() -> None:
    with pytest.raises(ValidationError):
        ChartSample(
            sample_id="sample",
            ticker="AAPL",
            timestamp=datetime.now(timezone.utc),
            image_path="",
            label=ChartLabel.UP,
            future_return=0.01,
            window_size=30,
        )


def test_dataset_size() -> None:
    samples = [
        make_sample(),
        ChartSample(
            sample_id="MSFT_2024-01-10",
            ticker="MSFT",
            timestamp=datetime(
                2024,
                1,
                10,
                tzinfo=timezone.utc,
            ),
            image_path="processed/MSFT/sample.png",
            label=ChartLabel.DOWN,
            future_return=-0.03,
            window_size=30,
        ),
    ]

    dataset = ChartDataset(samples=samples)

    assert dataset.size == 2


def test_dataset_tickers() -> None:
    dataset = ChartDataset(
        samples=[
            make_sample(),
            ChartSample(
                sample_id="MSFT_2024-01-10",
                ticker="MSFT",
                timestamp=datetime(
                    2024,
                    1,
                    10,
                    tzinfo=timezone.utc,
                ),
                image_path="processed/MSFT/sample.png",
                label=ChartLabel.DOWN,
                future_return=-0.03,
                window_size=30,
            ),
        ]
    )

    assert dataset.tickers == ["AAPL", "MSFT"]


def test_dataset_by_ticker() -> None:
    dataset = ChartDataset(
        samples=[
            make_sample(),
            ChartSample(
                sample_id="MSFT_2024-01-10",
                ticker="MSFT",
                timestamp=datetime(
                    2024,
                    1,
                    10,
                    tzinfo=timezone.utc,
                ),
                image_path="processed/MSFT/sample.png",
                label=ChartLabel.DOWN,
                future_return=-0.03,
                window_size=30,
            ),
        ]
    )

    assert len(dataset.by_ticker("aapl")) == 1
    assert dataset.by_ticker("AAPL")[0].ticker == "AAPL"