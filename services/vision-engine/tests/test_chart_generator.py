from __future__ import annotations

import pandas as pd
import pytest

from vision_engine.chart_generator import (
    ChartGenerationConfig,
    classify_future_return,
)


def test_default_configuration() -> None:
    config = ChartGenerationConfig()

    assert config.window_size == 30
    assert config.future_horizon == 5
    assert config.up_threshold == 0.01
    assert config.down_threshold == -0.01


def test_invalid_window_size() -> None:
    with pytest.raises(ValueError):
        ChartGenerationConfig(
            window_size=0
        )


def test_invalid_future_horizon() -> None:
    with pytest.raises(ValueError):
        ChartGenerationConfig(
            future_horizon=0
        )


def test_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        ChartGenerationConfig(
            up_threshold=-0.01,
            down_threshold=0.01,
        )


def test_classify_up_return() -> None:
    config = ChartGenerationConfig()

    assert classify_future_return(
        0.025,
        config,
    ).value == "UP"


def test_classify_down_return() -> None:
    config = ChartGenerationConfig()

    assert classify_future_return(
        -0.025,
        config,
    ).value == "DOWN"


def test_classify_flat_return() -> None:
    config = ChartGenerationConfig()

    assert classify_future_return(
        0.005,
        config,
    ).value == "FLAT"


def test_classify_positive_boundary() -> None:
    config = ChartGenerationConfig()

    assert classify_future_return(
        0.01,
        config,
    ).value == "UP"


def test_classify_negative_boundary() -> None:
    config = ChartGenerationConfig()

    assert classify_future_return(
        -0.01,
        config,
    ).value == "DOWN"


def test_timestamp_sorting() -> None:
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                ],
                utc=True,
            ),
        }
    )

    sorted_dataframe = dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    assert sorted_dataframe.iloc[0]["timestamp"] == pd.Timestamp(
        "2024-01-01",
        tz="UTC",
    )