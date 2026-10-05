from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from multimodal_engine.feature_fusion import (
    FeatureFusionError,
    FeatureFusionPipeline,
    MarketRepresentation,
)


def _frame(
    rows: int = 12,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2024-01-01",
        periods=rows,
        freq="D",
    )

    data = {
        "ticker": ["AAPL"] * rows,
        "timestamp": dates,
        "target_return_1d": np.linspace(
            -0.01,
            0.01,
            rows,
        ),
        "target_direction_1d": [
            0,
            1,
        ]
        * (rows // 2),
    }

    for index, column in enumerate(
        (
            "return_1d",
            "log_return_1d",
            "sma_5",
            "sma_20",
            "ema_5",
            "ema_20",
            "volatility_20",
            "volume_sma_20",
            "volume_ratio",
            "high_low_spread",
            "open_close_spread",
        )
    ):
        data[column] = (
            np.arange(
                rows,
                dtype=float,
            )
            + index
        )

    return pd.DataFrame(
        data
    )


def test_market_representation_fits_and_transforms() -> None:
    frame = _frame()

    representation = MarketRepresentation(
        explained_variance=0.95
    )

    representation.fit(
        frame.iloc[:8]
    )

    transformed = representation.transform(
        frame.iloc[8:]
    )

    assert transformed.shape[0] == 4
    assert transformed.shape[1] <= 11
    assert np.isfinite(
        transformed
    ).all()


def test_market_representation_cannot_transform_before_fit() -> None:
    with pytest.raises(
        FeatureFusionError
    ):
        MarketRepresentation().transform(
            _frame()
        )


def test_assemble_preserves_modality_order_and_dimensions() -> None:
    pipeline = FeatureFusionPipeline()

    market = np.ones(
        (3, 4),
        dtype=np.float32,
    )

    text = np.full(
        (3, 5),
        2.0,
        dtype=np.float32,
    )

    vision = np.full(
        (3, 6),
        3.0,
        dtype=np.float32,
    )

    fused = pipeline.assemble(
        market,
        text,
        vision,
    )

    assert fused.shape == (
        3,
        15,
    )

    assert np.all(
        fused[:, :4] == 1.0
    )

    assert np.all(
        fused[:, 4:9] == 2.0
    )

    assert np.all(
        fused[:, 9:] == 3.0
    )


def test_assemble_rejects_mismatched_rows() -> None:
    pipeline = FeatureFusionPipeline()

    with pytest.raises(
        FeatureFusionError
    ):
        pipeline.assemble(
            np.ones((3, 2)),
            np.ones((4, 2)),
            np.ones((3, 2)),
        )


def test_assemble_rejects_non_finite_values() -> None:
    pipeline = FeatureFusionPipeline()

    with pytest.raises(
        FeatureFusionError
    ):
        pipeline.assemble(
            np.array(
                [[np.nan]],
                dtype=np.float32,
            ),
            np.ones(
                (1, 1),
                dtype=np.float32,
            ),
            np.ones(
                (1, 1),
                dtype=np.float32,
            ),
        )