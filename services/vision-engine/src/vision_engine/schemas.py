from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChartLabel(StrEnum):
    """Supervised target derived from future price movement."""

    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"


class ChartSample(BaseModel):
    """Metadata contract for one historical chart-image sample."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    sample_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    timestamp: datetime
    image_path: str = Field(min_length=1)

    label: ChartLabel

    future_return: float
    window_size: int = Field(gt=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("image_path")
    @classmethod
    def normalize_image_path(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "image_path cannot be empty"
            )

        return value


class ChartDataset(BaseModel):
    """Collection of chart-image samples."""

    model_config = ConfigDict(
        extra="forbid",
    )

    samples: list[ChartSample] = Field(
        min_length=1,
    )

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def tickers(self) -> list[str]:
        return sorted(
            {sample.ticker for sample in self.samples}
        )

    def by_ticker(
        self,
        ticker: str,
    ) -> list[ChartSample]:
        normalized = ticker.strip().upper()

        return [
            sample
            for sample in self.samples
            if sample.ticker == normalized
        ]