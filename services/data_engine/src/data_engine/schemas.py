from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MarketDataPoint(BaseModel):
    """Validated OHLCV observation."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc_relationships(self) -> "MarketDataPoint":
        if self.high < max(self.open, self.close):
            raise ValueError(
                "high must be greater than or equal to open and close"
            )

        if self.low > min(self.open, self.close):
            raise ValueError(
                "low must be less than or equal to open and close"
            )

        return self