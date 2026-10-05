from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskOutput(BaseModel):
    

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    risk_score: float = Field(
        description="Risk score produced by the risk engine.",
    )

    risk_label: str = Field(
        min_length=1,
        description="Risk classification produced by the risk engine.",
    )

    model_name: str | None = Field(
        default=None,
        min_length=1,
    )

    model_version: str | None = Field(
        default=None,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def to_mapping(self) -> Mapping[str, object]:
        """
        Return the risk result in the representation expected
        by the grounded prompt builder.
        """

        result: dict[str, object] = {
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
        }

        if self.model_name is not None:
            result["model_name"] = self.model_name

        if self.model_version is not None:
            result["model_version"] = self.model_version

        if self.metadata:
            result["metadata"] = self.metadata

        return result