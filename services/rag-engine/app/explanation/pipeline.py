from __future__ import annotations

from app.explanation.evidence import EvidenceSet
from app.explanation.service import (
    ExplanationResult,
    GroundedExplanationService,
)
from app.explanation.validator import (
    GroundednessValidator,
)


class GroundedExplanationPipeline:
    """
    Generation and validation are deliberately separate stages:

        1. Build grounded prompt.
        2. Generate explanation.
        3. Validate grounding and safety.
        4. Return only validated output.
    """

    def __init__(
        self,
        *,
        explanation_service: GroundedExplanationService,
        validator: GroundednessValidator,
    ) -> None:
        self.explanation_service = explanation_service
        self.validator = validator

    def explain(
        self,
        *,
        risk_output,
        evidence: EvidenceSet,
    ) -> ExplanationResult:
        result = self.explanation_service.explain(
            risk_output=risk_output,
            evidence=evidence,
        )

        self.validator.validate_or_raise(
            result=result,
            evidence=evidence,
        )

        return result