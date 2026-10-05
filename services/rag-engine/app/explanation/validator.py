from __future__ import annotations

import re
from dataclasses import dataclass

from app.explanation.evidence import EvidenceSet
from app.explanation.service import ExplanationResult


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of groundedness and safety validation.
    """

    valid: bool
    violations: tuple[str, ...]

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class GroundednessValidator:
    """
    Validate generated explanations before they leave Sentinel-AI.

    The validator is intentionally conservative. It rejects content
    containing obvious prediction/advisory language and requires
    evidence attribution when evidence was supplied.
    """

    PROHIBITED_PATTERNS = (
        r"\bbuy\b",
        r"\bsell\b",
        r"\bhold(?:ing|s)?\b",
        r"\bprice target\b",
        r"\btarget price\b",
        r"\bwill rise\b",
        r"\bwill fall\b",
        r"\bwill increase\b",
        r"\bwill decrease\b",
        r"\bexpected to rise\b",
        r"\bexpected to fall\b",
        r"\bguaranteed\b",
        r"\bguarantee\b",
        r"\bshould invest\b",
        r"\brecommend(?:s|ation(?:s)?)?\b",
    )

    NUMERIC_PATTERN = re.compile(
        r"(?<![\w.])-?\d+(?:\.\d+)?%?"
    )

    CITATION_PATTERN = re.compile(
        r"\[(?:Evidence\s+\d+|[^\]]+)\]",
        re.IGNORECASE,
    )

    def validate(
        self,
        *,
        result: ExplanationResult,
        evidence: EvidenceSet,
    ) -> ValidationResult:
        violations: list[str] = []

        explanation = result.explanation.strip()

        if not explanation:
            violations.append(
                "Explanation is empty."
            )

        violations.extend(
            self._find_prohibited_language(
                explanation
            )
        )

        if evidence.has_evidence:
            if not self._has_citation(explanation):
                violations.append(
                    "Explanation contains retrieved evidence "
                    "but provides no evidence citation."
                )

        else:
            if not self._contains_insufficient_evidence_statement(
                explanation
            ):
                violations.append(
                    "Explanation has no retrieved evidence but "
                    "does not state that evidence is insufficient."
                )

        return ValidationResult(
            valid=not violations,
            violations=tuple(violations),
        )

    def validate_or_raise(
        self,
        *,
        result: ExplanationResult,
        evidence: EvidenceSet,
    ) -> None:
        validation = self.validate(
            result=result,
            evidence=evidence,
        )

        if not validation.valid:
            raise ValueError(
                "Grounded explanation validation failed: "
                + "; ".join(validation.violations)
            )

    def _find_prohibited_language(
        self,
        explanation: str,
    ) -> list[str]:
        violations: list[str] = []

        normalized = explanation.lower()

        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, normalized):
                violations.append(
                    "Prohibited advisory or predictive "
                    f"language detected: {pattern}"
                )

        return violations

    @classmethod
    def _has_citation(
        cls,
        explanation: str,
    ) -> bool:
        return bool(
            cls.CITATION_PATTERN.search(explanation)
        )

    @staticmethod
    def _contains_insufficient_evidence_statement(
        explanation: str,
    ) -> bool:
        normalized = explanation.lower()

        phrases = (
            "insufficient evidence",
            "evidence is insufficient",
            "not enough evidence",
            "no retrieved evidence",
            "available evidence does not",
        )

        return any(
            phrase in normalized
            for phrase in phrases
        )