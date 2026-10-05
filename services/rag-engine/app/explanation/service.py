from __future__ import annotations

from dataclasses import dataclass

from app.explanation.evidence import EvidenceSet
from app.explanation.prompt import GroundedPromptBuilder
from app.explanation.risk_output import RiskOutput
from app.llm.provider import LLMProvider, LLMResponse


@dataclass(frozen=True)
class ExplanationResult:
    """
    Result produced by the grounded explanation service.
    """

    explanation: str
    model: str
    provider: str
    evidence_count: int
    citations: tuple[str, ...]


class GroundedExplanationService:
    """
    Orchestrates grounded explanation generation.

    The service accepts an already-computed risk output and
    retrieved evidence, builds a constrained prompt, and delegates
    text generation to the configured LLM provider.

    It does not:
        - calculate risk;
        - retrieve evidence;
        - generate predictions;
        - generate investment recommendations.
    """

    def __init__(
        self,
        *,
        prompt_builder: GroundedPromptBuilder,
        llm_provider: LLMProvider,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider

    def explain(
        self,
        *,
        risk_output: RiskOutput,
        evidence: EvidenceSet,
    ) -> ExplanationResult:
        """
        Generate a grounded explanation for an existing risk result.
        """

        prompt = self.prompt_builder.build(
            risk_output=risk_output.to_mapping(),
            evidence=evidence,
        )

        try:
            response = self.llm_provider.generate(
                prompt,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Grounded explanation generation failed."
            ) from exc

        self._validate_response(response)

        return ExplanationResult(
            explanation=response.content.strip(),
            model=response.model,
            provider=response.provider,
            evidence_count=evidence.count,
            citations=tuple(
                evidence.citation_labels()
            ),
        )

    @staticmethod
    def _validate_response(
        response: LLMResponse,
    ) -> None:
        if not response.content.strip():
            raise RuntimeError(
                "LLM provider returned an empty explanation."
            )

        if not response.model.strip():
            raise RuntimeError(
                "LLM provider returned an empty model identifier."
            )

        if not response.provider.strip():
            raise RuntimeError(
                "LLM provider returned an empty provider identifier."
            )