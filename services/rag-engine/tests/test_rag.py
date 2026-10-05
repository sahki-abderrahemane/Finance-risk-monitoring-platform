from __future__ import annotations

from datetime import date

import pytest
from langchain_core.documents import Document

from app.api.explanation_schemas import (
    ExplanationRequest,
    ExplanationResponse,
    RiskOutputRequest,
)
from app.explanation.evidence import (
    EvidenceChunk,
    EvidenceSet,
)
from app.explanation.pipeline import (
    GroundedExplanationPipeline,
)
from app.explanation.prompt import GroundedPromptBuilder
from app.explanation.risk_output import RiskOutput
from app.explanation.service import (
    ExplanationResult,
    GroundedExplanationService,
)
from app.explanation.validator import (
    GroundednessValidator,
)
from app.evaluation.retrieval import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)
from app.llm.provider import (
    DisabledLLMProvider,
    LLMProvider,
    LLMResponse,
)
from app.retrieval.retriever import (
    RetrievalResult,
    Retriever,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evidence() -> EvidenceSet:
    return EvidenceSet.from_sequence(
        [
            EvidenceChunk(
                content=(
                    "The company reported increased operating "
                    "costs during the reporting period."
                ),
                distance=0.15,
                similarity=0.85,
                document_id="doc-001",
                chunk_id="doc-001:0",
                source="SEC Filing",
                source_type="sec_filing",
                ticker="TEST",
                publication_date=date(2026, 1, 15),
                page=10,
            ),
            EvidenceChunk(
                content=(
                    "Management disclosed higher uncertainty "
                    "around operating expenses."
                ),
                distance=0.25,
                similarity=0.75,
                document_id="doc-001",
                chunk_id="doc-001:1",
                source="SEC Filing",
                source_type="sec_filing",
                ticker="TEST",
                publication_date=date(2026, 1, 15),
                page=11,
            ),
        ]
    )


@pytest.fixture
def risk_output() -> RiskOutput:
    return RiskOutput(
        risk_score=0.72,
        risk_label="high",
        model_name="gradient_boosting",
        model_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Evidence contract
# ---------------------------------------------------------------------------


def test_evidence_set_preserves_provenance(
    evidence: EvidenceSet,
) -> None:
    assert evidence.count == 2
    assert evidence.has_evidence

    first = evidence.items[0]

    assert first.document_id == "doc-001"
    assert first.chunk_id == "doc-001:0"
    assert first.source_type == "sec_filing"
    assert first.ticker == "TEST"
    assert first.page == 10


def test_evidence_citation_label(
    evidence: EvidenceSet,
) -> None:
    label = evidence.items[0].citation_label()

    assert "SEC Filing" in label
    assert "doc-001" in label
    assert "page 10" in label


# ---------------------------------------------------------------------------
# Risk output contract
# ---------------------------------------------------------------------------


def test_risk_output_to_mapping(
    risk_output: RiskOutput,
) -> None:
    mapping = risk_output.to_mapping()

    assert mapping["risk_score"] == 0.72
    assert mapping["risk_label"] == "high"
    assert mapping["model_name"] == "gradient_boosting"
    assert mapping["model_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Grounded prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_risk_output_and_evidence(
    risk_output: RiskOutput,
    evidence: EvidenceSet,
) -> None:
    prompt = GroundedPromptBuilder().build(
        risk_output=risk_output.to_mapping(),
        evidence=evidence,
    )

    assert "0.72" in prompt
    assert "high" in prompt
    assert "operating costs" in prompt
    assert "Evidence 1" in prompt
    assert "Evidence 2" in prompt
    assert "MUST NOT" in prompt


def test_prompt_handles_missing_evidence(
    risk_output: RiskOutput,
) -> None:
    prompt = GroundedPromptBuilder().build(
        risk_output=risk_output.to_mapping(),
        evidence=EvidenceSet(),
    )

    assert "No retrieved evidence is available" in prompt
    assert "insufficient" in prompt.lower()


# ---------------------------------------------------------------------------
# LLM provider boundary
# ---------------------------------------------------------------------------


def test_disabled_provider_rejects_generation() -> None:
    provider = DisabledLLMProvider()

    with pytest.raises(RuntimeError, match="disabled"):
        provider.generate(
            "Explain the supplied risk result."
        )


class FakeLLMProvider(LLMProvider):
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.received_prompt = ""

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        self.received_prompt = prompt

        return LLMResponse(
            content=self.response,
            model="fake-model",
            provider="fake",
        )


# ---------------------------------------------------------------------------
# Grounded explanation service
# ---------------------------------------------------------------------------


def test_explanation_service_generates_result(
    risk_output: RiskOutput,
    evidence: EvidenceSet,
) -> None:
    provider = FakeLLMProvider(
        "The evidence indicates higher operating costs "
        "[Evidence 1]."
    )

    service = GroundedExplanationService(
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=provider,
    )

    result = service.explain(
        risk_output=risk_output,
        evidence=evidence,
    )

    assert isinstance(result, ExplanationResult)
    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.evidence_count == 2
    assert "[Evidence 1]" in result.explanation

    assert "0.72" in provider.received_prompt


def test_explanation_service_rejects_empty_response(
    risk_output: RiskOutput,
    evidence: EvidenceSet,
) -> None:
    provider = FakeLLMProvider("")

    service = GroundedExplanationService(
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=provider,
    )

    with pytest.raises(
        RuntimeError,
        match="empty explanation",
    ):
        service.explain(
            risk_output=risk_output,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Groundedness and safety
# ---------------------------------------------------------------------------


def make_result(
    explanation: str,
) -> ExplanationResult:
    return ExplanationResult(
        explanation=explanation,
        model="fake-model",
        provider="fake",
        evidence_count=1,
        citations=("SEC Filing | doc-001 | page 10",),
    )


def test_validator_accepts_grounded_explanation(
    evidence: EvidenceSet,
) -> None:
    result = make_result(
        "Higher operating costs were reported "
        "[Evidence 1]."
    )

    validation = GroundednessValidator().validate(
        result=result,
        evidence=evidence,
    )

    assert validation.valid
    assert validation.violations == ()


@pytest.mark.parametrize(
    "text",
    [
        "The stock will rise next quarter [Evidence 1].",
        "Investors should buy the asset [Evidence 1].",
        "The target price is 150 [Evidence 1].",
        "The model recommends holding the asset [Evidence 1].",
    ],
)
def test_validator_rejects_advisory_or_predictive_language(
    evidence: EvidenceSet,
    text: str,
) -> None:
    result = make_result(text)

    validation = GroundednessValidator().validate(
        result=result,
        evidence=evidence,
    )

    assert not validation.valid
    assert validation.violations


def test_validator_requires_citation(
    evidence: EvidenceSet,
) -> None:
    result = make_result(
        "Operating costs increased."
    )

    validation = GroundednessValidator().validate(
        result=result,
        evidence=evidence,
    )

    assert not validation.valid
    assert any(
        "citation" in violation.lower()
        for violation in validation.violations
    )


def test_validator_requires_insufficient_evidence_statement() -> None:
    result = ExplanationResult(
        explanation="The risk result is high.",
        model="fake-model",
        provider="fake",
        evidence_count=0,
        citations=(),
    )

    validation = GroundednessValidator().validate(
        result=result,
        evidence=EvidenceSet(),
    )

    assert not validation.valid


def test_validator_accepts_insufficient_evidence_statement() -> None:
    result = ExplanationResult(
        explanation=(
            "The available evidence is insufficient "
            "to explain the computed risk result."
        ),
        model="fake-model",
        provider="fake",
        evidence_count=0,
        citations=(),
    )

    validation = GroundednessValidator().validate(
        result=result,
        evidence=EvidenceSet(),
    )

    assert validation.valid


# ---------------------------------------------------------------------------
# Complete explanation pipeline
# ---------------------------------------------------------------------------


def test_pipeline_returns_validated_explanation(
    risk_output: RiskOutput,
    evidence: EvidenceSet,
) -> None:
    provider = FakeLLMProvider(
        "The evidence indicates higher operating costs "
        "[Evidence 1]."
    )

    service = GroundedExplanationService(
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=provider,
    )

    pipeline = GroundedExplanationPipeline(
        explanation_service=service,
        validator=GroundednessValidator(),
    )

    result = pipeline.explain(
        risk_output=risk_output,
        evidence=evidence,
    )

    assert result.explanation
    assert result.evidence_count == 2


def test_pipeline_rejects_unsafe_explanation(
    risk_output: RiskOutput,
    evidence: EvidenceSet,
) -> None:
    provider = FakeLLMProvider(
        "The stock will rise and investors should buy "
        "[Evidence 1]."
    )

    service = GroundedExplanationService(
        prompt_builder=GroundedPromptBuilder(),
        llm_provider=provider,
    )

    pipeline = GroundedExplanationPipeline(
        explanation_service=service,
        validator=GroundednessValidator(),
    )

    with pytest.raises(
        ValueError,
        match="validation failed",
    ):
        pipeline.explain(
            risk_output=risk_output,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def test_precision_at_k_and_mrr() -> None:
    evaluator = RetrievalEvaluator()

    cases = [
        RetrievalEvaluationCase(
            query="operating costs",
            relevant_chunk_ids=frozenset(
                {"chunk-1", "chunk-3"}
            ),
        ),
        RetrievalEvaluationCase(
            query="uncertainty",
            relevant_chunk_ids=frozenset(
                {"chunk-4"}
            ),
        ),
    ]

    result = evaluator.evaluate(
        cases=cases,
        retrieved_chunk_ids=[
            ["chunk-1", "chunk-2", "chunk-3"],
            ["chunk-2", "chunk-4", "chunk-5"],
        ],
        k=3,
    )

    assert result.query_count == 2
    assert result.precision_at_k == pytest.approx(
        0.5
    )
    assert result.mean_reciprocal_rank == pytest.approx(
        (1.0 + 0.5) / 2
    )


def test_retrieval_evaluator_rejects_mismatched_inputs() -> None:
    evaluator = RetrievalEvaluator()

    case = RetrievalEvaluationCase(
        query="risk",
        relevant_chunk_ids=frozenset({"chunk-1"}),
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            cases=[case],
            retrieved_chunk_ids=[],
            k=5,
        )


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------


def test_explanation_request_contract(
    evidence: EvidenceSet,
) -> None:
    request = ExplanationRequest(
        risk_output=RiskOutputRequest(
            risk_score=0.72,
            risk_label="high",
            model_name="gradient_boosting",
            model_version="1.0.0",
        ),
        evidence=list(evidence.items),
    )

    assert request.risk_output.risk_score == 0.72
    assert len(request.evidence) == 2


def test_explanation_response_contract() -> None:
    response = ExplanationResponse(
        explanation=(
            "Higher operating costs were reported "
            "[Evidence 1]."
        ),
        model="fake-model",
        provider="fake",
        evidence_count=1,
        citations=[
            "SEC Filing | doc-001 | page 10",
        ],
    )

    assert response.evidence_count == 1
    assert len(response.citations) == 1


# ---------------------------------------------------------------------------
# RetrievalResult compatibility
# ---------------------------------------------------------------------------


def test_retrieval_result_similarity_alias() -> None:
    document = Document(
        page_content="Evidence text",
        metadata={
            "document_id": "doc-1",
            "chunk_id": "doc-1:0",
        },
    )

    result = RetrievalResult(
        document=document,
        distance=0.2,
    )

    assert result.distance == 0.2
    assert result.similarity == pytest.approx(0.8)
    assert result.score == pytest.approx(0.8)