from __future__ import annotations

from typing import Mapping

from app.explanation.evidence import EvidenceSet


class GroundedPromptBuilder:
    
    SYSTEM_INSTRUCTIONS = """
You are the explanation component of Sentinel-AI.

Your task is to explain an already-computed financial risk result
using ONLY the evidence supplied in the prompt.

You MUST:
- explain the provided risk result rather than create a new one;
- ground factual claims in the supplied evidence;
- cite the evidence used for factual statements;
- distinguish supporting evidence from conflicting evidence;
- state when the supplied evidence is insufficient;
- avoid claims that cannot be traced to supplied evidence.

You MUST NOT:
- predict future prices, returns, or market movements;
- generate price targets;
- recommend buying, selling, or holding any asset;
- provide personalized financial advice;
- invent numerical values;
- introduce external facts or knowledge;
- modify or recalculate the supplied risk result.

The risk result is authoritative for this explanation.
The retrieved evidence is authoritative only for claims supported
by its supplied text.

If the evidence does not adequately support an explanation,
explicitly state that the available evidence is insufficient.

This is a research and risk-monitoring system, not an investment
advisory system.
""".strip()

    def build(
        self,
        *,
        risk_output: Mapping[str, object],
        evidence: EvidenceSet,
    ) -> str:
        """
        Build a complete grounded-generation prompt.
        """

        if not risk_output:
            raise ValueError(
                "risk_output cannot be empty."
            )

        risk_section = self._format_risk_output(
            risk_output
        )

        evidence_section = self._format_evidence(
            evidence
        )

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            "=== COMPUTED RISK OUTPUT ===\n"
            f"{risk_section}\n\n"
            "=== RETRIEVED EVIDENCE ===\n"
            f"{evidence_section}\n\n"
            "=== TASK ===\n"
            "Explain how the supplied evidence supports, "
            "contradicts, or does not sufficiently explain "
            "the computed risk output. Use evidence citations "
            "for factual claims. Do not introduce information "
            "that is not present in the supplied evidence.\n"
        )

    @staticmethod
    def _format_risk_output(
        risk_output: Mapping[str, object],
    ) -> str:
        lines = [
            f"- {key}: {value}"
            for key, value in risk_output.items()
        ]

        return "\n".join(lines)

    @staticmethod
    def _format_evidence(
        evidence: EvidenceSet,
    ) -> str:
        if not evidence.has_evidence:
            return (
                "No retrieved evidence is available. "
                "The explanation must explicitly state "
                "that the available evidence is insufficient."
            )

        sections: list[str] = []

        for index, item in enumerate(
            evidence.items,
            start=1,
        ):
            citation = item.citation_label()

            sections.append(
                f"[Evidence {index}]\n"
                f"Citation: {citation}\n"
                f"Source type: {item.source_type}\n"
                f"Ticker: {item.ticker or 'N/A'}\n"
                f"Publication date: "
                f"{item.publication_date or 'N/A'}\n"
                f"Similarity: {item.similarity:.6f}\n"
                f"Content:\n{item.content}"
            )

        return "\n\n".join(sections)