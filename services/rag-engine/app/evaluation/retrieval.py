from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalEvaluationCase:
  

    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """
    Aggregate retrieval-quality metrics.
    """

    precision_at_k: float
    mean_reciprocal_rank: float
    query_count: int


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using labeled evidence.

    Metrics:
        Precision@K:
            relevant retrieved chunks / K

        MRR:
            reciprocal rank of the first relevant retrieved chunk.
    """

    def evaluate(
        self,
        *,
        cases: Sequence[RetrievalEvaluationCase],
        retrieved_chunk_ids: Sequence[Sequence[str]],
        k: int,
    ) -> RetrievalEvaluationResult:
        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if len(cases) != len(retrieved_chunk_ids):
            raise ValueError(
                "cases and retrieved_chunk_ids must have "
                "the same length."
            )

        if not cases:
            return RetrievalEvaluationResult(
                precision_at_k=0.0,
                mean_reciprocal_rank=0.0,
                query_count=0,
            )

        precision_scores: list[float] = []
        reciprocal_ranks: list[float] = []

        for case, retrieved in zip(
            cases,
            retrieved_chunk_ids,
        ):
            top_k = list(retrieved[:k])

            relevant_count = sum(
                chunk_id in case.relevant_chunk_ids
                for chunk_id in top_k
            )

            precision_scores.append(
                relevant_count / k
            )

            reciprocal_ranks.append(
                self._reciprocal_rank(
                    retrieved=top_k,
                    relevant_ids=case.relevant_chunk_ids,
                )
            )

        return RetrievalEvaluationResult(
            precision_at_k=(
                sum(precision_scores)
                / len(precision_scores)
            ),
            mean_reciprocal_rank=(
                sum(reciprocal_ranks)
                / len(reciprocal_ranks)
            ),
            query_count=len(cases),
        )

    @staticmethod
    def _reciprocal_rank(
        *,
        retrieved: Sequence[str],
        relevant_ids: frozenset[str],
    ) -> float:
        for rank, chunk_id in enumerate(
            retrieved,
            start=1,
        ):
            if chunk_id in relevant_ids:
                return 1.0 / rank

        return 0.0