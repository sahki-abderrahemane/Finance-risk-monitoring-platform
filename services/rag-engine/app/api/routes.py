from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    RetrievedEvidence,
    RetrievalRequest,
    RetrievalResponse,
)
from app.dependencies import get_retriever
from app.retrieval.retriever import Retriever


router = APIRouter(
    prefix="/retrieval",
    tags=["retrieval"],
)


@router.post(
    "",
    response_model=RetrievalResponse,
)
def retrieve_evidence(
    request: RetrievalRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrievalResponse:
    """
    Retrieve financially relevant evidence for a query.

    This endpoint performs retrieval only. It does not generate
    predictions, recommendations, or LLM explanations.
    """

    try:
        if (
            request.ticker is not None
            and request.source_type is not None
        ):
            results = retriever.retrieve(
                request.query,
                top_k=request.top_k,
                max_distance=request.max_distance,
                filter={
                    "ticker": request.ticker.strip().upper(),
                    "source_type": request.source_type.strip(),
                },
            )

        elif request.ticker is not None:
            results = retriever.retrieve_for_ticker(
                request.query,
                ticker=request.ticker,
                top_k=request.top_k,
                max_distance=request.max_distance,
            )

        elif request.source_type is not None:
            results = retriever.retrieve_for_source_type(
                request.query,
                source_type=request.source_type,
                top_k=request.top_k,
                max_distance=request.max_distance,
            )

        else:
            results = retriever.retrieve(
                request.query,
                top_k=request.top_k,
                max_distance=request.max_distance,
            )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    evidence = [
        RetrievedEvidence(
            content=result.document.page_content,
            distance=result.distance,
            similarity=result.similarity,
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            source=result.source,
            source_type=result.source_type,
            ticker=result.ticker,
            publication_date=result.publication_date,
            page=result.page,
        )
        for result in results
    ]

    return RetrievalResponse(
        query=request.query.strip(),
        results=evidence,
        result_count=len(evidence),
    )