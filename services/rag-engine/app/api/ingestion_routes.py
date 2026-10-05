from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.config import get_settings
from app.ingestion.identity import generate_document_id
from app.ingestion.models import (
    DocumentSourceType,
    FinancialDocumentMetadata,
)
from app.ingestion.pipeline import (
    IngestionPipeline,
    IngestionResult,
)


router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)

ALLOWED_CONTENT_TYPES = {
    ".pdf": {
        "application/pdf",
    },
    ".txt": {
        "text/plain",
    },
}


def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


def _to_response(
    result: IngestionResult,
) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "source": result.source,
        "loaded_documents": result.loaded_documents,
        "generated_chunks": result.generated_chunks,
        "stored_vectors": result.stored_vectors,
        "vector_ids": list(result.vector_ids),
    }


def _validate_upload(
    file: UploadFile,
) -> Path:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    settings = get_settings()

    if suffix not in settings.allowed_document_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported document type. "
                f"Allowed types: "
                f"{list(settings.allowed_document_extensions)}"
            ),
        )

    allowed_types = ALLOWED_CONTENT_TYPES.get(
        suffix,
        set(),
    )

    if (
        file.content_type is not None
        and file.content_type not in allowed_types
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type '{file.content_type}' "
                f"for '{suffix}' document."
            ),
        )

    return Path(suffix)


async def _save_upload(
    file: UploadFile,
    suffix: Path,
) -> tuple[Path, bytes]:
    settings = get_settings()

    max_bytes = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    total_bytes = 0
    content_parts: list[bytes] = []
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name,
            )

            while True:
                chunk = await file.read(
                    1024 * 1024,
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded document exceeds the "
                            f"{settings.max_upload_size_mb} MB "
                            "maximum size."
                        ),
                    )

                temporary_file.write(chunk)
                content_parts.append(chunk)

    except HTTPException:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )
        raise

    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )

        raise HTTPException(
            status_code=500,
            detail="Failed to store uploaded document.",
        ) from exc

    if total_bytes == 0:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )

        raise HTTPException(
            status_code=400,
            detail="Uploaded document is empty.",
        )

    if temporary_path is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to create temporary document.",
        )

    return temporary_path, b"".join(content_parts)


@router.post("")
async def ingest_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    source_type: DocumentSourceType = Form(...),
    ticker: str | None = Form(default=None),
    publication_date: str | None = Form(default=None),
    filing_type: str | None = Form(default=None),
    section: str | None = Form(default=None),
    url: str | None = Form(default=None),
    pipeline: IngestionPipeline = Depends(
        get_ingestion_pipeline,
    ),
) -> dict[str, object]:
    """
    Ingest one financial source document.

    document_id is derived deterministically from the source
    metadata and document content.
    """

    suffix = _validate_upload(file)

    if publication_date is not None:
        try:
            parsed_publication_date = (
                datetime.fromisoformat(
                    publication_date,
                ).date()
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "publication_date must use ISO date format."
                ),
            ) from exc
    else:
        parsed_publication_date = None

    temporary_path, content = await _save_upload(
        file,
        suffix,
    )

    document_id = generate_document_id(
        source=source,
        source_type=source_type.value,
        content=content,
        ticker=ticker,
    )

    try:
        metadata = FinancialDocumentMetadata(
            document_id=document_id,
            source=source,
            source_type=source_type,
            ticker=ticker,
            publication_date=parsed_publication_date,
            filing_type=filing_type,
            section=section,
            url=url,
            ingestion_timestamp=datetime.now(
                timezone.utc,
            ),
        )

        result = pipeline.ingest_file(
            temporary_path,
            metadata=metadata,
        )

        return _to_response(result)

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        temporary_path.unlink(
            missing_ok=True,
        )
        await file.close()