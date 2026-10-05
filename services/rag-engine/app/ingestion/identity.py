from __future__ import annotations

import hashlib
from pathlib import Path


def generate_document_id(
    *,
    source: str,
    source_type: str,
    content: bytes,
    ticker: str | None = None,
) -> str:
    """
    Generate a deterministic identifier for a source document.

    The identifier changes when the actual document content changes,
    preventing different document versions from sharing the same
    identity.
    """

    if not source.strip():
        raise ValueError(
            "source cannot be empty."
        )

    if not source_type.strip():
        raise ValueError(
            "source_type cannot be empty."
        )

    if not content:
        raise ValueError(
            "content cannot be empty."
        )

    normalized_source = source.strip().lower()
    normalized_source_type = source_type.strip().lower()
    normalized_ticker = (
        ticker.strip().upper()
        if ticker is not None
        else ""
    )

    content_hash = hashlib.sha256(
        content
    ).hexdigest()

    identity = "|".join(
        [
            normalized_source_type,
            normalized_source,
            normalized_ticker,
            content_hash,
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()