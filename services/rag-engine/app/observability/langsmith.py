from __future__ import annotations

import os

from app.config import get_settings


def configure_langsmith() -> bool:
   

    settings = get_settings()

    if not settings.langsmith_tracing:
        return False

    if not settings.langsmith_api_key:
        raise RuntimeError(
            "LangSmith tracing is enabled but "
            "LANGSMITH_API_KEY is not configured."
        )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = (
        settings.langsmith_api_key
    )
    os.environ["LANGSMITH_ENDPOINT"] = (
        settings.langsmith_endpoint
    )
    os.environ["LANGSMITH_PROJECT"] = (
        settings.langsmith_project
    )

    return True


def langsmith_status() -> dict[str, object]:
    
    settings = get_settings()

    return {
        "enabled": settings.langsmith_tracing,
        "configured": bool(
            settings.langsmith_api_key
        ),
        "project": settings.langsmith_project,
        "endpoint": settings.langsmith_endpoint,
    }