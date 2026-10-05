from __future__ import annotations

from functools import lru_cache

from app.explanation.prompt import GroundedPromptBuilder
from app.explanation.service import (
    GroundedExplanationService,
)
from app.llm.provider import create_llm_provider


@lru_cache(maxsize=1)
def get_prompt_builder() -> GroundedPromptBuilder:
    """
    Return the shared grounded prompt builder.
    """

    return GroundedPromptBuilder()


@lru_cache(maxsize=1)
def get_explanation_service() -> GroundedExplanationService:
    """
    Return the application-wide grounded explanation service.
    """

    return GroundedExplanationService(
        prompt_builder=get_prompt_builder(),
        llm_provider=create_llm_provider(),
    )