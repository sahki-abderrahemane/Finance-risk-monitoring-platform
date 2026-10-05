from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-independent response returned by an LLM.
    """

    content: str
    model: str
    provider: str


class LLMProvider(ABC):
   

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        """
        Generate a response from a grounded explanation prompt.
        """
        raise NotImplementedError


class DisabledLLMProvider(LLMProvider):
    """
    Safe provider used when no LLM has been configured.

    This prevents accidental model calls during development and
    makes the provider boundary explicit.
    """

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        if not prompt.strip():
            raise ValueError(
                "LLM prompt cannot be empty."
            )

        raise RuntimeError(
            "LLM generation is disabled. "
            "Configure an LLM provider before generating "
            "grounded explanations."
        )


def create_llm_provider() -> LLMProvider:
    

    settings = get_settings()

    provider = settings.llm_provider.strip().lower()

    if provider in {"", "none", "disabled"}:
        return DisabledLLMProvider()

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}"
    )