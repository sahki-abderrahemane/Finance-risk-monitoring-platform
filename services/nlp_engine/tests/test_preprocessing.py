from __future__ import annotations

from datetime import datetime

import pytest

from nlp_engine.preprocessing import (
    FinancialNewsPreprocessor,
    PreprocessingConfig,
)
from nlp_engine.schemas import (
    NewsArticle,
    SentimentLabel,
)


def make_article(
    *,
    headline: str = "Strong earnings outlook",
    text: str = "The company reported improving demand.",
) -> NewsArticle:
    """Create a valid article for preprocessing tests."""

    return NewsArticle(
        timestamp=datetime(
            2025,
            1,
            2,
            10,
            30,
        ),
        ticker="AAPL",
        headline=headline,
        text=text,
        source="Synthetic Financial Desk",
        sentiment_label=SentimentLabel.POSITIVE,
    )


def test_clean_text_normalizes_whitespace() -> None:
    """Repeated whitespace should be normalized."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "  Strong   earnings\n\n outlook  "
    )

    assert result == "strong earnings outlook"


def test_clean_text_lowercases_by_default() -> None:
    """Default preprocessing should lowercase text."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "Apple REPORTS Strong Growth"
    )

    assert result == "apple reports strong growth"


def test_clean_text_can_preserve_case() -> None:
    """Lowercasing should be configurable."""

    preprocessor = FinancialNewsPreprocessor(
        PreprocessingConfig(
            lowercase=False
        )
    )

    result = preprocessor.clean_text(
        "Apple REPORTS Strong Growth"
    )

    assert result == "Apple REPORTS Strong Growth"


def test_urls_are_removed() -> None:
    """URLs should be removed by default."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "Company update https://example.com/results"
    )

    assert "https://" not in result
    assert result == "company update"


def test_html_is_removed() -> None:
    """HTML tags should be removed."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "<p>Strong quarterly results</p>"
    )

    assert result == "strong quarterly results"


def test_non_printable_characters_are_removed() -> None:
    """Control characters should not remain in processed text."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "Strong\x00 earnings\x07 outlook"
    )

    assert "\x00" not in result
    assert "\x07" not in result
    assert result == "strong earnings outlook"


def test_headline_and_body_are_combined() -> None:
    """Headline and body should be combined."""

    preprocessor = FinancialNewsPreprocessor()

    article = make_article()

    result = preprocessor.combine_article_text(
        article
    )

    assert (
        result
        == "strong earnings outlook [sep] "
        "the company reported improving demand."
    )


def test_headline_can_be_excluded() -> None:
    """Headline inclusion should be configurable."""

    preprocessor = FinancialNewsPreprocessor(
        PreprocessingConfig(
            include_headline=False
        )
    )

    article = make_article()

    result = preprocessor.combine_article_text(
        article
    )

    assert result == (
        "the company reported improving demand."
    )


def test_transform_preserves_metadata() -> None:
    """Transformation should retain alignment metadata."""

    preprocessor = FinancialNewsPreprocessor()

    article = make_article()

    result = preprocessor.transform(
        article
    )

    assert result.timestamp == article.timestamp
    assert result.ticker == "AAPL"
    assert result.source == (
        "Synthetic Financial Desk"
    )
    assert result.sentiment_label == "positive"


def test_transform_produces_model_text() -> None:
    """Transformation should produce non-empty model text."""

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.transform(
        make_article()
    )

    assert result.text
    assert (
        result.text_length
        == len(result.text)
    )


def test_transform_many_preserves_order() -> None:
    """Batch transformation should preserve input ordering."""

    preprocessor = FinancialNewsPreprocessor()

    first = make_article(
        headline="First article"
    )

    second = make_article(
        headline="Second article"
    )

    results = preprocessor.transform_many(
        [
            first,
            second,
        ]
    )

    assert len(results) == 2
    assert results[0].text.startswith(
        "first article"
    )
    assert results[1].text.startswith(
        "second article"
    )


def test_transform_rejects_invalid_input() -> None:
    """Only validated NewsArticle objects are accepted."""

    preprocessor = FinancialNewsPreprocessor()

    with pytest.raises(TypeError):
        preprocessor.transform(
            "not an article"  # type: ignore[arg-type]
        )


def test_clean_text_rejects_non_string() -> None:
    """Text cleaning should reject non-string values."""

    preprocessor = FinancialNewsPreprocessor()

    with pytest.raises(TypeError):
        preprocessor.clean_text(
            123  # type: ignore[arg-type]
        )


def test_clean_text_can_produce_empty_output() -> None:
    """
    clean_text performs normalization only and may produce
    an empty string when its input contains no usable content.
    """

    preprocessor = FinancialNewsPreprocessor()

    result = preprocessor.clean_text(
        "   "
    )

    assert result == ""


def test_transform_rejects_empty_processed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    transform() must reject empty model input.

    We patch the internal combination stage because NewsArticle
    itself correctly rejects whitespace-only headline/body values.
    """

    preprocessor = FinancialNewsPreprocessor()

    article = make_article()

    monkeypatch.setattr(
        preprocessor,
        "combine_article_text",
        lambda _: "",
    )

    with pytest.raises(
        ValueError,
        match="Preprocessing produced empty text",
    ):
        preprocessor.transform(
            article
        )
