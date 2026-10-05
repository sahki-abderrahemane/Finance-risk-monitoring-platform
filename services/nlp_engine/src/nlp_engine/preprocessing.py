from __future__ import annotations

import re
from dataclasses import dataclass

from nlp_engine.schemas import NewsArticle


_WHITESPACE_PATTERN = re.compile(r"\s+")
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_HTML_PATTERN = re.compile(
    r"<[^>]+>"
)
_NON_PRINTABLE_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration for financial-news text preprocessing.

    The default pipeline intentionally performs conservative
    normalization. BERT tokenizers handle casing, punctuation,
    and linguistic structure themselves, so we avoid aggressive
    transformations such as stop-word removal or stemming.
    """

    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_non_printable: bool = True
    normalize_whitespace: bool = True
    include_headline: bool = True
    headline_separator: str = " [SEP] "


@dataclass(frozen=True)
class PreprocessedArticle:
    """
    Model-ready representation of a financial-news article.

    Original article metadata is retained so model outputs can
    later be aligned with ticker and timestamp.
    """

    timestamp: object
    ticker: str
    source: str
    sentiment_label: str
    text: str

    @property
    def text_length(self) -> int:
        """Return the number of characters in the processed text."""

        return len(self.text)


class FinancialNewsPreprocessor:
    """
    Deterministic preprocessing pipeline for financial news.

    Important:
        This class does not perform tokenization. Tokenization is
        delegated to the BERT tokenizer so that preprocessing and
        model-specific tokenization remain separate concerns.
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else PreprocessingConfig()
        )

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize a text string.

        Processing is deliberately conservative because financial
        language can depend on punctuation and terminology.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        cleaned = text

        if self.config.remove_html:
            cleaned = _HTML_PATTERN.sub(
                " ",
                cleaned,
            )

        if self.config.remove_urls:
            cleaned = _URL_PATTERN.sub(
                " ",
                cleaned,
            )

        if self.config.remove_non_printable:
            cleaned = _NON_PRINTABLE_PATTERN.sub(
                " ",
                cleaned,
            )

        if self.config.normalize_whitespace:
            cleaned = _WHITESPACE_PATTERN.sub(
                " ",
                cleaned,
            )

        cleaned = cleaned.strip()

        if self.config.lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def combine_article_text(
        self,
        article: NewsArticle,
    ) -> str:
        """
        Combine headline and article body.

        The headline is useful financial-sentiment information and
        is therefore included by default.
        """

        if self.config.include_headline:
            headline = self.clean_text(
                article.headline
            )
            body = self.clean_text(
                article.text
            )

            combined = (
                headline
                + self.config.headline_separator
                + body
            )

            return self.clean_text(
                combined
            )

        return self.clean_text(
            article.text
        )

    def transform(
        self,
        article: NewsArticle,
    ) -> PreprocessedArticle:
        """
        Transform one validated NewsArticle.
        """

        if not isinstance(
            article,
            NewsArticle,
        ):
            raise TypeError(
                "article must be a NewsArticle"
            )

        processed_text = (
            self.combine_article_text(
                article
            )
        )

        if not processed_text:
            raise ValueError(
                "Preprocessing produced empty text"
            )

        return PreprocessedArticle(
            timestamp=article.timestamp,
            ticker=article.ticker,
            source=article.source,
            sentiment_label=(
                article.sentiment_label.value
            ),
            text=processed_text,
        )

    def transform_many(
        self,
        articles: list[NewsArticle],
    ) -> list[PreprocessedArticle]:
        """
        Transform multiple validated articles.

        Input ordering is preserved.
        """

        return [
            self.transform(article)
            for article in articles
        ]