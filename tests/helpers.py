"""Детерминированные заглушки для офлайн-тестов (без Ollama)."""

from __future__ import annotations

import math
import re

from langchain_core.documents import Document

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddings:
    """Мешок слов с хэшированием в фиксированную размерность и L2-нормировкой.

    Детерминированно и, в отличие от случайных векторов, сохраняет смысл: тексты
    с общими словами получают высокую косинусную близость. Этого достаточно,
    чтобы grounding-проверка в тестах вела себя как на настоящих эмбеддингах.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            v[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ScriptedLLM:
    """LLM-заглушка: возвращает заранее заданный ответ (можно callable от user-промпта)."""

    def __init__(self, answer):
        self._answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._answer(user) if callable(self._answer) else self._answer


def make_retriever(pairs: list[tuple[Document, float]]):
    """Статичный ретривер: всегда возвращает переданные пары (документ, оценка)."""

    def retrieve(_question: str) -> list[tuple[Document, float]]:
        return pairs

    return retrieve


def doc(text: str, *, source: str = "llm/LLM01.md", title: str = "LLM01", url: str = "http://x") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "title": title, "url": url, "origin": "official", "section": ""},
    )
