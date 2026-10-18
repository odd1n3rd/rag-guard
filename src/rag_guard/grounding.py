"""Проверка обоснованности ответа: опирается ли он на найденные фрагменты.

Это защитный слой поверх генерации (OWASP LLM09 Misinformation): LLM может
выдать правдоподобный, но не подтверждённый источниками текст. Мы разбиваем
ответ на предложения и для каждого считаем максимальную косинусную близость
к эмбеддингам найденных фрагментов. Предложение считается подтверждённым, если
эта близость выше порога; ответ считается обоснованным, если доля подтверждённых
предложений выше порога.

Модуль не зависит от Ollama/LangChain: принимает любой объект эмбеддингов с
методами embed_documents / embed_query, поэтому легко тестируется офлайн.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.UNICODE)
_CITATION_RE = re.compile(r"\[(\d+)\]")
# Короткие служебные строки (маркеры списков, «Источники:») не оцениваем.
_SKIP_RE = re.compile(r"^\s*(источники|sources|references)\s*:?\s*$", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    out = []
    for raw in _SENTENCE_RE.findall(text):
        s = raw.strip(" \t\r\n-•*")
        # отбрасываем голые ссылки на источники вида "[1][2]" и слишком короткие куски
        if len(s) < 12 or _SKIP_RE.match(s) or _CITATION_RE.sub("", s).strip() == "":
            continue
        out.append(s)
    return out


def cited_indices(text: str) -> list[int]:
    seen: list[int] = []
    for m in _CITATION_RE.findall(text):
        n = int(m)
        if n not in seen:
            seen.append(n)
    return seen


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class SentenceScore:
    sentence: str
    support: float


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    support_ratio: float
    sentences: list[SentenceScore]
    unsupported: list[str]

    @property
    def summary(self) -> str:
        supported = len(self.sentences) - len(self.unsupported)
        return (
            f"обоснованность {self.support_ratio:.0%} "
            f"(подтверждено {supported}/{len(self.sentences)} предложений)"
        )


def check_grounding(
    answer: str,
    contexts: list[str],
    embeddings: Embeddings,
    *,
    sentence_support: float,
    min_supported_ratio: float,
) -> GroundingResult:
    """Сравнивает предложения ответа с текстами найденных фрагментов."""
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        # Нечего проверять (например, отказ) — считаем обоснованным, но с нулевой долей.
        return GroundingResult(True, 1.0, [], [])

    ctx_vecs = embeddings.embed_documents(contexts)
    sent_vecs = embeddings.embed_documents(sentences)

    scored: list[SentenceScore] = []
    unsupported: list[str] = []
    for sent, svec in zip(sentences, sent_vecs, strict=True):
        best = max((_cosine(svec, cvec) for cvec in ctx_vecs), default=0.0)
        scored.append(SentenceScore(sent, best))
        if best < sentence_support:
            unsupported.append(sent)

    ratio = (len(scored) - len(unsupported)) / len(scored)
    return GroundingResult(
        grounded=ratio >= min_supported_ratio,
        support_ratio=ratio,
        sentences=scored,
        unsupported=unsupported,
    )
