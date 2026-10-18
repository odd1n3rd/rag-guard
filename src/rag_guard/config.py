"""Настройки проекта: всё переопределяется переменными окружения RAG_GUARD_*."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    llm_model: str = field(default_factory=lambda: _env("RAG_GUARD_LLM", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: _env("RAG_GUARD_EMBED", "bge-m3"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    top_k: int = field(default_factory=lambda: int(_env("RAG_GUARD_TOP_K", "4")))
    # Детерминизм генерации: greedy-декодирование и фиксированный seed.
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 400

    corpus_dir: Path = ROOT / "data" / "corpus"
    index_dir: Path = field(
        default_factory=lambda: Path(_env("RAG_GUARD_INDEX", str(ROOT / ".chroma")))
    )
    reports_dir: Path = ROOT / "reports"
    collection: str = "owasp_kb"

    # Пороги проверки обоснованности ответа (см. grounding.py, калибровка — в README).
    min_relevance: float = 0.45  # ниже — считаем, что в базе нет ничего по теме
    sentence_support: float = 0.55  # косинус предложение↔фрагмент, чтобы считать его подтверждённым
    min_supported_ratio: float = 0.6  # доля подтверждённых предложений в ответе
