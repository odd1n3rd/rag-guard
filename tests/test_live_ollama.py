"""Живой smoke-тест на реальных Ollama + Chroma.

Пропускается, если сервер Ollama недоступен или индекс не построен, поэтому
`make test` остаётся полностью офлайн. Запуск целенаправленно: `make test-live`.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from rag_guard.config import Settings

pytestmark = pytest.mark.live

SETTINGS = Settings()


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(SETTINGS.ollama_url + "/api/version", timeout=3)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError):
        return False


requires_live = pytest.mark.skipif(
    not _ollama_up() or not SETTINGS.index_dir.exists(),
    reason="нужен запущенный Ollama и построенный индекс (make index)",
)


@requires_live
def test_grounded_question_returns_cited_answer():
    from rag_guard.pipeline import load_pipeline

    graph = load_pipeline(SETTINGS)
    state = graph.invoke({"question": "Что такое prompt injection и чем direct отличается от indirect?"})
    assert state["refused"] is False, state.get("reason")
    assert state["citations"], "ожидались ссылки на источники"
    assert state["grounded"] is True


@requires_live
def test_offtopic_question_is_refused():
    from rag_guard.pipeline import load_pipeline

    graph = load_pipeline(SETTINGS)
    state = graph.invoke({"question": "Какой лучший рецепт борща?"})
    assert state["refused"] is True
