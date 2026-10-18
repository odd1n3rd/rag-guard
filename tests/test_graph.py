"""Офлайн-тесты графа retrieve → generate → grounding с заглушками."""

from __future__ import annotations

from helpers import HashingEmbeddings, ScriptedLLM, doc, make_retriever

from rag_guard.config import Settings
from rag_guard.graph import REFUSAL, UNGROUNDED, build_graph

SETTINGS = Settings()
EMB = HashingEmbeddings()

CTX = (
    "Prompt injection возникает, когда пользовательский ввод меняет поведение модели "
    "непредусмотренным образом и может обходить защитные ограничения."
)


def run(retriever, llm):
    graph = build_graph(retriever, llm, EMB, SETTINGS)
    return graph.invoke({"question": "Что такое prompt injection?"})


def test_grounded_answer_with_citation_passes():
    retriever = make_retriever([(doc(CTX), 0.82)])
    llm = ScriptedLLM(
        "Prompt injection возникает, когда пользовательский ввод меняет поведение модели "
        "и обходит защитные ограничения [1]."
    )
    state = run(retriever, llm)
    assert state["refused"] is False
    assert state["grounded"] is True
    assert [c["n"] for c in state["citations"]] == [1]
    assert state["citations"][0]["source"] == "llm/LLM01.md"


def test_offtopic_question_refuses_when_relevance_low():
    # оценка ниже порога min_relevance → срабатывает маршрут refuse, LLM не вызывается
    retriever = make_retriever([(doc(CTX), 0.10)])
    llm = ScriptedLLM("этот ответ не должен появиться")
    state = run(retriever, llm)
    assert state["answer"] == REFUSAL
    assert state["refused"] is True
    assert llm.calls == []


def test_ungrounded_answer_is_blocked():
    retriever = make_retriever([(doc(CTX), 0.80)])
    llm = ScriptedLLM("Завтра солнечно, купите акции металлургов и золото [1].")
    state = run(retriever, llm)
    assert state["answer"] == UNGROUNDED
    assert state["refused"] is True
    assert state["grounded"] is False


def test_answer_without_citation_is_blocked():
    retriever = make_retriever([(doc(CTX), 0.80)])
    llm = ScriptedLLM("Prompt injection меняет поведение модели и обходит ограничения.")
    state = run(retriever, llm)
    assert state["refused"] is True
    assert state["reason"] == "нет ссылок на источники"


def test_model_refusal_is_respected():
    retriever = make_retriever([(doc(CTX), 0.80)])
    llm = ScriptedLLM(REFUSAL)
    state = run(retriever, llm)
    assert state["answer"] == REFUSAL
    assert state["refused"] is True
    assert state["grounded"] is True
