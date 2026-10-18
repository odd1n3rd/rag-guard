"""Тесты модуля grounding на детерминированных эмбеддингах-заглушках."""

from __future__ import annotations

from helpers import HashingEmbeddings

from rag_guard.grounding import check_grounding, cited_indices, split_sentences

EMB = HashingEmbeddings()
THRESHOLDS = {"sentence_support": 0.55, "min_supported_ratio": 0.6}


def test_split_sentences_filters_short_and_citation_only():
    text = "Prompt injection меняет поведение модели [1]. [2]. ок"
    sents = split_sentences(text)
    assert sents == ["Prompt injection меняет поведение модели [1]."]


def test_cited_indices_unique_and_ordered():
    assert cited_indices("факт [2], ещё факт [1], снова [2]") == [2, 1]


def test_answer_from_context_is_grounded():
    context = [
        "Prompt injection возникает, когда пользовательский ввод меняет поведение модели "
        "непредусмотренным образом и обходит защитные ограничения.",
    ]
    answer = (
        "Prompt injection возникает, когда пользовательский ввод меняет поведение модели "
        "и обходит защитные ограничения [1]."
    )
    res = check_grounding(answer, context, EMB, **THRESHOLDS)
    assert res.grounded
    assert res.support_ratio == 1.0
    assert res.unsupported == []


def test_hallucinated_answer_is_not_grounded():
    context = [
        "Prompt injection возникает, когда пользовательский ввод меняет поведение модели.",
    ]
    answer = (
        "Погода завтра солнечная, курс валют растёт, рекомендую купить акции металлургов [1]."
    )
    res = check_grounding(answer, context, EMB, **THRESHOLDS)
    assert not res.grounded
    assert res.support_ratio < 0.6
    assert res.unsupported
