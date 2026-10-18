"""Тесты загрузки корпуса и проверки целостности (sha256)."""

from __future__ import annotations

import hashlib
import json

import pytest

from rag_guard import corpus
from rag_guard.config import Settings


def test_official_corpus_loads_and_matches_hashes():
    docs = corpus.load_official(Settings())
    assert len(docs) == 20
    assert all(d.metadata["origin"] == "official" for d in docs)
    titles = {d.metadata["title"] for d in docs}
    assert any("Prompt Injection" in t for t in titles)
    assert any(t.startswith("ASI01") for t in titles)


def test_split_prepends_citation_label_and_ids():
    docs = corpus.load_official(Settings())
    chunks = corpus.split(docs)
    assert len(chunks) > len(docs)
    assert all("chunk_id" in c.metadata for c in chunks)
    # заголовок источника вшит в начало фрагмента (это улучшает поиск)
    sample = chunks[0]
    assert sample.page_content.startswith(corpus.label(sample))


def _tmp_corpus(tmp_path, body: bytes, declared_sha: str):
    (tmp_path / "llm").mkdir()
    path = tmp_path / "llm" / "doc.md"
    path.write_bytes(body)
    manifest = {
        "documents": [
            {
                "path": "llm/doc.md",
                "title": "Test doc",
                "source_url": "http://example",
                "license": "CC BY-SA 4.0",
                "sha256": declared_sha,
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return Settings(corpus_dir=tmp_path)


def test_integrity_ok_when_hash_matches(tmp_path):
    body = b"# Test doc\n\ncontent\n"
    settings = _tmp_corpus(tmp_path, body, hashlib.sha256(body).hexdigest())
    docs = corpus.load_official(settings)
    assert docs[0].metadata["title"] == "Test doc"


def test_integrity_fails_when_file_tampered(tmp_path):
    body = b"# Test doc\n\ncontent\n"
    settings = _tmp_corpus(tmp_path, body, hashlib.sha256(body).hexdigest())
    # подменяем файл после фиксации хэша — имитация подмены данных (OWASP LLM04)
    (tmp_path / "llm" / "doc.md").write_bytes(b"# Test doc\n\nTAMPERED\n")
    with pytest.raises(corpus.CorpusIntegrityError):
        corpus.load_official(settings)
