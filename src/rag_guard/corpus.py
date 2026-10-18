"""Загрузка корпуса: проверка целостности, разметка источников, нарезка на фрагменты."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag_guard.config import Settings

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


class CorpusIntegrityError(RuntimeError):
    """Файл официального корпуса не совпадает с manifest.json (LLM04: подмена данных)."""


def load_official(settings: Settings) -> list[Document]:
    manifest = json.loads((settings.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = []
    for entry in manifest["documents"]:
        path = settings.corpus_dir / entry["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise CorpusIntegrityError(f"sha256 не совпадает для {entry['path']}")
        docs.append(
            Document(
                page_content=data.decode("utf-8"),
                metadata={
                    "source": entry["path"],
                    "title": entry["title"],
                    "origin": "official",
                    "url": entry["source_url"],
                },
            )
        )
    return docs


def split(docs: list[Document], chunk_size: int = 1000, overlap: int = 150) -> list[Document]:
    md = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)
    rc = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Document] = []
    for doc in docs:
        for part in md.split_text(doc.page_content):
            section = " › ".join(part.metadata[h] for _, h in HEADERS[2:] if h in part.metadata)
            for i, piece in enumerate(rc.split_text(part.page_content)):
                meta = {**doc.metadata, "section": section, "part": i}
                chunk = Document(page_content=piece, metadata=meta)
                # Заголовок документа в тексте фрагмента заметно улучшает поиск.
                chunk.page_content = f"{label(chunk)}\n{piece}"
                chunks.append(chunk)
    for n, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{n}"
    return chunks


def label(doc: Document) -> str:
    """Человекочитаемая подпись источника для цитаты."""
    section = doc.metadata.get("section")
    title = doc.metadata.get("title", doc.metadata.get("source", "?"))
    return f"{title} › {section}" if section else title


def official_paths(settings: Settings) -> list[Path]:
    return sorted(settings.corpus_dir.glob("*/*.md"))
