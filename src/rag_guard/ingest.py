"""Построение векторного индекса (Chroma) по официальному корпусу OWASP."""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag_guard import corpus
from rag_guard.config import Settings


def make_embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_url)


def build_index(settings: Settings | None = None, *, reset: bool = True) -> int:
    """Загружает корпус (с проверкой sha256), нарезает и индексирует в Chroma.

    Возвращает число проиндексированных фрагментов.
    """
    settings = settings or Settings()
    if reset and settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)

    docs = corpus.load_official(settings)
    chunks = corpus.split(docs)

    store = Chroma(
        collection_name=settings.collection,
        embedding_function=make_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )
    store.add_documents(chunks)
    return len(chunks)


def main() -> int:
    settings = Settings()
    print(f"Индексация корпуса из {settings.corpus_dir} …")
    n = build_index(settings)
    print(f"Готово: {n} фрагментов в {settings.index_dir} (модель эмбеддингов: {settings.embed_model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
