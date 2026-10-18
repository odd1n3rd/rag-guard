"""Живая обвязка: Ollama (LLM + эмбеддинги) + Chroma → скомпилированный граф."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from rag_guard.config import Settings
from rag_guard.graph import LLMFn, RetrieverFn, build_graph
from rag_guard.ingest import make_embeddings


def make_llm(settings: Settings) -> LLMFn:
    chat = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_url,
        temperature=settings.temperature,
        seed=settings.seed,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )

    def call(system: str, user: str) -> str:
        return chat.invoke([("system", system), ("human", user)]).content

    return call


def make_retriever(store: Chroma, settings: Settings) -> RetrieverFn:
    def retrieve(question: str) -> list[tuple[Document, float]]:
        return store.similarity_search_with_relevance_scores(question, k=settings.top_k)

    return retrieve


def load_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=make_embeddings(settings),
        persist_directory=str(settings.index_dir),
    )


def load_pipeline(settings: Settings | None = None):
    """Собирает граф на реальных Ollama и Chroma (индекс должен быть построен)."""
    settings = settings or Settings()
    store = load_store(settings)
    return build_graph(
        retriever=make_retriever(store, settings),
        llm=make_llm(settings),
        embeddings=make_embeddings(settings),
        settings=settings,
    )
