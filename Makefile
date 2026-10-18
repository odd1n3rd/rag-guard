.PHONY: help setup docs index test test-live lint fmt ask demo clean

help:
	@echo "Цели:"
	@echo "  make setup      — установить зависимости (uv sync)"
	@echo "  make docs       — скачать корпус OWASP в data/corpus (нужен интернет)"
	@echo "  make index      — построить индекс Chroma (нужен Ollama)"
	@echo "  make test       — офлайн-тесты (pytest, без Ollama)"
	@echo "  make test-live  — живой smoke-тест на Ollama (нужен индекс)"
	@echo "  make lint       — проверка ruff"
	@echo "  make ask Q=...  — задать вопрос ассистенту"
	@echo "  make demo       — index + несколько примеров вопросов"

setup:
	uv sync

docs:
	uv run python -I scripts/fetch_docs.py

index:
	uv run python -m rag_guard.ingest

test:
	uv run pytest

test-live:
	uv run pytest -m live

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

ask:
	@uv run python -m rag_guard.cli $(Q)

demo: index
	@uv run python -m rag_guard.cli "Чем прямой prompt injection отличается от косвенного?"
	@echo "----"
	@uv run python -m rag_guard.cli "Какой лучший рецепт борща?"

clean:
	rm -rf .chroma .pytest_cache .ruff_cache
