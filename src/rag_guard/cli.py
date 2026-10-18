"""CLI: задать вопрос ассистенту (нужен построенный индекс и запущенный Ollama)."""

from __future__ import annotations

import argparse
import sys

from rag_guard.config import Settings
from rag_guard.graph import RagState


def render(state: RagState) -> str:
    lines = [state.get("answer", "").strip(), ""]
    citations = state.get("citations") or []
    if citations:
        lines.append("Источники:")
        for c in citations:
            tail = f" — {c['url']}" if c["url"] else ""
            lines.append(f"  [{c['n']}] {c['label']} ({c['source']}){tail}")
        lines.append("")
    flag = "ОТКАЗ" if state.get("refused") else "OK"
    lines.append(f"[{flag}] {state.get('reason', '')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG-ассистент по OWASP (LLM/Agentic Top 10)")
    parser.add_argument("question", nargs="+", help="вопрос на русском или английском")
    args = parser.parse_args(argv)

    # Импорт здесь, чтобы --help работал без установленных Ollama-зависимостей.
    from rag_guard.pipeline import load_pipeline

    settings = Settings()
    if not settings.index_dir.exists():
        print("Индекс не найден. Сначала выполните: make index", file=sys.stderr)
        return 2

    graph = load_pipeline(settings)
    state = graph.invoke({"question": " ".join(args.question)})
    print(render(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
