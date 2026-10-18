"""Скачивает открытые документы OWASP и превращает их в markdown-корпус.

Документы обрабатываются строго как данные: markdown читается как текст,
из PDF извлекается только текст (pypdf), ничего не исполняется.

Результат: data/corpus/{llm,agentic}/*.md и data/corpus/manifest.json
(URL источника, лицензия, sha256 — ingest сверяет хэши перед индексацией).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

LLM_BASE = (
    "https://raw.githubusercontent.com/OWASP/"
    "www-project-top-10-for-large-language-model-applications/main/2_0_vulns/"
)
LLM_FILES = [
    "LLM01_PromptInjection",
    "LLM02_SensitiveInformationDisclosure",
    "LLM03_SupplyChain",
    "LLM04_DataModelPoisoning",
    "LLM05_ImproperOutputHandling",
    "LLM06_ExcessiveAgency",
    "LLM07_SystemPromptLeakage",
    "LLM08_VectorAndEmbeddingWeaknesses",
    "LLM09_Misinformation",
    "LLM10_UnboundedConsumption",
]

AGENTIC_URL = "https://genai.owasp.org/download/52117/?tmstv=1765059207"
AGENTIC_PAGE = "https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/"
AGENTIC_TITLES = {
    "ASI01": "Agent Goal Hijack",
    "ASI02": "Tool Misuse and Exploitation",
    "ASI03": "Identity and Privilege Abuse",
    "ASI04": "Agentic Supply Chain Vulnerabilities",
    "ASI05": "Unexpected Code Execution (RCE)",
    "ASI06": "Memory & Context Poisoning",
    "ASI07": "Insecure Inter-Agent Communication",
    "ASI08": "Cascading Failures",
    "ASI09": "Human-Agent Trust Exploitation",
    "ASI10": "Rogue Agents",
}
SUBHEADINGS = {
    "Description",
    "Common Examples of the Vulnerability",
    "Common Examples of Vulnerability",
    "Example Attack Scenarios",
    "Prevention and Mitigation Guidelines",
    "Reference Links",
    "References",
}
LICENSE = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"
UA = {"User-Agent": "rag-guard/0.1 (+portfolio project; docs fetch)"}


def _get(url: str, attempts: int = 4) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - фиксированные https-URL
                return resp.read()
        except (OSError, http.client.HTTPException) as exc:
            if attempt == attempts:
                raise
            print(f"    повтор {attempt}/{attempts - 1}: {exc}")
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_llm(manifest: list[dict]) -> None:
    out = CORPUS / "llm"
    out.mkdir(parents=True, exist_ok=True)
    for name in LLM_FILES:
        text = _get(LLM_BASE + name + ".md").decode("utf-8")
        data = text.strip().encode("utf-8") + b"\n"
        path = out / f"{name}.md"
        path.write_bytes(data)
        manifest.append(
            {
                "path": str(path.relative_to(CORPUS)),
                "title": text.splitlines()[0].lstrip("# ").strip(),
                "source_url": LLM_BASE + name + ".md",
                "license": LICENSE,
                "changes": "без изменений (только обрезаны пробелы по краям)",
                "sha256": _sha256(data),
            }
        )
        print(f"  llm/{name}.md")


_BOILERPLATE = re.compile(r"^\s*(Page \d+|genai\.owasp\.org)\s*$")
_LIST_START = re.compile(r"^(\d+\.|[•●○■▪-])\s")


def _page_lines(reader: PdfReader, i: int) -> list[str]:
    raw = reader.pages[i].extract_text() or ""
    return [ln.rstrip() for ln in raw.splitlines() if not _BOILERPLATE.match(ln)]


def _to_markdown(code: str, lines: list[str]) -> str:
    """Склеивает строки PDF в абзацы и размечает заголовки разделов."""
    body: list[str] = [f"## {code}:2026 {AGENTIC_TITLES[code]}", ""]
    para: list[str] = []

    def flush() -> None:
        if para:
            body.append(re.sub(r"\s+", " ", " ".join(para)).strip())
            body.append("")
            para.clear()

    # пропускаем строки заголовка раздела (он бывает перенесён на 2-3 строки)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "Description")
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            flush()
        elif s in SUBHEADINGS:
            flush()
            body.extend([f"### {s}", ""])
        elif _LIST_START.match(s):
            flush()
            para.append(s)
        else:
            if para and para[-1].endswith("-") and not para[-1].endswith(" -"):
                para[-1] = para[-1][:-1] + s  # перенос слова
            else:
                para.append(s)
    flush()
    return "\n".join(body).strip() + "\n"


def fetch_agentic(manifest: list[dict]) -> None:
    out = CORPUS / "agentic"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-guard-pdf-") as tmp:
        pdf_path = Path(tmp) / "agentic_top10_2026.pdf"
        pdf_bytes = _get(AGENTIC_URL)
        if not pdf_bytes.startswith(b"%PDF"):
            raise RuntimeError("Ожидался PDF, получено что-то другое — прерываю")
        pdf_path.write_bytes(pdf_bytes)
        reader = PdfReader(str(pdf_path))
        pages = [_page_lines(reader, i) for i in range(len(reader.pages))]

    # Начало раздела — страница, где первая строка "ASIxx: ..." и дальше есть "Description".
    starts: dict[str, int] = {}
    end = len(pages)
    for i, lines in enumerate(pages):
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        m = re.match(r"^(ASI\d\d):", first)
        if m and "Description" in (ln.strip() for ln in lines):
            starts.setdefault(m.group(1), i)
        if first.startswith("Appendix A") and starts:
            end = min(end, i)
    missing = set(AGENTIC_TITLES) - set(starts)
    if missing:
        raise RuntimeError(f"Не нашёл разделы {sorted(missing)} — структура PDF изменилась?")

    order = sorted(starts.items(), key=lambda kv: kv[1])
    for idx, (code, first_page) in enumerate(order):
        last_page = order[idx + 1][1] if idx + 1 < len(order) else end
        lines = [ln for p in range(first_page, last_page) for ln in pages[p]]
        md = _to_markdown(code, lines)
        slug = re.sub(r"[^A-Za-z0-9]+", "", AGENTIC_TITLES[code].title())
        path = out / f"{code}_{slug}.md"
        data = md.encode("utf-8")
        path.write_bytes(data)
        manifest.append(
            {
                "path": str(path.relative_to(CORPUS)),
                "title": f"{code}:2026 {AGENTIC_TITLES[code]}",
                "source_url": AGENTIC_PAGE,
                "license": LICENSE,
                "changes": (
                    f"текст извлечён из PDF (стр. {first_page + 1}-{last_page}), "
                    "удалены колонтитулы, строки склеены в абзацы, добавлены markdown-заголовки"
                ),
                "sha256": _sha256(data),
            }
        )
        print(f"  agentic/{path.name}")


def main() -> int:
    manifest: list[dict] = []
    print("OWASP Top 10 for LLM Applications 2025:")
    fetch_llm(manifest)
    print("OWASP Top 10 for Agentic Applications 2026:")
    fetch_agentic(manifest)
    (CORPUS / "manifest.json").write_text(
        json.dumps({"documents": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Готово: {len(manifest)} документов, manifest.json обновлён")
    return 0


if __name__ == "__main__":
    sys.exit(main())
