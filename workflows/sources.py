from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import PDF_CACHE_ROOT, WorkflowState
from .text import compact_text


def parse_document(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body_lines: list[str] = []
    in_metadata = True
    for line in raw.splitlines():
        if in_metadata and ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
            continue
        in_metadata = False
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    sections: dict[str, str] = {}
    current = "body"
    bucket: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(bucket).strip()
            current = line[3:].strip().lower()
            bucket = []
        else:
            bucket.append(line)
    sections[current] = "\n".join(bucket).strip()
    return {"path": str(path), "name": path.name, "metadata": metadata, "body": body, "sections": sections}


def extract_pdf_text(path: Path, max_chars: int = 20000) -> str:
    PDF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_file = PDF_CACHE_ROOT / f"{path.stem}.txt"
    if cache_file.exists() and cache_file.stat().st_mtime >= path.stat().st_mtime:
        cached_text = cache_file.read_text(encoding="utf-8")
        if cached_text.strip():
            return cached_text[:max_chars]

    text = extract_pdf_text_with_pypdf(path)
    if not text:
        text = extract_pdf_text_with_pymupdf(path)
    if not text and platform.system() == "Darwin":
        text = extract_pdf_text_with_pdfkit(path)
    if not text:
        text = extract_pdf_text_with_strings(path)

    compact = re.sub(r"\s+", " ", text).strip()
    cache_file.write_text(compact, encoding="utf-8")
    return compact[:max_chars]


def extract_pdf_text_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def extract_pdf_text_with_pymupdf(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    try:
        with fitz.open(str(path)) as document:
            return "\n".join(page.get_text("text") for page in document)
    except Exception:
        return ""


def extract_pdf_text_with_pdfkit(path: Path) -> str:
    try:
        swift_program = f"""
import Foundation
import PDFKit
let url = URL(fileURLWithPath: "{str(path)}")
if let doc = PDFDocument(url: url) {{
    let text = doc.string ?? ""
    print(text)
}} else {{
    fputs("FAILED_TO_OPEN_PDF", stderr)
    exit(1)
}}
"""
        result = subprocess.run(
            ["zsh", "-lc", "export CLANG_MODULE_CACHE_PATH=/tmp/swift-module-cache && mkdir -p /tmp/swift-module-cache && swift -"],
            input=swift_program,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception:
        return ""


def extract_pdf_text_with_strings(path: Path) -> str:
    try:
        result = subprocess.run(["strings", "-n", "8", str(path)], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception:
        return ""


def load_external_reference_pdfs(root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not root.exists():
        return docs
    for path in sorted(root.glob("*.pdf")):
        body = extract_pdf_text(path)
        docs.append(
            {
                "path": str(path),
                "name": path.name,
                "metadata": {"title": path.stem, "technology": "", "company": "", "sourcetype": "academic_pdf"},
                "body": body,
                "sections": {"overview": body, "body": body},
            }
        )
    return docs


def load_document_by_path(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if path.suffix.lower() == ".pdf":
        docs = load_external_reference_pdfs(path.parent)
        for doc in docs:
            if doc["path"] == str(path):
                return doc
        body = extract_pdf_text(path)
        return {
            "path": str(path),
            "name": path.name,
            "metadata": {"title": path.stem, "technology": "", "company": "", "sourcetype": "academic_pdf", "source": "11-RAG reference", "date": ""},
            "body": body,
            "sections": {"overview": body},
        }
    return parse_document(path)


def load_documents(root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() == ".md":
            docs.append(parse_document(path))
        elif path.suffix.lower() == ".pdf":
            docs.append(load_document_by_path(str(path)))
    return docs


def source_label(path_str: str) -> str:
    path = Path(path_str)
    return f"{path.stem} ({path.name})"


def source_label_from_finding(finding: dict[str, Any]) -> str:
    title = finding.get("title") or finding.get("source")
    if finding.get("source_type") == "web":
        domain = finding.get("domain") or finding.get("publisher") or "web"
        return f"{title} ({domain})"
    return source_label(finding["source"])


def source_label_from_state(source: str, state: WorkflowState) -> str:
    for finding in state.get("web_findings", []):
        if finding.get("source") == source:
            return source_label_from_finding(finding)
    return source_label(source)


def format_reference_entry(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    title = metadata.get("title", doc.get("name", ""))
    source_name = metadata.get("source", metadata.get("publisher", metadata.get("sourcetype", "source")))
    date_value = metadata.get("date", metadata.get("year", "날짜 미상"))
    return f"{title} / {source_name} / {date_value}"


def format_web_reference_entry(finding: dict[str, Any]) -> str:
    title = finding.get("title") or finding.get("source")
    source_name = finding.get("domain") or finding.get("publisher") or "web"
    date_value = finding.get("date") or "날짜 미상"
    if date_value == "날짜 미상" and finding.get("retrieved_at"):
        date_value = f"발행일 미상 (검색일 {finding['retrieved_at']})"
    return f"{title} / {source_name} / {date_value}"


def build_rag_search_text(doc: dict[str, Any]) -> str:
    return compact_text(" ".join([doc["metadata"].get("title", ""), doc["metadata"].get("technology", ""), doc["metadata"].get("company", ""), doc["body"][:6000]]))
