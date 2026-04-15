from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


def ensure_reportlab_importable() -> None:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidate_paths = [
        Path.home() / "Library" / "Python" / version / "lib" / "python" / "site-packages",
        Path.home() / ".pyenv" / "versions" / version / "lib" / f"python{version}" / "site-packages",
        Path.home() / ".pyenv" / "versions" / "3.12.12" / "lib" / "python3.12" / "site-packages",
    ]
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)


ensure_reportlab_importable()

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


Technology = Literal["HBM4", "PIM", "CXL"]


class WorkflowState(TypedDict, total=False):
    user_query: str
    focal_company: str
    target_technologies: list[Technology]
    companies: list[str]
    next_step: str
    retrieved_docs: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]
    web_findings: list[dict[str, Any]]
    latest_signals: dict[str, list[str]]
    normalized_evidence: list[dict[str, Any]]
    evidence_table: list[dict[str, Any]]
    trl_estimates: list[dict[str, Any]]
    threat_assessment: dict[str, list[dict[str, Any]]]
    technology_snapshots: list[dict[str, Any]]
    competitor_analysis_rows: list[dict[str, Any]]
    strategy_outline: dict[str, list[str]]
    draft_report: str
    review_feedback: dict[str, Any]
    review_iteration: int
    needs_review: bool
    final_report_markdown: str
    pdf_path: str
    formatting_status: str
    limitations: list[str]
    references: list[str]
    evidence_traceability: dict[str, Any]
    retrieval_metrics: dict[str, Any]
    retrieval_benchmarks: dict[str, Any]
    embedding_benchmarks: dict[str, Any]
    selected_retrieval_strategy: str
    embedding_model: str
    embedding_candidates: list[dict[str, Any]]
    checkpoint_path: str
    run_notes: list[str]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAG_ROOT = DATA_ROOT / "rag"
WEB_ROOT = DATA_ROOT / "web"
RAG_SEARCH_ROOT = DATA_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / ".cache"
PDF_CACHE_ROOT = CACHE_ROOT / "pdf_text"
EMBEDDING_CACHE_ROOT = CACHE_ROOT / "embeddings"
CHECKPOINT_ROOT = CACHE_ROOT / "checkpoints"
EVAL_ROOT = PROJECT_ROOT / "evaluation"
TRACEABILITY_OUTPUT_PATH = OUTPUT_ROOT / "evidence_traceability.json"

DEFAULT_TECHNOLOGIES: list[Technology] = ["HBM4", "PIM", "CXL"]
DEFAULT_COMPANIES = ["Samsung Electronics", "Micron"]
FOCAL_COMPANY = "SK hynix"
MAX_REVIEW_ITERATIONS = 2
RAG_CHUNK_SIZE = 1400
RAG_CHUNK_OVERLAP = 220
RAG_TOP_K = 6
RAG_TOP_N_DOCS = 2
MIN_WEB_TRUST_SCORE = 0.55
EMBEDDING_BATCH_SIZE = 16
WEB_SEARCH_MAX_WORKERS = 4
COMPANY_DISPLAY_NAMES = {
    "Samsung Electronics": "삼성전자",
    "Micron": "마이크론",
    "SK hynix": "SK하이닉스",
}
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_STRATEGIES = ("dense", "lexical", "hybrid")
EMBEDDING_CANDIDATES = [
    {"provider": "openai", "model": "text-embedding-3-small", "status": "active", "reason": "최종 선정된 임베딩 모델"},
]
TAVILY_API_URL = "https://api.tavily.com/search"
RETRIEVAL_EVALSET_PATH = EVAL_ROOT / "retrieval_evalset.json"
TRUSTED_DOMAINS = {
    "samsung.com": 0.92,
    "samsungsemiconductor.com": 0.92,
    "micron.com": 0.92,
    "skhynix.com": 0.92,
    "nvidia.com": 0.88,
    "intel.com": 0.88,
    "amd.com": 0.88,
    "arxiv.org": 0.95,
    "ieee.org": 0.94,
    "isscc.org": 0.93,
    "hotchips.org": 0.93,
    "cxlconsortium.org": 0.95,
    "tomshardware.com": 0.75,
    "anandtech.com": 0.8,
    "servethehome.com": 0.82,
    "theregister.com": 0.72,
}
ENV_CANDIDATES = [PROJECT_ROOT.parent / ".env", PROJECT_ROOT / ".env"]


def log_progress(agent: str, message: str) -> None:
    print(f"[{agent}] {message}", flush=True)


def initialize_state(
    user_query: str,
    target_technologies: list[Technology] | None = None,
    companies: list[str] | None = None,
) -> WorkflowState:
    return WorkflowState(
        user_query=user_query,
        focal_company=FOCAL_COMPANY,
        target_technologies=target_technologies or DEFAULT_TECHNOLOGIES,
        companies=companies or DEFAULT_COMPANIES,
        next_step="rag",
        retrieved_docs=[],
        rag_evidence=[],
        web_findings=[],
        latest_signals={},
        normalized_evidence=[],
        evidence_table=[],
        trl_estimates=[],
        threat_assessment={},
        technology_snapshots=[],
        competitor_analysis_rows=[],
        strategy_outline={},
        draft_report="",
        review_feedback={"passed": False, "issues": []},
        review_iteration=0,
        needs_review=False,
        final_report_markdown="",
        pdf_path="",
        formatting_status="pending",
        limitations=[],
        references=[],
        evidence_traceability={},
        retrieval_metrics={},
        retrieval_benchmarks={},
        embedding_benchmarks={},
        selected_retrieval_strategy="hybrid",
        embedding_model=OPENAI_EMBEDDING_MODEL,
        embedding_candidates=EMBEDDING_CANDIDATES,
        checkpoint_path="",
        run_notes=[],
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def retry_with_backoff(action_name: str, func: Any, *, retries: int = 3, base_delay: float = 1.5) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            log_progress(action_name, f"attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                break
            time.sleep(base_delay * attempt)
    raise RuntimeError(f"{action_name} failed after {retries} attempts") from last_exc


def save_checkpoint(node_name: str, state: dict[str, Any]) -> str:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_ROOT / f"{node_name}.json"
    checkpoint_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(checkpoint_path)
