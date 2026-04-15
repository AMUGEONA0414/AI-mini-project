from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import GENERATION_EVAL_OUTPUT_PATH, GOLD_REPORT_PATH, OPENAI_EMBEDDING_MODEL
from .retrieval import cosine_similarity, get_embeddings
from .text import compact_text


QUALITY_SECTION_KEYS = ("background", "technology_status", "competitor_trends", "strategic_implications")
SECTION_PATTERNS = {
    "summary": ("summary",),
    "background": ("1.", "analysis background", "분석 배경"),
    "technology_status": ("2.", "technology status", "기술 현황"),
    "competitor_trends": ("3.", "competitor", "경쟁사 동향"),
    "comparison": ("4.", "comparison", "비교"),
    "strategic_implications": ("5.", "strategy", "strategic implications", "전략적 시사점"),
    "limitations": ("limitation", "한계", "주의"),
}


def split_report_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key = "preamble"
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current_key = normalize_section_heading(line[3:])
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(raw_line)
    return {key: compact_text("\n".join(lines)) for key, lines in sections.items() if compact_text("\n".join(lines))}


def normalize_section_heading(heading: str) -> str:
    lowered = heading.lower()
    for key, patterns in SECTION_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return key
    return re.sub(r"[^a-z0-9가-힣]+", "_", lowered).strip("_") or "unknown"


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def cap_text(value: str, max_chars: int = 12000) -> str:
    return compact_text(value)[:max_chars]


def semantic_section_scores(*, report_sections: dict[str, str], gold_sections: dict[str, str], model: str) -> dict[str, Any]:
    comparable_keys = [key for key in QUALITY_SECTION_KEYS if report_sections.get(key) and gold_sections.get(key)]
    texts = [report_sections[key] for key in comparable_keys] + [gold_sections[key] for key in comparable_keys]
    embedding_map = get_embeddings(texts, model=model, input_type="query")
    scores = {}
    for key in comparable_keys:
        score = cosine_similarity(embedding_map[report_sections[key]], embedding_map[gold_sections[key]])
        scores[key] = round(score, 3)
    return {"score": mean(list(scores.values())), "section_scores": scores, "sections_compared": comparable_keys}


def build_evidence_texts(state: dict[str, Any]) -> dict[str, str]:
    rag_text = " ".join(item.get("summary", "") for item in state.get("rag_evidence", []))
    web_text = " ".join(
        " ".join(
            [
                item.get("technology", ""),
                item.get("company", ""),
                item.get("signal_type", ""),
                item.get("title", ""),
                item.get("summary", ""),
                item.get("content", "")[:1200],
            ]
        )
        for item in state.get("web_findings", [])
    )
    trl_text = " ".join(
        f"{item.get('technology', '')} {item.get('company', '')} TRL {item.get('estimated_trl', '')} "
        f"{item.get('confidence', '')} {' '.join(item.get('reasoning', []))}"
        for item in state.get("trl_estimates", [])
    )
    competitor_text = " ".join(
        f"{item.get('technology', '')} {item.get('company', '')} {item.get('technology_trend', '')} "
        f"TRL {item.get('estimated_trl', '')} {item.get('threat_level', '')} {item.get('key_evidence', '')}"
        for item in state.get("competitor_analysis_rows", [])
    )
    strategy = state.get("strategy_outline", {})
    strategy_text = " ".join(
        str(item)
        for bucket in [strategy.get("priorities", []), strategy.get("short_term", []), strategy.get("mid_term", []), state.get("limitations", [])]
        for item in bucket
    )
    return {
        "background": cap_text(f"{rag_text} {web_text}"),
        "technology_status": cap_text(rag_text),
        "competitor_trends": cap_text(f"{web_text} {trl_text} {competitor_text}"),
        "strategic_implications": cap_text(f"{strategy_text} {trl_text} {competitor_text}"),
    }


def evidence_grounding_scores(*, report_sections: dict[str, str], evidence_texts: dict[str, str], model: str) -> dict[str, Any]:
    comparable_keys = [key for key in QUALITY_SECTION_KEYS if report_sections.get(key) and evidence_texts.get(key)]
    texts = [report_sections[key] for key in comparable_keys] + [evidence_texts[key] for key in comparable_keys]
    embedding_map = get_embeddings(texts, model=model, input_type="query")
    scores = {}
    for key in comparable_keys:
        score = cosine_similarity(embedding_map[report_sections[key]], embedding_map[evidence_texts[key]])
        scores[key] = round(score, 3)
    return {"score": mean(list(scores.values())), "section_scores": scores, "sections_compared": comparable_keys}


def quality_criteria_checks(*, report_text: str, report_sections: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    lower_report = report_text.lower()
    references = state.get("references", [])
    web_findings = state.get("web_findings", [])
    rag_evidence = state.get("rag_evidence", [])
    trl_estimates = state.get("trl_estimates", [])
    positive_count = len([item for item in web_findings if item.get("signal_type") == "progress"])
    negative_count = len([item for item in web_findings if item.get("signal_type") == "risk"])
    companies_with_trl = {item.get("company") for item in trl_estimates if item.get("estimated_trl")}
    source_kinds = set()
    if rag_evidence:
        source_kinds.add("local_pdf")
    if any(str(ref).lower().endswith(".pdf") for ref in references):
        source_kinds.add("pdf")
    if web_findings:
        source_kinds.add("web")
    if any("arxiv" in str(ref).lower() or item.get("evidence_type") == "academic_pdf" for ref in references for item in rag_evidence[:1]):
        source_kinds.add("academic_or_reference_pdf")
    if any("news" in item.get("source_kind", "").lower() or item.get("source_type") == "web" for item in web_findings):
        source_kinds.add("public_signal")

    checks = {
        "required_sections_present": {
            "passed": all(report_sections.get(key) for key in QUALITY_SECTION_KEYS),
            "detail": {key: bool(report_sections.get(key)) for key in QUALITY_SECTION_KEYS},
        },
        "trl_assessment_for_samsung_and_micron": {
            "passed": {"Samsung Electronics", "Micron"}.issubset(companies_with_trl)
            and "trl" in lower_report
            and ("samsung" in lower_report or "삼성" in report_text)
            and ("micron" in lower_report or "마이크론" in report_text),
            "detail": sorted(str(item) for item in companies_with_trl),
        },
        "trl_4_to_6_limitations_and_evidence": {
            "passed": bool("trl 4~6" in lower_report and (report_sections.get("limitations") or "limitation" in lower_report or "한계" in report_text)),
            "detail": "Requires explicit TRL 4~6 uncertainty plus evidence/limitation language.",
        },
        "multi_source_balance": {
            "passed": len(source_kinds) >= 3 and len(references) >= 8,
            "detail": {"source_kinds": sorted(source_kinds), "reference_count": len(references)},
        },
        "positive_and_negative_signals_collected": {
            "passed": positive_count > 0 and negative_count > 0,
            "detail": {"progress_signals": positive_count, "risk_signals": negative_count},
        },
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def evaluate_generated_report(state: dict[str, Any], *, model: str = OPENAI_EMBEDDING_MODEL) -> dict[str, Any]:
    report_text = state.get("final_report_markdown", "")
    gold_text = GOLD_REPORT_PATH.read_text(encoding="utf-8")
    report_sections = split_report_sections(report_text)
    gold_sections = split_report_sections(gold_text)
    semscore = semantic_section_scores(report_sections=report_sections, gold_sections=gold_sections, model=model)
    grounding = evidence_grounding_scores(report_sections=report_sections, evidence_texts=build_evidence_texts(state), model=model)
    criteria = quality_criteria_checks(report_text=report_text, report_sections=report_sections, state=state)
    result = {
        "semscore": semscore["score"],
        "semscore_threshold": 0.7,
        "semscore_passed": semscore["score"] >= 0.7,
        "section_semantic_scores": semscore["section_scores"],
        "evidence_grounding_score": grounding["score"],
        "evidence_grounding_threshold": 0.7,
        "evidence_grounding_passed": grounding["score"] >= 0.7,
        "evidence_grounding_note": "Diagnostic score only; overall pass is gated by SemScore and quality criteria.",
        "section_grounding_scores": grounding["section_scores"],
        "quality_criteria": criteria,
        "overall_passed": semscore["score"] >= 0.7 and criteria["passed"],
        "gold_report_path": str(GOLD_REPORT_PATH),
        "model": model,
    }
    return result


def write_generation_eval(result: dict[str, Any], output_path: Path = GENERATION_EVAL_OUTPUT_PATH) -> None:
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
