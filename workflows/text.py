from __future__ import annotations

import re
from datetime import date


def display_company_name(company: str) -> str:
    company_map = {
        "Samsung Electronics": "삼성전자",
        "Micron": "마이크론",
        "SK hynix": "SK하이닉스",
    }
    return company_map.get(company, company)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-\+]+", text.lower())


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_sentence(text: str) -> str:
    cleaned = compact_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def strip_leading_subject(text: str, subjects: list[str]) -> str:
    cleaned = compact_text(text)
    for subject in subjects:
        pattern = rf"^(?:{re.escape(subject)})(?:[는은이가]\s*)?"
        cleaned = re.sub(pattern, "", cleaned).strip()
    return cleaned


def first_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [part.strip() for part in parts if part.strip()]
    return " ".join(cleaned[:limit])


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def sanitize_markdown_for_pdf(text: str) -> str:
    cleaned = strip_markdown_fence(text)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    return cleaned


def strip_table_markup(value: str) -> str:
    plain = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    plain = re.sub(r"\s+", " ", plain)
    return plain.strip()


def normalize_report_structure(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []
    current_section = ""
    strategy_mode = ""
    technology_mode = False
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            strategy_mode = ""
            technology_mode = False
            normalized.append(stripped)
            continue
        if stripped.startswith("### "):
            title = stripped[4:].strip()
            if current_section == "2. 분석 대상 기술 현황":
                technology_mode = True
                normalized.append(f"- {title}")
                continue
            if current_section == "5. 전략적 시사점":
                strategy_mode = title
                normalized.append(f"- {title}")
                continue
            normalized.append(f"## {title}")
            continue
        if current_section == "2. 분석 대상 기술 현황" and re.match(r"^- (HBM4|PIM|PIM \(Processing-In-Memory\)|CXL|CXL \(Compute Express Link\))", stripped):
            technology_mode = True
            normalized.append(stripped)
            continue
        if current_section == "2. 분석 대상 기술 현황" and technology_mode:
            if stripped.startswith("- 현재:") or stripped.startswith("- 차별점:") or stripped.startswith("- 도전과제:"):
                normalized.append(f"  {stripped}")
                continue
            if not stripped:
                technology_mode = False
        if current_section == "5. 전략적 시사점" and strategy_mode and stripped.startswith("- "):
            normalized.append(f"  {stripped}")
            continue
        normalized.append(line)
    return "\n".join(normalized)


def ensure_report_header(text: str) -> str:
    report_title = "# SK하이닉스 관점 반도체 기술 전략 분석 보고서"
    report_date = f"작성일: {date.today().isoformat()}"
    stripped = text.lstrip()
    lines = stripped.splitlines()
    while lines and (lines[0].strip() == report_title or lines[0].strip().startswith("작성일:") or lines[0].strip() == ""):
        lines.pop(0)
    body = "\n".join(lines).strip()
    return f"{report_title}\n{report_date}\n\n{body}".strip()


def sanitize_report_markdown(text: str) -> str:
    cleaned = strip_markdown_fence(text)
    cleaned = re.sub(r"^\s*```markdown\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = normalize_report_structure(cleaned)
    cleaned = ensure_report_header(cleaned)
    return cleaned.strip()


def build_tavily_queries(tech: str, company: str) -> list[tuple[str, str]]:
    return [
        ("progress", f"{company} {tech} semiconductor R&D roadmap prototype sample validation packaging partnership"),
        ("risk", f"{company} {tech} semiconductor bottleneck limitation challenge delay yield interoperability issue"),
    ]


def extract_domain(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url)
    return match.group(1) if match else "web"


def _extract_date_candidate(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"(20\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])",
        r"(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return None


def normalize_search_date(result: dict[str, str], *, url: str = "", content: str = "") -> str:
    for value in [
        result.get("published_date", ""),
        result.get("date", ""),
        result.get("published_at", ""),
        result.get("updated_date", ""),
        result.get("updated_at", ""),
    ]:
        if value:
            return value.split("T", 1)[0]
    for candidate_text in [url, result.get("title", ""), content[:1000]]:
        parsed = _extract_date_candidate(candidate_text)
        if parsed:
            return parsed
    return "날짜 미상"


def summarize_search_result(result: dict[str, str]) -> tuple[str, str]:
    content = compact_text(result.get("raw_content") or result.get("content") or "")
    if not content:
        content = compact_text(result.get("content") or result.get("title") or "")
    summary = first_sentences(content, 2) if content else compact_text(result.get("title", ""))
    return summary, content[:4000]


def confidence_label(value: str) -> str:
    return {"high": "높음", "medium": "중간", "low": "낮음"}.get(value, value)


def threat_label(value: str) -> str:
    return {"high": "높음", "medium": "중간", "low": "낮음"}.get(value, value)
