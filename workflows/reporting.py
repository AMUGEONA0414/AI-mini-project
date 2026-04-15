from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import (
    A4,
    OUTPUT_ROOT,
    Paragraph,
    ParagraphStyle,
    SimpleDocTemplate,
    Spacer,
    TA_LEFT,
    TRACEABILITY_OUTPUT_PATH,
    TTFont,
    Table,
    TableStyle,
    WorkflowState,
    colors,
    getSampleStyleSheet,
    mm,
    pdfmetrics,
)
from .sources import format_reference_entry, format_web_reference_entry, load_document_by_path, source_label_from_state
from .text import compact_text, confidence_label, sanitize_markdown_for_pdf, threat_label, strip_table_markup


def infer_table_col_widths(rows: list[list[str]], available_width: float) -> list[float]:
    max_columns = max(len(row) for row in rows)
    weights = [0.0] * max_columns
    header = rows[0] if rows else []
    for row in rows:
        padded = row + [""] * (max_columns - len(row))
        for idx, cell in enumerate(padded):
            plain = strip_table_markup(cell)
            effective_len = max(4, min(len(plain), 80))
            if idx == 0:
                effective_len *= 0.8
            elif idx in {1, 3, 4}:
                effective_len *= 0.9
            elif idx == 2:
                effective_len *= 1.25
            elif idx == 5:
                effective_len *= 1.1
            weights[idx] = max(weights[idx], effective_len)
    for idx, cell in enumerate(header):
        plain = strip_table_markup(cell)
        if "주요 차별점 및 특징" in plain:
            weights[idx] *= 1.8
        elif "SK하이닉스 대응 포인트" in plain:
            weights[idx] *= 1.55
        elif "기술 동향" in plain:
            weights[idx] *= 1.35
        elif "삼성전자 위협 수준" in plain or "마이크론 위협 수준" in plain or "삼성전자 위협도" in plain or "마이크론 위협도" in plain:
            weights[idx] *= 1.02
        elif plain == "기술":
            weights[idx] *= 0.82
        elif plain == "회사":
            weights[idx] *= 1.2
        elif plain == "위협 수준":
            weights[idx] *= 1.12
    total = sum(weights) or max_columns
    widths = [available_width * (weight / total) for weight in weights]
    if max_columns == 6:
        min_widths = [14 * mm, 16 * mm, 30 * mm, 18 * mm, 18 * mm, 24 * mm]
        max_widths = [19 * mm, 24 * mm, 68 * mm, 24 * mm, 24 * mm, 52 * mm]
    elif max_columns == 5:
        min_widths = [14 * mm, 17 * mm, 17 * mm, 46 * mm, 38 * mm]
        max_widths = [18 * mm, 22 * mm, 22 * mm, 76 * mm, 62 * mm]
    else:
        base_min = available_width / max_columns * 0.6
        base_max = available_width / max_columns * 1.6
        min_widths = [base_min] * max_columns
        max_widths = [base_max] * max_columns
    widths = [max(min_widths[i], min(widths[i], max_widths[i])) for i in range(max_columns)]
    if max_columns == 5 and header:
        threat_headers = [strip_table_markup(cell) for cell in header]
        if len(threat_headers) >= 3 and any("삼성전자 위협" in threat_headers[i] for i in range(len(threat_headers))) and any("마이크론 위협" in threat_headers[i] for i in range(len(threat_headers))):
            threat_width = (widths[1] + widths[2]) / 2
            widths[1] = threat_width
            widths[2] = threat_width
    remainder = available_width - sum(widths)
    widths[-1] += remainder
    return widths


def _format_table_cell(value: str, esc_func: Any) -> str:
    placeholder = "__CODEX_BR__"
    normalized = value.replace("&lt;br&gt;", placeholder).replace("&lt;br/&gt;", placeholder).replace("<br>", placeholder).replace("<br/>", placeholder).replace("\n", placeholder)
    escaped = esc_func(normalized)
    return escaped.replace(placeholder, "<br/>")


def resolve_pdf_font() -> tuple[str, Path | None]:
    candidates = [
        ("MalgunGothic", Path("C:/Windows/Fonts/malgun.ttf")),
        ("MalgunGothic", Path("C:/Windows/Fonts/malgunbd.ttf")),
        ("AppleGothic", Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")),
        ("AppleSDGothicNeo", Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")),
        ("NotoSansCJK", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")),
        ("NotoSansCJK", Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")),
    ]
    for font_name, font_path in candidates:
        if font_path.exists():
            return font_name, font_path
    return "Helvetica", None


def render_plain_text_pdf(text: str, pdf_path: Path) -> None:
    text = sanitize_markdown_for_pdf(text)
    font_name, font_path = resolve_pdf_font()
    if font_path and font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    stylesheet = getSampleStyleSheet()
    title_style = ParagraphStyle("KTitle", parent=stylesheet["Title"], fontName=font_name, fontSize=18, leading=24, alignment=TA_LEFT, spaceAfter=8)
    heading_style = ParagraphStyle("KHeading", parent=stylesheet["Heading2"], fontName=font_name, fontSize=13, leading=18, textColor=colors.HexColor("#183153"), spaceBefore=8, spaceAfter=6)
    body_style = ParagraphStyle("KBody", parent=stylesheet["BodyText"], fontName=font_name, fontSize=10, leading=15, alignment=TA_LEFT, spaceAfter=4)
    bullet_style = ParagraphStyle("KBullet", parent=body_style, leftIndent=10, bulletIndent=0)
    table_body_style = ParagraphStyle("KTableBody", parent=body_style, fontName=font_name, fontSize=8.3, leading=10.5, wordWrap="CJK", spaceAfter=0)
    table_header_style = ParagraphStyle("KTableHeader", parent=table_body_style, fontName=font_name, fontSize=8.5, leading=10.5)

    def esc(value: str) -> str:
        value = sanitize_markdown_for_pdf(value)
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\t", " ")

    story: list[Any] = []
    lines = text.splitlines()
    table_buffer: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        max_columns = max(len(row) for row in table_buffer)
        normalized_rows: list[list[Any]] = []
        available_width = A4[0] - (18 * mm) - (18 * mm)
        col_widths = infer_table_col_widths(table_buffer, available_width)
        for row_index, row in enumerate(table_buffer):
            padded = row + [""] * (max_columns - len(row))
            style = table_header_style if row_index == 0 else table_body_style
            normalized_rows.append([Paragraph(_format_table_cell(cell, esc), style) for cell in padded])
        table = Table(normalized_rows, colWidths=col_widths, repeatRows=1, splitByRow=1)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9BA9BE")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(table)
        story.append(Spacer(1, 4 * mm))
        table_buffer = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_table()
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("|") and line.endswith("|"):
            parts = [esc(part.strip()) for part in line.strip("|").split("|")]
            if not all(part.startswith("---") for part in parts):
                table_buffer.append(parts)
            continue
        flush_table()
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), title_style))
        elif line.startswith("## "):
            story.append(Paragraph(esc(line[3:]), heading_style))
        elif line.startswith("- "):
            story.append(Paragraph(esc(line[2:]), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(esc(line), body_style))
    flush_table()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="기술 전략 분석 보고서", author="Codex")
    doc.build(story)


def build_technology_snapshot_block(state: WorkflowState) -> str:
    lines: list[str] = []
    for item in state.get("technology_snapshots", []):
        lines.append(f"- {item['technology']}")
        lines.append(f"  - 현재: {item['current']}")
        lines.append(f"  - 차별점: {item['differentiator']}")
        lines.append(f"  - 도전과제: {item['challenge']}")
    return "\n".join(lines)


def build_competitor_analysis_table(state: WorkflowState) -> str:
    rows = []
    for item in state.get("competitor_analysis_rows", []):
        rows.append(f"| {item['technology']} | {item['company'] if item['company'] not in {'Samsung Electronics','Micron'} else ('삼성전자' if item['company']=='Samsung Electronics' else '마이크론')} | {item['technology_trend']} | TRL {item['estimated_trl']} ({confidence_label(item['confidence'])}) | {threat_label(item['threat_level'])} | {item['key_evidence']} |")
    return "\n".join(rows)


def _clean_comparison_feature_text(text: str) -> str:
    cleaned = sanitize_markdown_for_pdf(text)
    cleaned = cleaned.replace("<br/>", " ").replace("<br>", " ")
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
    cleaned = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", cleaned).strip()
    cleaned = compact_text(cleaned)
    for delimiter in [" / ", " · ", " | ", "; "]:
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter, 1)[0].strip()
    if ", " in cleaned:
        cleaned = cleaned.split(", ", 1)[0].strip()
    cleaned = re.sub(r"^(Samsung Unveils|Micron vs\.|Tech News:|Week \d+, \d{4}\s*-\s*|Applied Materials and Micron Partner To)\s*", "", cleaned, flags=re.IGNORECASE).strip()
    replacements = {
        "HBM4E": "AI 특화 HBM4E 공개",
        "CXL Roadmap": "CXL 로드맵·샘플 공개",
        "Applied Materials와 협력해 AI 메모리 혁신 추진": "미국 내 AI 메모리 협력",
        "Applied Materials and Micron Partner": "미국 내 AI 메모리 협력",
        "TrendForce 뉴스 중심 생태계 구축 노력": "패키징·생태계 강화",
        "Micron vs. SK Hynix HBM technology roadmap to 2026": "HBM 로드맵 추격",
    }
    for source, target in replacements.items():
        if source.lower() in cleaned.lower():
            return target
    if len(cleaned) > 28:
        cleaned = cleaned[:28].rstrip() + "..."
    return cleaned or "핵심 신호 재점검"


def build_comparison_table(state: WorkflowState) -> str:
    rows = []
    grouped: dict[str, dict[str, Any]] = {}
    for item in state.get("competitor_analysis_rows", []):
        tech = item["technology"]
        company = "삼성전자" if item["company"] == "Samsung Electronics" else "마이크론" if item["company"] == "Micron" else item["company"]
        grouped.setdefault(tech, {"threats": {}, "features": {}, "response": ""})
        grouped[tech]["threats"][company] = threat_label(item["threat_level"])
        grouped[tech]["features"][company] = _clean_comparison_feature_text(item["technology_trend"])
        if not grouped[tech]["response"]:
            guidance_map = {
                "HBM4": "패키징·수율·고객 검증 모니터링",
                "PIM": "생태계·적용 워크로드 검증",
                "CXL": "표준 대응·플랫폼 통합 준비",
            }
            grouped[tech]["response"] = guidance_map.get(tech, "핵심 경쟁 신호 재점검")
    for tech in state.get("target_technologies", []):
        item = grouped.get(tech, {"threats": {}, "features": {}, "response": "핵심 경쟁 신호 재점검"})
        samsung_threat = item["threats"].get("삼성전자", "중간")
        micron_threat = item["threats"].get("마이크론", "중간")
        samsung_feature = item["features"].get("삼성전자", "핵심 신호 재점검")
        micron_feature = item["features"].get("마이크론", "핵심 신호 재점검")
        feature_block = f"삼성전자: {samsung_feature}<br/>마이크론: {micron_feature}"
        rows.append(f"| {tech} | {samsung_threat} | {micron_threat} | {feature_block} | {item['response']} |")
    return "\n".join(rows)


def build_compact_comparison_guidance(state: WorkflowState) -> str:
    grouped: dict[str, dict[str, Any]] = {}
    for item in state.get("competitor_analysis_rows", []):
        tech = item["technology"]
        grouped.setdefault(tech, {"threats": {}, "features": []})
        company = item["company"] if item["company"] not in {"Samsung Electronics", "Micron"} else ("삼성전자" if item["company"] == "Samsung Electronics" else "마이크론")
        grouped[tech]["threats"][company] = threat_label(item["threat_level"])
        grouped[tech]["features"].append(f"{company}: {_clean_comparison_feature_text(item['technology_trend'])}")
    guidance_map = {"HBM4": "패키징, 수율, 고객 검증 집중 모니터링", "PIM": "생태계 확보와 적용 워크로드 검증", "CXL": "표준 대응과 플랫폼 통합 준비 강화"}
    lines = []
    for tech in state["target_technologies"]:
        item = grouped.get(tech, {"threats": {}, "features": []})
        samsung = item["threats"].get("삼성전자", "중간")
        micron = item["threats"].get("마이크론", "중간")
        feature_lines = item["features"][:2]
        feature_text = "<br/>".join(compact_text(feature) for feature in feature_lines if feature)
        lines.append(
            f"- {tech}: 삼성전자 위협 수준={samsung}, 마이크론 위협 수준={micron}, "
            f"주요 차별점 및 특징='{feature_text}', 대응='{guidance_map.get(tech, '핵심 경쟁 신호 재점검')}'"
        )
    return "\n".join(lines)


def build_strategy_outline(state: WorkflowState) -> str:
    strategy = state.get("strategy_outline", {})
    lines = ["- R&D 우선순위"]
    lines.extend(f"  - {item}" for item in strategy.get("priorities", []))
    lines.append("- 단기 대응 방향")
    lines.extend(f"  - {item}" for item in strategy.get("short_term", []))
    lines.append("- 중기 대응 방향")
    lines.extend(f"  - {item}" for item in strategy.get("mid_term", []))
    return "\n".join(lines)


def build_background_context(state: WorkflowState) -> str:
    focus_points = [
        "AI 가속기 및 데이터센터 수요 확대가 메모리와 인터커넥트 경쟁을 동시에 자극하고 있다.",
        "HBM4, PIM, CXL은 성능 경쟁뿐 아니라 고객 검증, 패키지 통합, 생태계 호환성까지 함께 봐야 하는 영역이다.",
        "SK하이닉스는 경쟁사 대비 양산 준비도와 고객 접점 신호를 동시에 추적해야 한다.",
    ]
    dynamic_points = [f"{tech}: {signals[0]}" for tech, signals in state.get("latest_signals", {}).items() if signals]
    return "\n".join(f"- {item}" for item in focus_points + dynamic_points[:3])


def build_reference_block(state: WorkflowState) -> str:
    entries = []
    web_reference_map = {item["source"]: item for item in state.get("web_findings", []) if item.get("source_type") == "web"}
    for ref in sorted(set(state.get("references", []))):
        if ref in web_reference_map:
            entries.append(f"- {format_web_reference_entry(web_reference_map[ref])}")
        else:
            entries.append(f"- {format_reference_entry(load_document_by_path(ref))}")
    return "\n".join(entries)


def rule_based_review_checks(draft: str) -> list[str]:
    issues: list[str] = []
    for section in ["## SUMMARY", "## 1. 분석 배경", "## 2. 분석 대상 기술 현황", "## 3. 경쟁사 동향 분석", "## 4. 기술별 경쟁사 비교표", "## 5. 전략적 시사점"]:
        if section not in draft:
            issues.append(f"{section} 섹션 누락")
    if "### " in draft:
        issues.append("분석 대상 기술 현황에 금지된 소제목 형식 사용")
    for marker in ["- HBM4", "- PIM", "- CXL", "- 현재:", "- 차별점:", "- 도전과제:"]:
        if marker not in draft:
            issues.append(f"필수 구조 누락: {marker}")
    if "| 회사" not in draft or "| 기술" not in draft:
        issues.append("경쟁사 동향 분석 표 누락")
    if "| 기술 | 회사 |" not in draft:
        issues.append("경쟁사 동향 분석 표 컬럼 순서 오류")
    if "삼성전자" not in draft or "마이크론" not in draft:
        issues.append("경쟁사 언급 누락")
    if "TRL 4~6" not in draft and "추정치" not in draft:
        issues.append("TRL 추정 한계 명시 부족")
    summary_match = re.search(r"## SUMMARY\s+(.+?)(?:\n## |\Z)", draft, flags=re.DOTALL)
    if summary_match:
        summary_text = compact_text(summary_match.group(1))
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=다)\s+", summary_text) if part.strip()]
        sentence_count = len(sentence_parts)
        if sentence_count > 10:
            issues.append("SUMMARY 10문장 초과")
    else:
        issues.append("SUMMARY 본문 누락")
    for marker in ["- R&D 우선순위", "- 단기 대응 방향", "- 중기 대응 방향"]:
        if marker not in draft:
            issues.append(f"전략적 시사점 구조 누락: {marker}")
    return list(dict.fromkeys(issues))
