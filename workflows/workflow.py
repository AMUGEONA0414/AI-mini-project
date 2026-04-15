from __future__ import annotations

from .nodes import (
    build_workflow,
    formatting_node,
    run_demo,
    run_workflow_without_langgraph,
)
from .shared import OUTPUT_ROOT, render_plain_text_pdf


if __name__ == "__main__":
    result = run_demo()
    print(f"Markdown: {OUTPUT_ROOT / 'tech_strategy_report.md'}")
    print(f"PDF: {result['pdf_path']}")
