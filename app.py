from workflows.semiconductor_strategy_workflow import OUTPUT_ROOT, run_demo


def main() -> int:
    """Run the mini-project workflow and print output artifact paths."""
    result = run_demo()
    print(f"Markdown: {OUTPUT_ROOT / 'tech_strategy_report.md'}")
    print(f"PDF: {result['pdf_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
