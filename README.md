# Semiconductor Strategy Agent

SK하이닉스 관점에서 HBM4, PIM, CXL 경쟁사 위협을 분석하고 보고서를 생성하는 Supervisor 기반 워크플로우입니다.

## Setup

```bash
cp .env.example .env
uv venv .venv
source .venv/bin/activate
uv sync
```

`.env`에는 최소한 아래 값이 필요합니다.

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

## Run

```bash
uv run semiconductor-strategy
```

또는

```bash
source .venv/bin/activate
python app.py
```

## Outputs

- `outputs/tech_strategy_report.md`
- `outputs/ai-mini_output.pdf`
- `outputs/evidence_traceability.json`
