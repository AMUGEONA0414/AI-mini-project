# Subject
SK hynix 관점에서 HBM4, PIM, CXL 관련 공개 자료와 최신 웹 신호를 수집하고, 경쟁사 기술 성숙도와 위협 수준을 비교해 기술 전략 분석 보고서를 생성하는 Agentic Workflow 프로젝트입니다.

## Overview
- Objective : HBM4, PIM, CXL 분야의 경쟁사 R&D 동향을 정리하고 SK하이닉스 관점의 대응 포인트를 도출합니다.
- Method : Supervisor 기반 멀티 에이전트 워크플로우로 RAG, Web Search, TRL 추정, 위협 분석, 보고서 생성 단계를 순차적으로 수행합니다.
- Tools : OpenAI API, Tavily API, LangGraph, Python, ReportLab

## Features
- PDF 자료 기반 정보 추출 및 RAG 검색
- Tavily 기반 실시간 웹 검색과 공개 신호 수집
- TRL 기반 기술 성숙도 추정 및 경쟁사 위협 수준 분석
- Retrieval 성능 평가 지원: Hit Rate@K, MRR
- 한국어 기술 전략 분석 보고서 Markdown/PDF 생성
- 확증 편향 방지 전략 : 경쟁사별로 progress/risk 쿼리를 대칭적으로 수행하고, 저신뢰 도메인과 짧은 본문은 필터링합니다.

## Tech Stack

| Category | Details |
|---|---|
| Framework | LangGraph, Python |
| LLM | gpt-4.1-mini via OpenAI API |
| Web Search | Tavily Search API |
| Retrieval | Custom RAG pipeline with Hit Rate@K, MRR evaluation |
| Embedding | text-embedding-3-small |
| PDF | ReportLab |

## Agents

- Supervisor: 전체 상태를 보고 다음 노드를 선택하고 종료 조건을 통제합니다.
- RAG Agent: 로컬 PDF 자료를 chunking/embedding 기반으로 검색합니다.
- Web Search Agent: Tavily로 경쟁사별 최신 공개 신호를 수집합니다.
- TRL Assessment Node: 공개 자료와 간접 신호를 기준으로 기술 성숙도를 추정합니다.
- Threat Strategy Node: 경쟁사 위협 수준과 SK하이닉스 대응 포인트를 도출합니다.
- Draft / Review / Final / Formatting Nodes: 보고서 초안 작성, 품질 점검, 최종 편집, PDF 렌더링을 수행합니다.

## Architecture
(그래프 이미지)

## Directory Structure
```text
├── data/                  # PDF 문서 및 로컬 RAG 코퍼스
├── evaluation/            # retrieval_evalset.json
├── outputs/               # 최종 PDF 산출물
├── workflows/             # workflow, agents, retrieval, reporting 모듈
├── app.py                 # 실행 스크립트
├── pyproject.toml         # 프로젝트 설정
├── uv.lock                # 의존성 lock 파일
└── README.md
```

## Contributors
- 이름1 : Agent Design, Prompt Engineering
- 이름2 : PDF Parsing, Retrieval Evaluation
- 이름3 : Web Search, TRL Analysis
- 이름4 : Report Generation, PDF Formatting
