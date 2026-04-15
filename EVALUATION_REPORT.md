# Evaluation Report

## Summary

현재 workflow는 Windows 환경에서 `uv run python app.py`로 end-to-end 실행되며, Markdown/PDF 보고서와 평가 산출물을 생성합니다.

생성 산출물:

- `outputs/tech_strategy_report.md`
- `outputs/ai-mini_output.pdf`
- `outputs/evidence_traceability.json`
- `outputs/generation_eval.json`

## Retrieval Evaluation

Retrieval 평가셋은 `evaluation/retrieval_evalset.json`에 정의되어 있으며, HBM4/PIM/CXL 각 10개씩 총 30개 query로 구성했습니다.

최근 실행 결과:

| Strategy | HitRate@6 | MRR | Queries |
| --- | ---: | ---: | ---: |
| dense | 0.967 | 0.900 | 30 |
| lexical | 0.967 | 0.828 | 30 |
| hybrid | 0.967 | 0.867 | 30 |

선택된 retrieval strategy는 `dense`입니다.

## Generation Evaluation

생성 보고서는 `evaluation/gold_report.md`를 기준 보고서로 사용해 section-level embedding similarity를 계산합니다. SemScore 기준은 0.7 이상입니다.

최근 실행 결과:

| Metric | Score | Threshold | Result |
| --- | ---: | ---: | --- |
| SemScore | 0.817 | 0.700 | Pass |
| Evidence-grounding score | 0.639 | 0.700 | Diagnostic |

Section-level SemScore:

| Section | Score |
| --- | ---: |
| Analysis Background | 0.790 |
| Technology Status | 0.751 |
| Competitor Trends | 0.861 |
| Strategic Implications | 0.866 |

## Quality Criteria

보고서 품질 기준은 다음 5개 항목으로 확인합니다.

| Criterion | Result |
| --- | --- |
| 분석 배경 / 기술 현황 / 경쟁사 동향 / 전략적 시사점 섹션 포함 | Pass |
| Samsung, Micron 각각에 대해 TRL 기반 성숙도 평가 포함 | Pass |
| TRL 4~6 추정 구간의 한계와 근거 명시 | Pass |
| 논문, 보도자료, 웹 공개 신호 등 다각도 출처 반영 | Pass |
| 긍정/부정 신호를 함께 수집해 확증 편향 완화 | Pass |

최종 판정은 SemScore와 품질 기준을 기준으로 하며, 현재 `overall_passed=true`입니다. Evidence-grounding score는 근거 연결성을 추가 진단하기 위한 보조 지표로 저장합니다.

## Notes

- PDF 25개 중 20개는 텍스트 추출에 성공해 RAG chunk로 사용됩니다.
- 나머지 5개 PDF는 텍스트 레이어가 없는 이미지형 PDF라 일반 PDF parser로는 본문 추출이 되지 않습니다.
- 이미지형 PDF까지 검색 근거로 활용하려면 OCR 적용 또는 텍스트 원본 자료로 교체가 필요합니다.
