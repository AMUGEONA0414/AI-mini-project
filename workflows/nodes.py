from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .generation_eval import evaluate_generated_report, write_generation_eval
from .shared import *


def supervisor_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Supervisor", "Evaluating next step")
    if not state.get("rag_evidence"):
        return {"next_step": "rag"}
    if not state.get("web_findings"):
        return {"next_step": "web"}
    if not state.get("normalized_evidence"):
        return {"next_step": "normalize"}
    if not state.get("trl_estimates"):
        return {"next_step": "trl"}
    if not state.get("threat_assessment"):
        return {"next_step": "threat"}
    if not state.get("draft_report"):
        return {"next_step": "draft"}
    if state.get("needs_review"):
        return {"next_step": "review"}
    if state.get("final_report_markdown"):
        if state.get("formatting_status") != "completed":
            return {"next_step": "format"}
        return {"next_step": "end"}
    if not state.get("review_feedback", {}).get("passed", False):
        if state.get("review_iteration", 0) >= MAX_REVIEW_ITERATIONS:
            return {"next_step": "final"}
        return {"next_step": "draft"}
    return {"next_step": "final"}


def route_from_supervisor(state: WorkflowState) -> str:
    return state["next_step"]


def rag_agent_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("RAG", "Loading corpus and preparing embeddings")
    corpus = load_documents(RAG_SEARCH_ROOT)
    log_progress("RAG", f"Loaded {len(corpus)} source documents from {RAG_SEARCH_ROOT}")
    retrieved_docs: list[dict[str, Any]] = []
    rag_evidence: list[dict[str, Any]] = []
    references = state.get("references", [])
    chunked_corpus = build_chunked_corpus(corpus)
    selected_embedding_model, embedding_benchmarks = benchmark_embedding_models(
        documents=corpus,
        chunked_corpus=chunked_corpus,
    )
    chunk_embeddings: dict[str, list[float]] = {}
    if selected_embedding_model:
        chunk_embedding_map = get_embeddings(
            [chunk["text"] for chunk in chunked_corpus],
            model=selected_embedding_model,
            input_type="document",
        )
        chunk_embeddings = {chunk["chunk_id"]: chunk_embedding_map[chunk["text"]] for chunk in chunked_corpus}
    else:
        log_progress("RAG", "No embedding backend available for the selected bge model; falling back to lexical retrieval.")
    doc_lookup = {doc["path"]: doc for doc in corpus}
    retrieval_metrics = compute_retrieval_metrics(
        corpus,
        chunked_corpus=chunked_corpus,
        chunk_embeddings=chunk_embeddings or None,
        embedding_model=selected_embedding_model,
    )
    selected_strategy = retrieval_metrics.get("strategy", "hybrid")
    retrieval_benchmarks = retrieval_metrics.get("benchmarks", {})
    query_texts = {
        "HBM4": f"HBM4 memory packaging thermal validation roadmap {display_company_name(state['focal_company'])}",
        "PIM": f"PIM processing-in-memory AiM architecture survey research trend {display_company_name(state['focal_company'])}",
        "CXL": f"CXL compute express link memory expansion ecosystem standard direction {display_company_name(state['focal_company'])}",
    }
    query_embedding_map = (
        get_embeddings(
            list(query_texts.values()),
            model=selected_embedding_model,
            input_type="query",
        )
        if selected_embedding_model and selected_strategy != "lexical"
        else {}
    )

    for tech in state["target_technologies"]:
        log_progress("RAG", f"Retrieving documents for {tech}")
        query_text = query_texts[tech]
        query_embedding = query_embedding_map[query_text] if selected_strategy != "lexical" and query_text in query_embedding_map else None
        selected_docs = rank_documents_for_strategy(
            strategy=selected_strategy,
            query_text=query_text,
            technology=tech,
            chunked_corpus=chunked_corpus,
            chunk_embeddings=chunk_embeddings,
            doc_lookup=doc_lookup,
            query_embedding=query_embedding,
            limit=RAG_TOP_N_DOCS,
        )

        for doc in selected_docs:
            retrieved_docs.append(
                {
                    "technology": tech,
                    "source": doc["path"],
                    "title": doc["metadata"].get("title", doc["name"]),
                    "doc_type": doc["metadata"].get("sourcetype", "reference"),
                }
            )
            rag_evidence.append(
                {
                    "technology": tech,
                    "source_type": "rag",
                    "source": doc["path"],
                    "title": doc["metadata"].get("title", doc["name"]),
                    "summary": first_sentences(doc["sections"].get("overview", doc["body"])),
                    "evidence_type": doc["metadata"].get("sourcetype", "reference"),
                }
            )
            references.append(doc["path"])
        log_progress("RAG", f"{tech} top docs: {', '.join(doc['metadata'].get('title', doc['name']) for doc in selected_docs)}")

    notes = state.get("run_notes", []) + [
        f"RAG retrieved {len(retrieved_docs)} documents from mini-project/data.",
        f"Retrieval strategy selected: {selected_strategy}",
        f"Retrieval metrics: HitRate@{retrieval_metrics['k']}={retrieval_metrics['hit_rate_at_k']}, MRR={retrieval_metrics['mrr']}",
    ]
    log_progress("RAG", f"Completed with {len(retrieved_docs)} retrieved docs")
    return {
        "retrieved_docs": retrieved_docs,
        "rag_evidence": rag_evidence,
        "references": dedupe(references),
        "retrieval_metrics": retrieval_metrics,
        "retrieval_benchmarks": retrieval_benchmarks,
        "embedding_benchmarks": embedding_benchmarks,
        "selected_retrieval_strategy": selected_strategy,
        "embedding_model": selected_embedding_model,
        "run_notes": notes,
    }


def web_search_agent_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("WebSearch", "Starting Tavily-based web search")
    findings: list[dict[str, Any]] = []
    latest_signals: dict[str, list[str]] = {}
    references = state.get("references", [])
    filtered_count = 0
    jobs: list[tuple[str, str, str, str]] = []
    for tech in state["target_technologies"]:
        latest_signals[tech] = []
        for company in state["companies"]:
            log_progress("WebSearch", f"Queueing search {tech} / {display_company_name(company)}")
            for signal_type, query in build_tavily_queries(tech, company):
                jobs.append((tech, company, signal_type, query))
    log_progress("WebSearch", f"Queued {len(jobs)} Tavily jobs with max_workers={WEB_SEARCH_MAX_WORKERS}")

    def _job_runner(job: tuple[str, str, str, str]) -> tuple[tuple[str, str, str, str], list[dict[str, Any]]]:
        tech, company, signal_type, query = job
        return job, call_tavily_search(query=query, days=180, max_results=2)

    with ThreadPoolExecutor(max_workers=WEB_SEARCH_MAX_WORKERS) as executor:
        future_map = {executor.submit(_job_runner, job): job for job in jobs}
        for future in as_completed(future_map):
            tech, company, signal_type, query = future_map[future]
            try:
                results = future.result()[1]
            except Exception as exc:
                log_progress("WebSearch", f"Search failed for {tech} / {display_company_name(company)} / {signal_type}: {exc}")
                continue
            if not results:
                continue
            for rank, result in enumerate(results, start=1):
                summary, content = summarize_search_result(result)
                url = result.get("url", "")
                title = result.get("title") or url or f"{company} {tech} result"
                date_value = normalize_search_date(result, url=url, content=content)
                domain = extract_domain(url)
                trust_score = score_web_result(result, query=query, company=company, tech=tech, content=content)
                if not is_usable_web_result(result, trust_score=trust_score, content=content):
                    filtered_count += 1
                    log_progress("WebSearch", f"Filtered low-trust result: {title} ({domain}) score={trust_score}")
                    continue
                finding = {
                    "technology": tech,
                    "company": company,
                    "source_type": "web",
                    "source": url or f"tavily://{company}/{tech}/{signal_type}/{rank}",
                    "title": title,
                    "domain": domain,
                    "publisher": domain,
                    "source_kind": "tavily_search_result",
                    "trust_score": trust_score,
                    "signal_type": signal_type,
                    "date": date_value,
                    "retrieved_at": date.today().isoformat(),
                    "summary": summary,
                    "content": content,
                    "query": query,
                    "rank": rank,
                }
                findings.append(finding)
                references.append(finding["source"])
                latest_signals[tech].append(f"{display_company_name(company)}: {summary}")

    notes = state.get("run_notes", []) + [f"Web search collected {len(findings)} Tavily-based public-signal findings."]
    log_progress("WebSearch", f"Completed with {len(findings)} findings, filtered={filtered_count}")
    return {
        "web_findings": findings,
        "latest_signals": latest_signals,
        "references": dedupe(references),
        "run_notes": notes,
    }


def evidence_normalizer_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Analyzer", "Normalizing evidence from RAG and web search")
    rag_docs = {item["source"]: load_document_by_path(item["source"]) for item in state["rag_evidence"]}
    web_docs = {
        item["source"]: {
            "path": item["source"],
            "name": item.get("title", item["source"]),
            "metadata": {
                "title": item.get("title", item["source"]),
                "company": item["company"],
                "technology": item["technology"],
                "sourcetype": item.get("source_kind", "tavily_search_result"),
                "source": item.get("domain", "web"),
                "date": item.get("date", "날짜 미상"),
            },
            "body": item.get("content", item.get("summary", "")),
            "sections": {"overview": item.get("content", item.get("summary", ""))},
        }
        for item in state["web_findings"]
    }
    normalized_evidence: list[dict[str, Any]] = []
    evidence_table: list[dict[str, Any]] = []
    evidence_traceability: dict[str, Any] = {
        "background": [],
        "technology_sections": {},
        "competitor_rows": [],
        "strategy": [],
    }
    for tech in state["target_technologies"]:
        tech_rag = [item for item in state["rag_evidence"] if item["technology"] == tech]
        tech_web = [item for item in state["web_findings"] if item["technology"] == tech]
        evidence_traceability["technology_sections"][tech] = {
            "rag_sources": [item["source"] for item in tech_rag[:2]],
            "web_sources": [item["source"] for item in tech_web[:2]],
        }
        for company in state["companies"]:
            company_web = [item for item in tech_web if item["company"] == company]
            company_rag = [
                item
                for item in tech_rag
                if company.lower().split()[0] in rag_docs[item["source"]]["body"].lower()
            ]
            source_types = dedupe(
                [rag_docs[item["source"]]["metadata"].get("sourcetype", "reference") for item in company_rag]
                + [web_docs[item["source"]]["metadata"].get("sourcetype", "public_signal") for item in company_web]
            )
            evidence_sources = dedupe([item["source"] for item in company_rag + company_web])[:4]
            row = {
                "technology": tech,
                "company": company,
                "rag_items": company_rag,
                "web_items": company_web,
                "source_types": source_types,
                "evidence_sources": evidence_sources,
                "combined_text": " ".join(
                    [rag_docs[item["source"]]["body"] for item in company_rag]
                    + [web_docs[item["source"]]["body"] for item in company_web]
                ),
                "progress_count": len([item for item in company_web if item["signal_type"] == "progress"]),
                "risk_count": len([item for item in company_web if item["signal_type"] == "risk"]),
            }
            normalized_evidence.append(row)
            evidence_table.append(
                {
                    "technology": tech,
                    "company": company,
                    "signal_type": dedupe([item["signal_type"] for item in company_web]),
                    "sources": evidence_sources,
                    "source_types": source_types,
                    "updated_within_months": 12,
                }
            )
    evidence_traceability["background"] = dedupe(
        [item["source"] for item in state["web_findings"][:4]] + [item["source"] for item in state["rag_evidence"][:2]]
    )
    notes = state.get("run_notes", []) + ["Evidence normalizer standardized RAG and web findings into company-technology evidence rows."]
    log_progress("Analyzer", f"Normalization completed with {len(normalized_evidence)} evidence rows")
    return {
        "normalized_evidence": normalized_evidence,
        "evidence_table": evidence_table,
        "evidence_traceability": evidence_traceability,
        "run_notes": notes,
    }


def trl_assessment_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("TRL", "Assessing TRL from normalized evidence")
    trl_estimates: list[dict[str, Any]] = []
    for item in state.get("normalized_evidence", []):
        indirect_signals = dedupe([web_item["signal_type"] for web_item in item["web_items"]] + item["source_types"])
        estimated_trl, confidence, trl_reasons = classify_trl(
            item["combined_text"],
            len(item["source_types"]),
            item["source_types"],
            len(item["web_items"]),
        )
        trl_estimates.append(
            {
                "technology": item["technology"],
                "company": item["company"],
                "estimated_trl": estimated_trl,
                "confidence": confidence,
                "evidence": item["evidence_sources"],
                "indirect_signals": indirect_signals,
                "reasoning": trl_reasons,
            }
        )
    notes = state.get("run_notes", []) + ["TRL assessment node estimated company-by-technology readiness levels."]
    log_progress("TRL", f"TRL assessment completed with {len(trl_estimates)} estimates")
    return {"trl_estimates": trl_estimates, "run_notes": notes}


def threat_strategy_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Threat", "Deriving threat assessment and strategy outline")
    threat_assessment: dict[str, list[dict[str, Any]]] = {}
    technology_snapshots: list[dict[str, Any]] = []
    competitor_analysis_rows: list[dict[str, Any]] = []
    priority_targets: list[str] = []
    trl_lookup = {(item["technology"], item["company"]): item for item in state.get("trl_estimates", [])}
    evidence_traceability = dict(state.get("evidence_traceability", {}))
    evidence_traceability.setdefault("competitor_rows", [])
    evidence_traceability.setdefault("strategy", [])
    for tech in state["target_technologies"]:
        threat_assessment[tech] = []
        tech_rag = [item for item in state["rag_evidence"] if item["technology"] == tech]
        tech_web = [item for item in state["web_findings"] if item["technology"] == tech]
        progress_summaries = [item["summary"] for item in tech_web if item["signal_type"] == "progress"]
        risk_summaries = [item["summary"] for item in tech_web if item["signal_type"] == "risk"]
        base_summary = tech_rag[0]["summary"] if tech_rag else ""
        technology_snapshots.append(
            {
                "technology": tech,
                "current": to_sentence(strip_leading_subject(base_summary, [tech, "삼성전자", "마이크론"])),
                "differentiator": to_sentence(
                    strip_leading_subject(first_sentences(" ".join(progress_summaries), 1), ["삼성전자", "마이크론"])
                    or "고대역폭, 시스템 통합, 메모리 효율 측면에서 기존 세대 대비 차별화 포인트가 관찰된다"
                ),
                "challenge": to_sentence(
                    strip_leading_subject(first_sentences(" ".join(risk_summaries), 1), ["삼성전자", "마이크론"])
                    or "상용화 단계에서 수율, 호환성, 비용, 검증 일정 리스크가 남아 있다"
                ),
            }
        )
        for company in state["companies"]:
            evidence_row = next(
                item for item in state.get("normalized_evidence", []) if item["technology"] == tech and item["company"] == company
            )
            company_web = evidence_row["web_items"]
            company_progress = [item for item in company_web if item["signal_type"] == "progress"]
            company_risk = [item for item in company_web if item["signal_type"] == "risk"]
            trl = trl_lookup[(tech, company)]
            estimated_trl = trl["estimated_trl"]
            confidence = trl["confidence"]
            trl_reasons = trl["reasoning"]
            threat_level = "high" if estimated_trl >= 6 else ("medium" if estimated_trl >= 5 else "low")
            reason = (
                f"progress 신호 {len(company_progress)}건, risk 신호 {len(company_risk)}건, "
                f"출처 유형 {len(evidence_row['source_types'])}종을 기준으로 판단"
            )
            evidence_sources = evidence_row["evidence_sources"][:3]
            threat_assessment[tech].append(
                {
                    "technology": tech,
                    "company": company,
                    "threat_level": threat_level,
                    "reason": reason,
                }
            )
            competitor_analysis_rows.append(
                {
                    "company": company,
                    "technology": tech,
                    "technology_trend": to_sentence(first_sentences(" ".join(item["summary"] for item in company_progress), 2))
                    or "최신 공개 신호가 제한적이다.",
                    "estimated_trl": estimated_trl,
                    "confidence": confidence,
                    "threat_level": threat_level,
                    "key_evidence": ", ".join(source_label_from_state(src, state) for src in evidence_sources),
                    "notes": reason,
                }
            )
            evidence_traceability["competitor_rows"].append(
                {
                    "technology": tech,
                    "company": company,
                    "sources": evidence_sources,
                    "reasoning": trl_reasons,
                    "signal_count": len(company_web),
                }
            )
            if threat_level == "high":
                priority_targets.append(f"{tech}:{company}")
    limitations = [
        "TRL 4~6 구간은 공개 정보만으로 확정할 수 없으므로 추정치로만 사용한다.",
        "비공개 수율, 원가, 고객 검증 상태는 분석 범위에서 제외한다.",
        "증거가 부족한 경우 보수적으로 더 낮은 TRL을 선택했다.",
    ]
    strategy_outline = {
        "priorities": dedupe(
            [
                "HBM4와 CXL은 경쟁사 위협이 직접적으로 관찰되는 영역으로 최우선 추적 대상으로 둔다."
                if any(item.startswith("HBM4") or item.startswith("CXL") for item in priority_targets)
                else "위협도가 높은 기술 영역부터 우선적으로 추적한다.",
                "PIM은 상용화 시점보다 생태계 선점 가능성과 응용처 확보 여부를 중심으로 추적한다.",
            ]
        ),
        "short_term": [
            "고객 검증, 샘플 공급, 인터페이스 호환성 관련 공개 신호를 분기 단위로 재점검한다.",
            "HBM4와 CXL 관련 패키지, 시스템, 고객 협업 이슈를 내부 로드맵과 연결해 비교 점검한다.",
        ],
        "mid_term": [
            "TRL 6 이상으로 추정되는 경쟁사 기술에 대해 고객사 협력, 생태계 표준, 양산 준비 신호를 묶어 추적한다.",
            "PIM은 단독 성능보다 적용 워크로드와 소프트웨어 생태계 확보 가능성을 포함해 대응 로드맵을 준비한다.",
        ],
    }
    evidence_traceability["strategy"] = [
        {
            "priority": item,
            "supporting_sources": [row["sources"] for row in evidence_traceability["competitor_rows"] if row["technology"] in item],
        }
        for item in strategy_outline["priorities"]
    ]
    notes = state.get("run_notes", []) + ["Threat strategy node translated TRL estimates into threat levels and action priorities."]
    log_progress("Threat", f"Threat assessment completed with {len(competitor_analysis_rows)} comparison rows")
    return {
        "threat_assessment": threat_assessment,
        "technology_snapshots": technology_snapshots,
        "competitor_analysis_rows": competitor_analysis_rows,
        "strategy_outline": strategy_outline,
        "limitations": limitations,
        "evidence_traceability": evidence_traceability,
        "run_notes": notes,
    }


def draft_generation_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Draft", "Generating report draft")
    issues = state.get("review_feedback", {}).get("issues", [])
    revision_prefix = ""
    if issues:
        revision_prefix = "이전 리뷰에서 지적된 누락 사항을 반영해 본문 구조와 한계 기술을 보강했다.\n\n"
    summary_sentences = [
        "HBM4, PIM, CXL은 차세대 메모리 경쟁력과 시스템 확장성에 직접 영향을 주는 핵심 축이다.",
        "현재 공개 신호 기준으로는 삼성전자와 마이크론 모두 TRL 4~6 구간의 실증·검증 단계에 있으며, HBM4와 CXL 관련 시그널이 가장 직접적인 위협 요인으로 해석된다.",
        "다만 평가 근거는 공개 문서, 특허, 채용 공고, 보도자료에 기반한 추정이며 수율과 고객 검증의 세부 수준은 확인되지 않았다.",
        "따라서 SK하이닉스 관점에서 경쟁사 위협 우선순위와 대응 포인트를 정리하는 비교 자료로 활용하는 것이 적절하다.",
    ]
    section_two = build_technology_snapshot_block(state)
    competitor_table = build_competitor_analysis_table(state)
    strategy_outline = build_strategy_outline(state)
    background_context = build_background_context(state)
    system_prompt = """너는 SK하이닉스 R&D 담당자를 위한 반도체 경쟁사 분석 보고서를 작성하는 한국어 분석가다.
반드시 한국어로만 작성하고, 주어진 근거를 벗어나서 추측을 확대하지 마라.
출력은 마크다운이어야 하며 아래 섹션만 작성한다.
## SUMMARY
## 1. 분석 배경
## 2. 분석 대상 기술 현황
## 3. 경쟁사 동향 분석
## 4. 기술별 경쟁사 비교표
## 5. 전략적 시사점
SUMMARY는 배경 설명 없이 최종 결론만 10문장 이내로 요약한다.
1. 분석 배경은 '왜 지금 분석해야 하는가'를 설명하고 현재 상황 파악을 포함한다.
2. 분석 대상 기술 현황은 기술별로 반드시 현재 / 차별점 / 도전과제를 모두 작성한다.
   기술명은 `### HBM4` 같은 소제목으로 쓰지 말고 아래 예시처럼 bullet 형식으로만 작성한다.
   - HBM4
     - 현재: ...
     - 차별점: ...
     - 도전과제: ...
3. 경쟁사 동향 분석 섹션은 반드시 마크다운 표로 작성하고, 열 순서는 기술 / 회사 / 기술 동향 / 추정 TRL / 위협 수준 / 핵심 근거를 따른다.
4. 기술별 경쟁사 비교표는 별도 섹션으로 작성한다.
5. 전략적 시사점은 반드시 R&D 우선순위 / 단기 대응 방향 / 중기 대응 방향의 세 묶음으로 작성한다.
회사명은 삼성전자, 마이크론, SK하이닉스로 표기한다.
전체 서술은 반드시 SK하이닉스 관점에서 경쟁사를 평가하는 형식이어야 한다.
보고서 최상단에는 제목과 작성일을 아래 형식으로 먼저 출력한다.
# SK하이닉스 관점 반도체 기술 전략 분석 보고서
작성일: YYYY-MM-DD
과장된 수사나 홍보성 문장은 금지한다."""
    user_prompt = f"""사용자 요청:
{state['user_query']}

기준 회사:
{display_company_name(state['focal_company'])}

이전 리뷰 반영 메모:
{revision_prefix or '없음'}

핵심 메시지 초안:
{" ".join(summary_sentences)}

분석 배경 근거:
{background_context}

기술 현황 근거:
{section_two}

경쟁사 동향 분석 표 초안:
| 기술 | 회사 | 기술 동향 | 추정 TRL | 위협 수준 | 핵심 근거 |
| --- | --- | --- | --- | --- | --- |
{competitor_table}

전략적 시사점 방향:
{strategy_outline}

한계:
{chr(10).join(f"- {item}" for item in state['limitations'])}
"""
    draft = call_openai_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2).strip()
    notes = state.get("run_notes", []) + ["Draft generator assembled the markdown report from structured evidence."]
    log_progress("Draft", "Draft generation completed")
    return {"draft_report": draft, "needs_review": True, "run_notes": notes}


def review_agent_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Review", "Reviewing draft")
    draft = state.get("draft_report", "")
    deterministic_issues = rule_based_review_checks(draft)
    if deterministic_issues:
        log_progress("Review", f"Rule-based checks found issues={len(deterministic_issues)}")
    else:
        log_progress("Review", "Rule-based checks passed with no issues")
    system_prompt = """너는 SK하이닉스 관점 반도체 전략 보고서 품질 검토자다.
주어진 초안을 보고 아래 기준을 엄격하게 검사하라.
1. SUMMARY, 분석 배경, 분석 대상 기술 현황, 경쟁사 동향 분석, 기술별 경쟁사 비교표, 전략적 시사점 섹션 존재
2. 경쟁사 동향 분석이 표 형태인지 여부
3. 분석 대상 기술 현황에 각 기술별 현재/차별점/도전과제가 모두 있는지 여부
4. 분석 대상 기술 현황이 `###` 소제목이 아니라 bullet 구조인지 여부
5. 삼성전자와 마이크론이 모두 언급되는지 여부
6. TRL 4~6 추정 한계가 명시되는지 여부
7. SUMMARY가 10문장 이내이며 SK하이닉스 관점의 최종 결론 요약인지 여부
8. 전략적 시사점에 R&D 우선순위/단기 대응 방향/중기 대응 방향이 모두 있는지 여부
중요:
- issues에는 실패한 항목만 넣어라.
- 통과한 항목, 적합함, 존재함, 충족, 문제없음 같은 문장은 issues에 넣지 마라.
- 모든 기준을 충족하면 passed=true, issues=[] 로 반환하라.
JSON만 반환하라.
형식: {{"passed": bool, "issues": [str, ...]}}"""
    user_prompt = f"""초안:
{draft}

규칙 기반 사전 점검 결과:
{chr(10).join(f"- {item}" for item in deterministic_issues) or "- 없음"}

참고 한계:
{chr(10).join(f"- {item}" for item in state['limitations'])}
"""
    review = call_openai_json(system_prompt=system_prompt, user_prompt=user_prompt)
    raw_passed = bool(review.get("passed"))
    raw_issues = [str(item) for item in review.get("issues", [])]
    failure_keywords = ["누락", "부족", "없음", "아님", "않음", "미포함", "미기재", "미명시", "초과", "실패", "불충분", "필요", "문제"]
    pass_keywords = ["적합", "존재", "충족", "포함되어 있음", "명확히 명시", "작성되어 있음", "문제없", "통과"]
    issues = list(deterministic_issues)
    for item in raw_issues:
        lowered = item.lower()
        if any(keyword in item for keyword in pass_keywords):
            continue
        if not any(keyword in item for keyword in failure_keywords):
            continue
        if "없음" in item and any(token in lowered for token in ["소제목이 없음", "문제 없음"]):
            continue
        issues.append(item)
    issues = dedupe(issues)
    passed = (raw_passed and not issues) or (not issues)
    notes = state.get("run_notes", []) + [f"Review agent {'passed' if passed else 'requested revision'} at iteration {state.get('review_iteration', 0) + 1}."]
    if passed:
        log_progress("Review", "Review completed: passed=True")
    else:
        issue_text = " | ".join(issues) if issues else "사유 미상"
        log_progress("Review", f"Review completed: passed=False issues={issue_text}")
    return {
        "review_feedback": {
            "passed": passed,
            "issues": issues,
            "quality_gate": "section/table/technology-structure/company/TRL limitation/strategy",
        },
        "review_iteration": state.get("review_iteration", 0) + 1,
        "needs_review": False,
        "run_notes": notes,
    }


def final_report_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Final", "Generating final report")
    references = build_reference_block(state)
    limitations = "\n".join(f"- {item}" for item in state.get("limitations", []))
    comparison_rows = build_comparison_table(state)
    background_context = build_background_context(state)
    technology_snapshots = build_technology_snapshot_block(state)
    strategy_outline = build_strategy_outline(state)
    compact_comparison_guidance = build_compact_comparison_guidance(state)
    system_prompt = """너는 SK하이닉스 R&D 담당자에게 제출할 반도체 경쟁사 분석 보고서의 최종 편집자다.
반드시 한국어 마크다운으로만 답하고, 아래 조건을 지켜라.
1. 초안의 핵심 내용을 유지하되 문장을 더 자연스럽게 다듬을 것
2. '## 기술별 경쟁사 비교표' 섹션은 반드시 '## REFERENCE'보다 위에 둘 것
3. 보고서 순서는 SUMMARY -> 1. 분석 배경 -> 2. 분석 대상 기술 현황 -> 3. 경쟁사 동향 분석 -> 4. 기술별 경쟁사 비교표 -> 5. 전략적 시사점 -> 한계 및 해석 주의사항 -> REFERENCE 여야 한다
4. 회사명은 삼성전자, 마이크론, SK하이닉스로 표기할 것
5. SUMMARY는 경쟁사 분석 결과를 바탕으로 SK하이닉스가 당장 주목해야 할 위협과 대응 포인트를 최종 요약해야 한다
5-1. SUMMARY는 최대 10문장까지만 허용한다
6. REFERENCE는 경로명이 아니라 '자료명 / 출처 / 날짜' 형식의 불릿 목록으로 정리할 것
7. 코드펜스, 굵게 표시, 불필요한 마크다운 장식은 사용하지 말 것
8. '2. 분석 대상 기술 현황'에서는 `### HBM4` 같은 소제목을 쓰지 말고 bullet 구조만 사용할 것
9. '4. 기술별 경쟁사 비교표'는 반드시 다음 5개 열만 사용할 것: 기술 | 삼성전자 위협 수준 | 마이크론 위협 수준 | 주요 차별점 및 특징 | SK하이닉스 대응 포인트
10. '삼성전자 위협 수준'과 '마이크론 위협 수준'은 동일한 성격의 짧은 값(높음/중간/낮음)으로 맞출 것
11. '주요 차별점 및 특징' 셀은 반드시 두 줄로 쓸 것: 첫 줄은 '삼성전자: ...', 둘째 줄은 '마이크론: ...'
12. '주요 차별점 및 특징'과 'SK하이닉스 대응 포인트'는 셀당 1문장 또는 짧은 구문으로 제한할 것
11. 경쟁사 동향 분석 표는 기술 열이 가장 왼쪽에 오도록 작성할 것
13. 보고서 최상단에는 제목과 작성일을 아래 형식으로 먼저 둘 것
# SK하이닉스 관점 반도체 기술 전략 분석 보고서
작성일: YYYY-MM-DD"""
    user_prompt = f"""초안:
{state['draft_report']}

기준 회사:
{display_company_name(state['focal_company'])}

배경 정리:
{background_context}

기술 현황 정리:
{technology_snapshots}

전략 시사점 정리:
{strategy_outline}

기술별 경쟁사 비교표 작성 가이드:
{compact_comparison_guidance}

반드시 아래 형식의 비교표를 보고서에 포함:
## 4. 기술별 경쟁사 비교표
| 기술 | 삼성전자 위협 수준 | 마이크론 위협 수준 | 주요 차별점 및 특징 | SK하이닉스 대응 포인트 |
| --- | --- | --- | --- | --- |
{comparison_rows}

반드시 아래 참고문헌을 포함:
## REFERENCE
{references}

반드시 아래 한계를 포함:
## 한계 및 해석 주의사항
{limitations}
"""
    final_report = call_openai_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1).strip()
    final_report = sanitize_report_markdown(final_report)
    notes = state.get("run_notes", []) + ["Final report node locked the approved markdown report."]
    log_progress("Final", "Final report generation completed")
    return {"final_report_markdown": final_report, "run_notes": notes}


def formatting_node(state: WorkflowState) -> dict[str, Any]:
    log_progress("Formatting", "Writing markdown and rendering PDF")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    markdown_path = OUTPUT_ROOT / "tech_strategy_report.md"
    pdf_path = OUTPUT_ROOT / "ai-mini_output.pdf"
    cleaned_report = sanitize_report_markdown(state["final_report_markdown"])
    markdown_path.write_text(cleaned_report, encoding="utf-8")
    generation_eval = evaluate_generated_report({**state, "final_report_markdown": cleaned_report})
    write_generation_eval(generation_eval)
    TRACEABILITY_OUTPUT_PATH.write_text(
        json.dumps(
            {
                "retrieval_metrics": state.get("retrieval_metrics", {}),
                "retrieval_benchmarks": state.get("retrieval_benchmarks", {}),
                "embedding_benchmarks": state.get("embedding_benchmarks", {}),
                "selected_retrieval_strategy": state.get("selected_retrieval_strategy", ""),
                "embedding_model": state.get("embedding_model", ""),
                "embedding_candidates": state.get("embedding_candidates", []),
                "generation_eval": generation_eval,
                "evidence_traceability": state.get("evidence_traceability", {}),
                "references": state.get("references", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    render_plain_text_pdf(cleaned_report, pdf_path)
    log_progress("Formatting", f"Traceability artifact saved: {TRACEABILITY_OUTPUT_PATH}")
    notes = state.get("run_notes", []) + ["Formatting node wrote markdown and a reportlab-rendered PDF artifact."]
    notes.append(f"Traceability artifact saved to {TRACEABILITY_OUTPUT_PATH}")
    notes.append(f"Generation evaluation saved to {GENERATION_EVAL_OUTPUT_PATH}")
    return {
        "pdf_path": str(pdf_path),
        "formatting_status": "completed",
        "generation_eval": generation_eval,
        "run_notes": notes,
    }


def checkpointed_node(node_name: str, handler: Any):
    def _wrapped(state: WorkflowState) -> dict[str, Any]:
        result = handler(state)
        next_state = dict(state)
        next_state.update(result)
        checkpoint_path = save_checkpoint(node_name, next_state)
        result["checkpoint_path"] = checkpoint_path
        log_progress("Checkpoint", f"Saved state after {node_name}: {checkpoint_path}")
        return result

    return _wrapped


def build_workflow():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed in the current Python environment.")
    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", checkpointed_node("supervisor", supervisor_node))
    graph.add_node("rag", checkpointed_node("rag", rag_agent_node))
    graph.add_node("web", checkpointed_node("web", web_search_agent_node))
    graph.add_node("normalize", checkpointed_node("normalize", evidence_normalizer_node))
    graph.add_node("trl", checkpointed_node("trl", trl_assessment_node))
    graph.add_node("threat", checkpointed_node("threat", threat_strategy_node))
    graph.add_node("draft", checkpointed_node("draft", draft_generation_node))
    graph.add_node("review", checkpointed_node("review", review_agent_node))
    graph.add_node("final", checkpointed_node("final", final_report_node))
    graph.add_node("format", checkpointed_node("format", formatting_node))
    graph.add_edge(START, "supervisor")
    for node in ["rag", "web", "normalize", "trl", "threat", "draft", "review", "final", "format"]:
        graph.add_edge(node, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "rag": "rag",
            "web": "web",
            "normalize": "normalize",
            "trl": "trl",
            "threat": "threat",
            "draft": "draft",
            "review": "review",
            "final": "final",
            "format": "format",
            "end": END,
        },
    )
    return graph.compile()


def run_workflow_without_langgraph(initial_state: WorkflowState) -> WorkflowState:
    state = dict(initial_state)
    node_map = {
        "rag": checkpointed_node("rag", rag_agent_node),
        "web": checkpointed_node("web", web_search_agent_node),
        "normalize": checkpointed_node("normalize", evidence_normalizer_node),
        "trl": checkpointed_node("trl", trl_assessment_node),
        "threat": checkpointed_node("threat", threat_strategy_node),
        "draft": checkpointed_node("draft", draft_generation_node),
        "review": checkpointed_node("review", review_agent_node),
        "final": checkpointed_node("final", final_report_node),
        "format": checkpointed_node("format", formatting_node),
    }
    while True:
        state.update(checkpointed_node("supervisor", supervisor_node)(state))
        next_step = state["next_step"]
        if next_step == "end":
            return WorkflowState(**state)
        state.update(node_map[next_step](state))


def run_demo() -> WorkflowState:
    initial_state = initialize_state(
        user_query="SK하이닉스 관점에서 HBM4, PIM, CXL 분야의 경쟁사 위협을 비교한 기술 전략 보고서를 만들어줘."
    )
    if StateGraph is None:
        return run_workflow_without_langgraph(initial_state)
    app = build_workflow()
    return app.invoke(initial_state)
