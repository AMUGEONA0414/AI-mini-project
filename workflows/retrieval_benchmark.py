from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .clients import (
    call_huggingface_embeddings,
    call_jina_embeddings,
    call_openai_embeddings,
    call_voyage_embeddings,
)
from .config import (
    DATA_ROOT,
    EMBEDDING_CACHE_ROOT,
    EVAL_ROOT,
    RETRIEVAL_EVALSET_PATH,
    log_progress,
)
from .retrieval import (
    build_chunked_corpus,
    cosine_similarity,
    get_embeddings,
    lexical_overlap_score,
    score_chunk_for_strategy,
)
from .sources import load_external_reference_pdfs


def available_candidate_configs() -> dict[str, dict[str, Any]]:
    """
    노트북에서 기대하는 후보 설정을 반환합니다.
    """
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_voyage = bool(os.getenv("VOYAGE_API_KEY"))
    has_jina = bool(os.getenv("JINA_API_KEY"))
    has_hf = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"))

    configs = {
        "tfidf_baseline": {
            "kind": "lexical",
            "available": True,
            "builder": "tfidf",
            "note": "빠른 로컬 baseline",
            "model_name": None,
            "provider": None,
        },
        "bm25": {
            "kind": "sparse",
            "available": True,
            "builder": "bm25",
            "note": "전문용어 exact match baseline",
            "model_name": None,
            "provider": None,
        },
        "bge_m3": {
            "kind": "dense",
            "available": True,
            "note": "project.md dense 후보",
            "model_name": "BAAI/bge-m3",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "document_prefix": "",
            "provider": "huggingface" if has_hf else "openai", # Fallback to openai if no HF key, though models differ
        },
        "multilingual_e5_base": {
            "kind": "dense",
            "available": True,
            "note": "e5 large 대체 quick candidate",
            "model_name": "intfloat/multilingual-e5-base",
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
            "provider": "huggingface" if has_hf else "openai",
        },
        "voyage_3_large": {
            "kind": "remote_dense",
            "available": has_voyage,
            "note": "Voyage API 후보",
            "model_name": "voyage-3-large",
            "provider": "voyage",
        },
        "hybrid_bge_m3_bm25": {
            "kind": "hybrid",
            "available": True,
            "note": "dense + sparse fusion",
            "dense_model_name": "BAAI/bge-m3",
            "provider": "huggingface" if has_hf else "openai",
        },
    }
    return configs


def load_qa_ground(max_questions: int = 10) -> pd.DataFrame:
    """
    평가 데이터셋을 로드합니다.
    """
    if not RETRIEVAL_EVALSET_PATH.exists():
        # 기본 데이터 생성 (노트북에서 본 질문들 일부 포함)
        eval_data = [
            {
                "question_no": 1,
                "q_type": "Factual",
                "question": "PIM의 두 가지 핵심 접근 방식인 Processing-Using-Memory와 Processing-Near-Memory의 차이점은?",
                "expected_source_ids": ["2012.03112"],
            },
            {
                "question_no": 2,
                "q_type": "None",
                "question": "UPMEM PIM 아키텍처에서 각 DPU가 독립적으로 접근하는 메모리의 명칭과 크기는?",
                "expected_source_ids": ["2105.03814"],
            },
            {
                "question_no": 3,
                "q_type": "None",
                "question": "CXL의 세 가지 프로토콜(CXL.io, CXL.mem, CXL.cache)의 역할은?",
                "expected_source_ids": ["2306.11227", "2412.20249"],
            },
        ]
        EVAL_ROOT.mkdir(parents=True, exist_ok=True)
        RETRIEVAL_EVALSET_PATH.write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    with open(RETRIEVAL_EVALSET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    if "question" not in df.columns and "query" in df.columns:
        df = df.rename(columns={"query": "question"})
    if "expected_source_ids" not in df.columns and "expected_titles" in df.columns:
        df = df.rename(columns={"expected_titles": "expected_source_ids"})
    
    return df.head(max_questions)


def run_retrieval_benchmark(
    max_pdfs: int = 5,
    max_pages_per_pdf: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_questions: int = 5,
    top_k: int = 5,
    time_budget_seconds: int = 600,
    candidate_names: list[str] | None = None,
    remote_min_interval_seconds: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """
    다양한 검색 후보에 대해 벤치마크를 실행합니다.
    """
    start_time = time.time()
    configs = available_candidate_configs()
    if candidate_names:
        configs = {k: v for k, v in configs.items() if k in candidate_names}
    
    qa_df = load_qa_ground(max_questions=max_questions)
    docs = load_external_reference_pdfs(DATA_ROOT)
    if not docs:
        # data 폴더에 직접 있는 경우도 체크 (sources.py는 glob("*.pdf")를 씀)
        pass 
    
    docs = docs[:max_pdfs]
    chunked_corpus = build_chunked_corpus(docs)
    
    summary_results = []
    detail_frames = {}
    benchmark_errors = []
    
    for name, config in configs.items():
        if not config.get("available", False):
            benchmark_errors.append({"candidate": name, "error": "candidate unavailable in current environment"})
            continue
            
        if time.time() - start_time > time_budget_seconds:
            benchmark_errors.append({"candidate": name, "error": "time budget exceeded"})
            continue
            
        try:
            cand_start = time.time()
            kind = config["kind"]
            model = config.get("model_name") or config.get("dense_model_name")
            
            # 임베딩 확보 (dense/hybrid인 경우)
            chunk_embeddings = {}
            if kind in ("dense", "remote_dense", "hybrid"):
                texts = [c["text"] for c in chunked_corpus]
                # get_embeddings는 내부적으로 캐시를 사용함
                emb_map = get_embeddings(texts, model=model, input_type="document")
                chunk_embeddings = {c["chunk_id"]: emb_map[c["text"]] for c in chunked_corpus}
            
            # 각 질문에 대해 검색 수행
            rows = []
            for _, qa_row in qa_df.iterrows():
                query = qa_row["question"]
                expected_ids = qa_row["expected_source_ids"]
                if isinstance(expected_ids, str):
                    expected_ids = [expected_ids]
                
                query_embedding = None
                if kind in ("dense", "remote_dense", "hybrid"):
                    query_embedding_map = get_embeddings([query], model=model, input_type="query")
                    query_embedding = query_embedding_map[query]
                
                # 점수 계산
                scores = []
                for chunk in chunked_corpus:
                    # 간단한 strategy 점수 계산 (retrieval.py의 score_chunk_for_strategy 활용)
                    # 여기서는 technology가 명확하지 않으므로 빈 문자열 전달
                    score = score_chunk_for_strategy(
                        strategy="dense" if kind in ("dense", "remote_dense") else ("lexical" if kind in ("lexical", "sparse") else "hybrid"),
                        query_text=query,
                        query_embedding=query_embedding,
                        chunk=chunk,
                        chunk_embedding=chunk_embeddings.get(chunk["chunk_id"]),
                        technology="",
                    )
                    scores.append((chunk, score))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                top_results = scores[:top_k]
                
                # 메트릭 계산
                hit = 0
                mrr = 0.0
                semscore_proxy = 0.0
                top_titles = []
                
                for rank, (chunk, score) in enumerate(top_results, start=1):
                    doc_path = Path(chunk["doc_path"])
                    doc_id = doc_path.stem
                    top_titles.append(chunk["title"])
                    
                    if any(expected in doc_id for expected in expected_ids):
                        if hit == 0:
                            hit = 1
                            mrr = 1.0 / rank
                    
                    # SemScore Proxy: 정답 문서와 검색된 청크의 유사도 (여기서는 간단히 cosine similarity 사용)
                    # 실제 정답 텍스트가 없으므로 검색된 점수를 proxy로 활용하거나 
                    # 정답 문서의 전체 텍스트와 비교해야 함. 
                    # 여기선 단순히 검색 점수의 평균을 활용하거나 1위 점수 활용
                
                # SemScore Proxy 보정 (검색 점수 기반)
                semscore_proxy = top_results[0][1] if top_results else 0.0
                
                rows.append({
                    "question_no": qa_row["question_no"],
                    "q_type": qa_row.get("q_type"),
                    "question": query,
                    "expected_source_ids": ", ".join(expected_ids),
                    "hit@k": hit,
                    "mrr": mrr,
                    "semscore_proxy": round(semscore_proxy, 4),
                    "top_source_titles": " | ".join(top_titles[:3])
                })
                
                if remote_min_interval_seconds > 0 and kind == "remote_dense":
                    time.sleep(remote_min_interval_seconds)
            
            res_df = pd.DataFrame(rows)
            detail_frames[name] = res_df
            
            elapsed = time.time() - cand_start
            summary_results.append({
                "candidate": name,
                "kind": kind,
                "avg_hit_rate": res_df["hit@k"].mean(),
                "avg_mrr": res_df["mrr"].mean(),
                "avg_semscore_proxy": res_df["semscore_proxy"].mean(),
                "num_questions": len(res_df),
                "elapsed_seconds": round(elapsed, 2),
                "note": config.get("note", "")
            })
            
        except Exception as exc:
            benchmark_errors.append({"candidate": name, "error": str(exc)})
            log_progress("Benchmark", f"Candidate {name} failed: {exc}")
            
    summary_df = pd.DataFrame(summary_results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["avg_semscore_proxy", "avg_hit_rate", "avg_mrr"], ascending=False).reset_index(drop=True)
        
    return summary_df, detail_frames, benchmark_errors
