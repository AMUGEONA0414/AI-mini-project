from __future__ import annotations

import json
import math
import os

from .clients import (
    call_huggingface_embeddings,
    call_jina_embeddings,
    call_openai_embeddings,
    call_voyage_embeddings,
)
from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_ROOT,
    EMBEDDING_CANDIDATES,
    EVAL_ROOT,
    MIN_WEB_TRUST_SCORE,
    OPENAI_EMBEDDING_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
    RAG_TOP_N_DOCS,
    RETRIEVAL_EVALSET_PATH,
    RETRIEVAL_STRATEGIES,
    TRUSTED_DOMAINS,
    log_progress,
    stable_hash,
)
from .sources import build_rag_search_text
from .text import compact_text, extract_domain, normalize_search_date, tokenize


def chunk_text(text: str, *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
    normalized = compact_text(text)
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunked_corpus(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    chunks = []
    for doc in documents:
        base_text = build_rag_search_text(doc)
        inferred_technology = infer_document_technology(doc)
        for index, chunk in enumerate(chunk_text(base_text), start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc['path']}#chunk-{index}",
                    "doc_path": doc["path"],
                    "technology": (doc["metadata"].get("technology") or inferred_technology).upper(),
                    "title": doc["metadata"].get("title", doc["name"]),
                    "text": chunk,
                }
            )
    log_progress("RAG", f"Chunked corpus built: {len(documents)} docs -> {len(chunks)} chunks")
    return chunks


def infer_document_technology(doc: dict[str, object]) -> str:
    title = str(doc["metadata"].get("title", doc["name"])).lower()
    body = str(doc.get("body", "")).lower()[:12000]
    haystack = f"{title} {body}"
    if "hbm4" in haystack or "next-gen hbm" in haystack or "hybrid bonding" in haystack:
        return "HBM4"
    if any(token in haystack for token in ["processing-in-memory", "pim", "aimx", "gddr6-aim", "aim "]):
        return "PIM"
    if any(token in haystack for token in ["compute express link", "cxl"]):
        return "CXL"
    return str(doc["metadata"].get("technology", ""))


def domain_trust_score(domain: str) -> float:
    domain = domain.lower()
    for trusted_domain, score in TRUSTED_DOMAINS.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            return score
    return 0.45


def has_recent_date(date_value: str) -> bool:
    import re
    return bool(date_value and "미상" not in date_value and re.match(r"20\d{2}-\d{2}-\d{2}$", date_value))


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def score_chunk_for_strategy(*, strategy: str, query_text: str, query_embedding: list[float] | None, chunk: dict[str, object], chunk_embedding: list[float] | None, technology: str) -> float:
    dense_score = 0.0
    lexical_score = lexical_overlap_score(query_text, str(chunk["text"]))
    if query_embedding is not None and chunk_embedding is not None:
        dense_score = cosine_similarity(query_embedding, chunk_embedding)
    tech_bonus = 0.2 if chunk["technology"] == technology else 0.0
    title_bonus = 0.12 if technology.lower() in str(chunk["title"]).lower() else 0.0
    intent_bonus = query_intent_bonus(query_text, str(chunk["text"]))
    if strategy == "dense":
        return dense_score + tech_bonus + title_bonus + intent_bonus
    if strategy == "lexical":
        return lexical_score + tech_bonus + title_bonus + intent_bonus
    return (0.75 * dense_score) + (0.25 * lexical_score) + tech_bonus + title_bonus + intent_bonus


def query_intent_bonus(query_text: str, chunk_text: str) -> float:
    query_lower = query_text.lower()
    chunk_lower = chunk_text.lower()
    bonus = 0.0
    if any(token in query_lower for token in ["overview", "survey", "primer", "introduction"]):
        if any(token in chunk_lower for token in ["overview", "survey", "primer", "introduction", "abstract"]):
            bonus += 0.08
    if any(token in query_lower for token in ["ecosystem", "standard", "direction"]):
        if any(token in chunk_lower for token in ["ecosystem", "standard", "future directions", "industry-standard"]):
            bonus += 0.08
    if any(token in query_lower for token in ["packaging", "thermal", "challenge"]):
        if any(token in chunk_lower for token in ["packaging", "thermal", "bonding", "heat"]):
            bonus += 0.08
    return bonus


def rank_documents_for_strategy(*, strategy: str, query_text: str, technology: str, chunked_corpus: list[dict[str, object]], chunk_embeddings: dict[str, list[float]], doc_lookup: dict[str, dict[str, object]], query_embedding: list[float] | None = None, limit: int = RAG_TOP_N_DOCS) -> list[dict[str, object]]:
    doc_scores: dict[str, float] = {}
    for chunk in chunked_corpus:
        if chunk["technology"] and chunk["technology"] != technology:
            continue
        chunk_score = score_chunk_for_strategy(
            strategy=strategy,
            query_text=query_text,
            query_embedding=query_embedding,
            chunk=chunk,
            chunk_embedding=chunk_embeddings.get(chunk["chunk_id"]),
            technology=technology,
        )
        doc = doc_lookup[chunk["doc_path"]]
        title = str(doc["metadata"].get("title", doc["name"]))
        title_score = lexical_overlap_score(query_text, title) * 0.6
        technology_score = 0.2 if infer_document_technology(doc).upper() == technology else 0.0
        total_score = chunk_score + title_score + technology_score
        current = doc_scores.get(chunk["doc_path"], float("-inf"))
        if total_score > current:
            doc_scores[chunk["doc_path"]] = total_score
    ranked_paths = sorted(doc_scores, key=doc_scores.get, reverse=True)[:limit]
    return [doc_lookup[path] for path in ranked_paths]


def build_retrieval_evalset() -> list[dict[str, object]]:
    return [
        {
            "query": "HBM4 packaging and thermal challenge overview",
            "technology": "HBM4",
            "expected_titles": [
                "Hybrid Bonding Expands from Logic to Memory_ SK Hynix, Applied Materials, BESI Drive Co-optimization to Scale Next-gen HBM",
                "HBM4",
            ],
        },
        {
            "query": "PIM survey and architecture research trends",
            "technology": "PIM",
            "expected_titles": ["2012.03112v5", "2105.03814v7"],
        },
        {
            "query": "CXL memory expansion ecosystem and standard direction",
            "technology": "CXL",
            "expected_titles": ["2306.11227v3", "2412.20249v2"],
        },
    ]


def ensure_retrieval_evalset() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    if not RETRIEVAL_EVALSET_PATH.exists():
        RETRIEVAL_EVALSET_PATH.write_text(json.dumps(build_retrieval_evalset(), ensure_ascii=False, indent=2), encoding="utf-8")


def embedding_cache_key(text: str, model: str, input_type: str) -> str:
    return stable_hash(f"{model}::{input_type}::{text}")


def embedding_provider_for_model(model: str) -> str:
    for candidate in EMBEDDING_CANDIDATES:
        if candidate["model"] == model:
            return candidate["provider"]
    return "openai"


def call_embedding_batch(texts: list[str], *, model: str, input_type: str = "document") -> list[list[float]]:
    provider = embedding_provider_for_model(model)
    if provider == "openai":
        return call_openai_embeddings(texts, model=model)
    if provider == "huggingface":
        return call_huggingface_embeddings(texts, model=model)
    if provider == "voyage":
        return call_voyage_embeddings(texts, model=model, input_type=input_type)
    if provider == "jina":
        return call_jina_embeddings(texts, model=model, input_type=input_type)
    raise RuntimeError(f"지원하지 않는 embedding provider: {provider} ({model})")


def get_embeddings(texts: list[str], *, model: str = OPENAI_EMBEDDING_MODEL, input_type: str = "document") -> dict[str, list[float]]:
    EMBEDDING_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[float]] = {}
    uncached_texts: list[str] = []
    uncached_paths = []
    for text in texts:
        cache_path = EMBEDDING_CACHE_ROOT / f"{embedding_cache_key(text, model, input_type)}.json"
        if cache_path.exists():
            results[text] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            uncached_texts.append(text)
            uncached_paths.append(cache_path)
    for start in range(0, len(uncached_texts), EMBEDDING_BATCH_SIZE):
        batch_texts = uncached_texts[start : start + EMBEDDING_BATCH_SIZE]
        batch_paths = uncached_paths[start : start + EMBEDDING_BATCH_SIZE]
        batch_embeddings = call_embedding_batch(batch_texts, model=model, input_type=input_type)
        for text, cache_path, embedding in zip(batch_texts, batch_paths, batch_embeddings):
            cache_path.write_text(json.dumps(embedding), encoding="utf-8")
            results[text] = embedding
    return results


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def compute_retrieval_metrics(documents: list[dict[str, object]], *, chunked_corpus: list[dict[str, object]] | None = None, chunk_embeddings: dict[str, list[float]] | None = None, embedding_model: str = OPENAI_EMBEDDING_MODEL) -> dict[str, object]:
    log_progress("RAG", "Computing retrieval metrics")
    ensure_retrieval_evalset()
    evalset = json.loads(RETRIEVAL_EVALSET_PATH.read_text(encoding="utf-8"))
    chunked_corpus = chunked_corpus or build_chunked_corpus(documents)
    doc_lookup = {doc["path"]: doc for doc in documents}
    if chunk_embeddings is None:
        embedding_map = get_embeddings([item["text"] for item in chunked_corpus], model=embedding_model, input_type="document")
        chunk_embeddings = {item["chunk_id"]: embedding_map[item["text"]] for item in chunked_corpus}
    query_texts = [f"{row['query']} {row['technology']}" for row in evalset]
    query_embedding_map = get_embeddings(query_texts, model=embedding_model, input_type="query")
    benchmarks: dict[str, object] = {}
    for strategy in RETRIEVAL_STRATEGIES:
        hits = 0
        reciprocal_sum = 0.0
        evaluated = 0
        for row in evalset:
            query_text = f"{row['query']} {row['technology']}"
            query_embedding = query_embedding_map[query_text] if strategy != "lexical" else None
            ranked_docs = rank_documents_for_strategy(strategy=strategy, query_text=query_text, technology=row["technology"], chunked_corpus=chunked_corpus, chunk_embeddings=chunk_embeddings, doc_lookup=doc_lookup, query_embedding=query_embedding, limit=RAG_TOP_K)
            evaluated += 1
            matched_rank = 0
            for rank, item in enumerate(ranked_docs, start=1):
                title = item["metadata"].get("title", item["name"])
                if any(expected in title for expected in row["expected_titles"]):
                    matched_rank = rank
                    break
            if matched_rank:
                hits += 1
                reciprocal_sum += 1.0 / matched_rank
        benchmarks[strategy] = {"hit_rate_at_k": round(hits / evaluated, 3) if evaluated else 0.0, "mrr": round(reciprocal_sum / evaluated, 3) if evaluated else 0.0, "queries": evaluated, "k": RAG_TOP_K}
    selected_strategy = max(RETRIEVAL_STRATEGIES, key=lambda name: (benchmarks[name]["mrr"], benchmarks[name]["hit_rate_at_k"]))
    metrics = dict(benchmarks[selected_strategy])
    metrics["strategy"] = selected_strategy
    metrics["embedding_model"] = embedding_model
    metrics["benchmarks"] = benchmarks
    log_progress("RAG", "Retrieval metrics ready: " + ", ".join(f"{name}=HitRate@{benchmarks[name]['k']} {benchmarks[name]['hit_rate_at_k']}, MRR {benchmarks[name]['mrr']}" for name in RETRIEVAL_STRATEGIES) + f" | selected={selected_strategy}")
    return metrics


def available_embedding_candidates() -> list[dict[str, object]]:
    candidates = []
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_hf = bool(os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"))
    has_voyage = bool(os.getenv("VOYAGE_API_KEY"))
    has_jina = bool(os.getenv("JINA_API_KEY"))
    for item in EMBEDDING_CANDIDATES:
        if item.get("status") != "active":
            continue
        provider = item["provider"]
        if provider == "openai" and has_openai:
            candidates.append(item)
        if provider == "huggingface" and has_hf:
            candidates.append(item)
        if provider == "voyage" and has_voyage:
            candidates.append(item)
        if provider == "jina" and has_jina:
            candidates.append(item)
    return candidates


def benchmark_embedding_models(*, documents: list[dict[str, object]], chunked_corpus: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    log_progress("RAG", "Benchmarking embedding candidates")
    candidates = available_embedding_candidates()
    if not candidates:
        raise RuntimeError("실행 가능한 embedding candidate가 없습니다.")
    benchmarks: dict[str, object] = {}
    best_model = OPENAI_EMBEDDING_MODEL
    best_key = (-1.0, -1.0)
    for candidate in candidates:
        model = candidate["model"]
        try:
            chunk_embedding_map = get_embeddings([chunk["text"] for chunk in chunked_corpus], model=model, input_type="document")
            chunk_embeddings = {chunk["chunk_id"]: chunk_embedding_map[chunk["text"]] for chunk in chunked_corpus}
            metrics = compute_retrieval_metrics(documents, chunked_corpus=chunked_corpus, chunk_embeddings=chunk_embeddings, embedding_model=model)
            benchmarks[model] = {"provider": candidate["provider"], "strategy": metrics["strategy"], "hit_rate_at_k": metrics["hit_rate_at_k"], "mrr": metrics["mrr"], "strategy_benchmarks": metrics["benchmarks"]}
            score_key = (metrics["mrr"], metrics["hit_rate_at_k"])
            if score_key > best_key:
                best_key = score_key
                best_model = model
        except Exception as exc:
            benchmarks[model] = {"provider": candidate["provider"], "error": str(exc)}
            log_progress("RAG", f"Embedding benchmark failed for {model}: {exc}")
    log_progress("RAG", "Embedding benchmarks ready: " + ", ".join((f"{model}=MRR {item['mrr']} HitRate@{RAG_TOP_K} {item['hit_rate_at_k']}" if "mrr" in item else f"{model}=failed") for model, item in benchmarks.items()) + f" | selected={best_model}")
    return best_model, benchmarks


def score_document(doc: dict[str, object], query: str, tech: str, company: str | None = None) -> int:
    haystack = " ".join([doc["metadata"].get("title", ""), doc["metadata"].get("technology", ""), doc["metadata"].get("company", ""), doc["body"]]).lower()
    score = 0
    for token in set(tokenize(query)):
        if token in haystack:
            score += 2
    if tech.lower() in haystack:
        score += 8
    if company and company.lower() in haystack:
        score += 8
    return score


def score_web_result(result: dict[str, object], *, query: str, company: str, tech: str, content: str) -> float:
    domain = extract_domain(result.get("url", ""))
    trust = domain_trust_score(domain)
    lexical = lexical_overlap_score(f"{query} {company} {tech}", content or result.get("title", ""))
    date_bonus = 0.08 if has_recent_date(normalize_search_date(result, url=result.get("url", ""), content=content)) else 0.0
    return round(trust + lexical + date_bonus, 3)


def is_usable_web_result(result: dict[str, object], *, trust_score: float, content: str) -> bool:
    if trust_score < MIN_WEB_TRUST_SCORE or len(content) < 180:
        return False
    domain = extract_domain(result.get("url", ""))
    if any(domain.endswith(noisy) for noisy in {"youtube.com", "twitter.com", "x.com", "facebook.com", "instagram.com"}):
        return False
    return True


def classify_trl(combined_text: str, evidence_count: int, source_types: list[str], signal_count: int) -> tuple[int, str, list[str]]:
    lowered = combined_text.lower()
    reasons: list[str] = []
    score = 3.8
    if any(key in lowered for key in ["customer validation", "qualification", "reliability qualification", "system demo", "partner demo", "sample shipment"]):
        reasons.append("고객 검증 또는 시스템 시연 신호")
        score += 1.4
    if any(key in lowered for key in ["prototype", "bring-up", "interoperability", "characterization", "sample evaluation", "pre-qualification"]):
        reasons.append("시제품 또는 평가 보드 단계 신호")
        score += 0.8
    if "academic_pdf" in source_types:
        score += 0.2
    if "tavily_search_result" in source_types:
        score += 0.3
    if evidence_count >= 3:
        score += 0.4
    if signal_count >= 3:
        score += 0.3
    trl = 4 if score < 4.8 else (5 if score < 5.8 else 6)
    confidence = "high" if evidence_count >= 3 and signal_count >= 2 else ("medium" if evidence_count >= 2 else "low")
    if not reasons:
        reasons.append("개념·부품 수준 공개 신호 중심")
    return trl, confidence, reasons
