from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .config import (
    EMBEDDING_CANDIDATES,
    EMBEDDING_MODEL,
    ENV_CANDIDATES,
    OPENAI_MODEL,
    TAVILY_API_URL,
    log_progress,
    retry_with_backoff,
)
from .text import strip_markdown_fence


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return api_key


def require_tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY가 설정되어 있지 않습니다.")
    return api_key


def require_huggingface_api_key() -> str:
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN 또는 HF_TOKEN이 설정되어 있지 않습니다.")
    return api_key


def require_voyage_api_key() -> str:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY가 설정되어 있지 않습니다.")
    return api_key


def require_jina_api_key() -> str:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY가 설정되어 있지 않습니다.")
    return api_key


def load_env_file() -> None:
    for env_path in ENV_CANDIDATES:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_env_file()


def call_openai_chat(*, system_prompt: str, user_prompt: str, temperature: float = 0.2, response_format: dict[str, Any] | None = None) -> str:
    log_progress("LLM", f"OpenAI chat request start: model={OPENAI_MODEL}")
    payload: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {require_openai_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenAI API 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("LLM", _request, retries=3, base_delay=2.0)
    content = strip_markdown_fence(data["choices"][0]["message"]["content"])
    log_progress("LLM", "OpenAI chat request completed")
    return content


def call_openai_json(*, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return json.loads(
        call_openai_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0, response_format={"type": "json_object"})
    )


def call_openai_embeddings(texts: list[str], *, model: str = EMBEDDING_MODEL) -> list[list[float]]:
    if not texts:
        return []
    log_progress("Embedding", f"Batch embedding request start: model={model} count={len(texts)}")
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={"Authorization": f"Bearer {require_openai_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenAI Embedding API 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("Embedding", _request, retries=3, base_delay=2.0)
    log_progress("Embedding", "Batch embedding request completed")
    return [item["embedding"] for item in data["data"]]


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    pooled = [0.0] * dim
    for vector in vectors:
        for idx, value in enumerate(vector):
            pooled[idx] += float(value)
    return [value / len(vectors) for value in pooled]


def _normalize_hf_embedding_output(data: Any, batch_size: int) -> list[list[float]]:
    if batch_size == 1:
        if data and isinstance(data, list) and isinstance(data[0], (int, float)):
            return [[float(value) for value in data]]
        if data and isinstance(data, list) and isinstance(data[0], list):
            if data[0] and isinstance(data[0][0], (int, float)):
                return [_mean_pool([[float(value) for value in row] for row in data])]
            return [[float(value) for value in row] for row in data]
        raise RuntimeError("Hugging Face embedding 응답 형식을 해석할 수 없습니다.")
    if data and isinstance(data, list) and isinstance(data[0], list):
        if data[0] and isinstance(data[0][0], (int, float)):
            return [[float(value) for value in row] for row in data]
        if data[0] and isinstance(data[0][0], list):
            return [_mean_pool([[float(value) for value in row] for row in item]) for item in data]
    raise RuntimeError("Hugging Face batch embedding 응답 형식을 해석할 수 없습니다.")


def call_huggingface_embeddings(texts: list[str], *, model: str) -> list[list[float]]:
    if not texts:
        return []
    log_progress("Embedding", f"HF embedding request start: model={model} count={len(texts)}")
    request = urllib.request.Request(
        f"https://router.huggingface.co/hf-inference/models/{model}",
        data=json.dumps({"inputs": texts if len(texts) > 1 else texts[0], "normalize": True}).encode("utf-8"),
        headers={"Authorization": f"Bearer {require_huggingface_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> Any:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Hugging Face Inference 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("Embedding", _request, retries=3, base_delay=2.0)
    log_progress("Embedding", "HF embedding request completed")
    return _normalize_hf_embedding_output(data, len(texts))


def call_voyage_embeddings(texts: list[str], *, model: str, input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []
    log_progress("Embedding", f"Voyage embedding request start: model={model} count={len(texts)}")
    request = urllib.request.Request(
        "https://api.voyageai.com/v1/embeddings",
        data=json.dumps({"model": model, "input": texts, "input_type": input_type}).encode("utf-8"),
        headers={"Authorization": f"Bearer {require_voyage_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Voyage API 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("Embedding", _request, retries=3, base_delay=2.0)
    log_progress("Embedding", "Voyage embedding request completed")
    return [item["embedding"] for item in data["data"]]


def call_jina_embeddings(texts: list[str], *, model: str, input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []
    log_progress("Embedding", f"Jina embedding request start: model={model} count={len(texts)}")
    request = urllib.request.Request(
        "https://api.jina.ai/v1/embeddings",
        data=json.dumps({"model": model, "task": "text-matching", "late_chunking": False, "input": texts}).encode("utf-8"),
        headers={"Authorization": f"Bearer {require_jina_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Jina API 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("Embedding", _request, retries=3, base_delay=2.0)
    log_progress("Embedding", "Jina embedding request completed")
    return [item["embedding"] for item in data["data"]]


def call_tavily_search(*, query: str, topic: str = "general", days: int = 180, max_results: int = 2) -> list[dict[str, Any]]:
    log_progress("WebSearch", f"Tavily query start: {query}")
    request = urllib.request.Request(
        TAVILY_API_URL,
        data=json.dumps({
            "api_key": require_tavily_api_key(),
            "query": query,
            "topic": topic,
            "days": days,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    def _request() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Tavily API 호출 실패: {exc.code} {exc.read().decode('utf-8', errors='ignore')}") from exc
    data = retry_with_backoff("WebSearch", _request, retries=2, base_delay=1.5)
    results = data.get("results", [])
    log_progress("WebSearch", f"Tavily query completed: {len(results)} results")
    return results
