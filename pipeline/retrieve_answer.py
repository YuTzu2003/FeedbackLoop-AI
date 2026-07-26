from __future__ import annotations
import logging
import re
from collections import defaultdict
import openai
import requests
from services.api import LLMSettings, get_rag_prompt, llm_client
from services.config import Settings
from services.vectordb import RagServiceError, search_chunks

def unique_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for query in queries:
        normalized = query.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(query.strip())
    return result

def build_search_queries(question: str, llm_settings: LLMSettings) -> list[str]:
    prompt = "Rewrite this question into at most three short document retrieval queries. Return one query per line; do not answer.\n\n" + question
    try:
        response = llm_client(llm_settings).chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        rewrites = (response.choices[0].message.content or "").splitlines()
    except openai.OpenAIError:
        rewrites = []
    return unique_queries([question, *(re.sub(r"^(?:[-*]|\d+[.)])\s*", "", item).strip() for item in rewrites)])[:4]

def reciprocal_rank_fusion(result_groups: list[list[dict]], k: int = 60) -> list[dict]:
    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, dict] = {}
    for group in result_groups:
        for rank, document in enumerate(group, start=1):
            key = str(document.get("uuid") or document.get("chunk_id"))
            scores[key] += 1 / (k + rank)
            documents[key] = document
    return [{**documents[key], "rrf_score": score} for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

def rerank_documents(query: str, candidates: list[dict], settings: Settings) -> list[dict]:
    if not settings.reranker_base_url or not candidates:
        return candidates
    selected = [candidate for candidate in candidates if str(candidate.get("content", "")).strip()][:settings.rerank_top_k]
    if not selected:
        return candidates
    endpoint = settings.reranker_base_url.rstrip("/")
    if not endpoint.endswith("/rerank"):
        endpoint += "/rerank"
    try:
        response = requests.post(endpoint, json={
            "model": settings.reranker_model or "jina-reranker",
            "query": query[:settings.reranker_max_query_chars],
            "documents": [str(candidate["content"])[:settings.reranker_max_document_chars] for candidate in selected],
            "top_n": len(selected),
        }, timeout=15)
        response.raise_for_status()
        results = response.json().get("results", [])
        ranked = []
        for result in sorted(results, key=lambda item: item.get("relevance_score", 0), reverse=True):
            index = result.get("index")
            if isinstance(index, int) and 0 <= index < len(selected):
                ranked.append({**selected[index], "rerank_score": result.get("relevance_score", 0)})
        return ranked or candidates
    except requests.HTTPError as error:
        body = error.response.text[:500] if error.response is not None else ""
        logging.getLogger(__name__).warning("Reranker rejected the request: %s; response=%s", error, body)
        return candidates
    except requests.RequestException as error:
        logging.getLogger(__name__).warning("Reranker unavailable: %s", error)
        return candidates

def retrieve_chunks(question: str, document_id: str, settings: Settings, llm_settings: LLMSettings, search_mode: str) -> list[dict]:
    queries = build_search_queries(question, llm_settings)
    groups = [search_chunks(query, document_id, settings, limit=settings.retrieval_top_k, hybrid=search_mode == "hybrid") for query in queries]
    candidates = reciprocal_rank_fusion(groups)[:settings.rrf_candidate_top_k]
    return rerank_documents(question, candidates, settings)[:settings.final_context_top_k]

def answer_from_chunks(question: str, chunks: list[dict], llm_settings: LLMSettings, personal_instruction: str = "") -> str:
    contexts = "\n\n".join(f"[{item.get('title', '')} | {item.get('url', '')}]\n{item['content']}" for item in chunks)
    try:
        response = llm_client(llm_settings).chat.completions.create(
            model=llm_settings.model,
            messages=[{"role": "user", "content": get_rag_prompt(question, contexts, personal_instruction)}],
            temperature=llm_settings.temperature,
        )
        return (response.choices[0].message.content or "").strip()
    except openai.OpenAIError as error:
        raise RagServiceError(f"LLM request failed: {error}", 500) from error

def answer_from_history(messages: list[dict], llm_settings: LLMSettings) -> str:
    try:
        response = llm_client(llm_settings).chat.completions.create(model=llm_settings.model, messages=messages, temperature=llm_settings.temperature)
        return (response.choices[0].message.content or "").strip()
    except openai.OpenAIError as error:
        raise RagServiceError(f"LLM request failed: {error}", 500) from error
