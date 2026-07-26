"""Elasticsearch-backed document storage for the Flask RAG application."""

from __future__ import annotations

import base64
import json
from typing import Any

import openai
import requests

from services.config import Settings

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


class RagServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.elasticsearch_api_key:
        headers["Authorization"] = f"ApiKey {settings.elasticsearch_api_key}"
    elif settings.elasticsearch_username and settings.elasticsearch_password:
        auth_string = f"{settings.elasticsearch_username}:{settings.elasticsearch_password}"
        encoded_auth = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
        headers["Authorization"] = f"Basic {encoded_auth}"
    return headers


def _url(settings: Settings, path: str) -> str:
    return f"{settings.elasticsearch_url.rstrip('/')}/{path.lstrip('/')}"


def embedding(text: str, settings: Settings) -> list[float]:
    client = openai.OpenAI(base_url=settings.embedding_base_url, api_key="EMPTY", timeout=settings.embedding_timeout)
    try:
        vector = client.embeddings.create(model=settings.embedding_model, input=text).data[0].embedding
    except openai.OpenAIError as error:
        raise RagServiceError("Embedding service is unavailable.", 503) from error
    if not vector:
        raise RagServiceError("Embedding service returned an empty vector.", 503)
    return vector


def ensure_index(settings: Settings) -> None:
    mapping = {
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "url": {"type": "keyword", "index": False},
                "title": {"type": "text"},
                "content": {"type": "text"},
                "embedding": {"type": "dense_vector", "index": False},
                "page_number": {"type": "integer"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "block_type": {"type": "keyword"},
                "field_code": {"type": "keyword"},
                "field_name": {"type": "text"},
                "field_english_name": {"type": "text"},
                "detail_type": {"type": "keyword"},
                "parent_chunk_id": {"type": "keyword"},
                "created_at": {"type": "date"},
            }
        }
    }
    try:
        response = requests.put(_url(settings, settings.elasticsearch_index), headers=_headers(settings), json=mapping, timeout=15)
        if response.status_code in {200, 201}:
            return
        if response.status_code == 400 and response.json().get("error", {}).get("type") == "resource_already_exists_exception":
            return
        response.raise_for_status()
    except requests.RequestException as error:
        raise RagServiceError("Elasticsearch service is unavailable.", 503) from error


def elasticsearch_status(settings: Settings) -> dict[str, bool]:
    try:
        response = requests.get(_url(settings, "_cluster/health"), headers=_headers(settings), timeout=5)
        response.raise_for_status()
        return {"ready": True, "live": True}
    except requests.RequestException:
        return {"ready": False, "live": False}


def index_chunks(document_id: str, chunks: list[dict[str, Any]], settings: Settings) -> None:
    ensure_index(settings)
    lines: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        lines.append(json.dumps({"index": {"_index": settings.elasticsearch_index, "_id": f"{document_id}:{chunk_id}"}}))
        lines.append(json.dumps({**chunk, "document_id": document_id, "embedding": embedding(str(chunk["content"]), settings)}, ensure_ascii=False))
    try:
        response = requests.post(_url(settings, "_bulk?refresh=wait_for"), headers={**_headers(settings), "Content-Type": "application/x-ndjson"}, data="\n".join(lines) + "\n", timeout=60)
        response.raise_for_status()
        if response.json().get("errors"):
            raise RagServiceError("Elasticsearch rejected one or more document chunks.", 503)
    except requests.RequestException as error:
        raise RagServiceError("Could not index document chunks in Elasticsearch.", 503) from error


def delete_document(document_id: str, settings: Settings) -> None:
    query = {"query": {"term": {"document_id": document_id}}}
    try:
        response = requests.post(_url(settings, f"{settings.elasticsearch_index}/_delete_by_query?refresh=true"), headers=_headers(settings), json=query, timeout=30)
        if response.status_code != 404:
            response.raise_for_status()
    except requests.RequestException as error:
        raise RagServiceError("Could not delete document chunks from Elasticsearch.", 503) from error


def search_chunks(query: str, document_id: str, settings: Settings, *, limit: int, hybrid: bool, block_types: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
    if block_types:
        filters.append({"terms": {"block_type": list(block_types)}})
    vector_query = {
        "script_score": {
            "query": {"bool": {"filter": filters}},
            "script": {"source": "cosineSimilarity(params.vector, 'embedding') + 1.0", "params": {"vector": embedding(query, settings)}},
        }
    }
    if hybrid:
        search_query: dict[str, Any] = {"bool": {"filter": filters, "should": [{"match": {"content": {"query": query, "boost": 1.0}}}], "minimum_should_match": 0}}
        body = {"size": limit, "query": {"script_score": {"query": search_query, "script": {"source": "cosineSimilarity(params.vector, 'embedding') + 1.0 + _score", "params": {"vector": vector_query["script_score"]["script"]["params"]["vector"]}}}}}
    else:
        body = {"size": limit, "query": vector_query}
    try:
        response = requests.post(_url(settings, f"{settings.elasticsearch_index}/_search"), headers=_headers(settings), json=body, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RagServiceError("Could not retrieve document chunks from Elasticsearch.", 503) from error
    return [{"uuid": hit["_id"], **hit["_source"], "score": hit.get("_score", 0.0)} for hit in response.json().get("hits", {}).get("hits", [])]
