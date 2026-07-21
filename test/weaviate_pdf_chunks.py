"""Import RAGFlow-style PDF chunks into Weaviate and query them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_COLLECTION = "FeedbackLoopDocuments"


def document_id_for_report(report: dict[str, Any]) -> str:
    source = str(report.get("source") or "unknown")
    pages = str(report.get("total_pages") or report.get("processed_pages") or "")
    digest = hashlib.sha256(f"{source}:{pages}".encode("utf-8")).hexdigest()[:16]
    return f"pdf_{digest}"


def load_chunks(report_path: Path, *, document_id: str | None = None, user_id: str = "local") -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved_document_id = document_id or document_id_for_report(report)
    filename = Path(str(report.get("source") or report_path)).name
    created_at = datetime.now(UTC).isoformat()
    chunks = []

    for index, chunk in enumerate(report.get("chunks", []), start=1):
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        chunks.append(
            {
                "chunk_id": str(chunk.get("chunk_id") or f"chunk_{index:05d}"),
                "document_id": resolved_document_id,
                "user_id": user_id,
                "source_type": "pdf",
                "content": content,
                "filename": filename,
                "page_number": int(chunk.get("page_number") or 0),
                "sheet_name": "",
                "row_start": 0,
                "row_end": 0,
                "url": "",
                "title": "",
                "chunk_index": index,
                "layout_types": list(chunk.get("layout_types") or []),
                "source_blocks_json": json.dumps(chunk.get("source_blocks") or [], ensure_ascii=False),
                "created_at": created_at,
            }
        )
    return chunks


# def embed_texts(texts: list[str], *, base_url: str, model: str, timeout: float = 120) -> list[list[float]]:
#     if not texts:
#         return []

#     url = f"{base_url.rstrip('/')}/api/embed"
    
#     response = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
#     if response.ok:
#         payload = response.json()
#         embeddings = payload.get("embeddings")
#         if isinstance(embeddings, list) and len(embeddings) == len(texts):
#             return embeddings

#     fallback_url = f"{base_url.rstrip('/')}/api/embeddings"
#     embeddings = []
#     for text in texts:
#         fallback_response = requests.post(fallback_url, json={"model": model, "prompt": text}, timeout=timeout)
#         fallback_response.raise_for_status()
#         payload = fallback_response.json()
#         embedding = payload.get("embedding")
#         if not isinstance(embedding, list):
#             raise RuntimeError("Ollama embedding response did not contain an embedding list")
#         embeddings.append(embedding)
#     return embeddings

def embed_texts(texts: list[str], *, base_url: str, model: str, timeout: float = 120) -> list[list[float]]:
    """Embed text using a configured endpoint or a backward-compatible base URL."""
    if not texts:
        return []
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = f"{url}/embeddings"
    elif not url.endswith(("/v1/embeddings", "/api/embed", "/api/embeddings")):
        url = f"{url}/api/embed"

    response = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
    if not response.ok:
        response.raise_for_status()
        
    payload = response.json()
    if "data" in payload:
        embeddings = [item["embedding"] for item in sorted(payload["data"], key=lambda item: item.get("index", 0))]
    else:
        embeddings = payload.get("embeddings", [])
            
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding 回傳數量 ({len(embeddings)}) 與文本數量 ({len(texts)}) 不符")
        
    return embeddings

def connect_weaviate():
    import weaviate

    host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
    http_port = int(os.getenv("WEAVIATE_PORT") or os.getenv("WEAVIATE_HTTP_PORT") or "8080")
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    return weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)


def ensure_collection(client, collection_name: str):
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(collection_name):
        return client.collections.get(collection_name)
    return client.collections.create(
        collection_name,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="user_id", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.INT),
            Property(name="sheet_name", data_type=DataType.TEXT),
            Property(name="row_start", data_type=DataType.INT),
            Property(name="row_end", data_type=DataType.INT),
            Property(name="url", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="layout_types", data_type=DataType.TEXT_ARRAY),
            Property(name="source_blocks_json", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.DATE),
        ],
    )


def import_chunks(
    report_path: Path,
    *,
    collection_name: str,
    document_id: str | None,
    user_id: str,
    batch_size: int,
) -> tuple[int, str]:
    load_dotenv()
    base_url = os.getenv("EMBEDDING_URL") or os.getenv("EMBEDDING_BASE_URL") or os.environ["OLLAMA_BASE_URL"]
    model = os.environ["EMBEDDING_MODEL"]
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
    chunks = load_chunks(report_path, document_id=document_id, user_id=user_id)
    resolved_document_id = chunks[0]["document_id"] if chunks else (document_id or "")

    client = connect_weaviate()
    try:
        collection = ensure_collection(client, collection_name)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embed_texts([item["content"] for item in batch], base_url=base_url, model=model, timeout=timeout)
            with collection.batch.fixed_size(batch_size=batch_size) as weaviate_batch:
                for properties, vector in zip(batch, vectors, strict=True):
                    weaviate_batch.add_object(properties=properties, vector=vector)
        return len(chunks), resolved_document_id
    finally:
        client.close()


def search_chunks(
    question: str,
    *,
    collection_name: str,
    document_id: str | None,
    user_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    load_dotenv()
    base_url = os.getenv("EMBEDDING_URL") or os.getenv("EMBEDDING_BASE_URL") or os.environ["OLLAMA_BASE_URL"]
    model = os.environ["EMBEDDING_MODEL"]
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
    query_vector = embed_texts([question], base_url=base_url, model=model, timeout=timeout)[0]

    from weaviate.classes.query import Filter, MetadataQuery

    filters = None
    if document_id:
        filters = Filter.by_property("document_id").equal(document_id)
    if user_id:
        user_filter = Filter.by_property("user_id").equal(user_id)
        filters = user_filter if filters is None else filters & user_filter

    client = connect_weaviate()
    try:
        collection = client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for item in response.objects:
            results.append(
                {
                    "distance": item.metadata.distance,
                    "chunk_id": item.properties.get("chunk_id"),
                    "document_id": item.properties.get("document_id"),
                    "filename": item.properties.get("filename"),
                    "page_number": item.properties.get("page_number"),
                    "content": item.properties.get("content"),
                }
            )
        return results
    finally:
        client.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Import and search PDF chunks in Weaviate.")
    parser.add_argument("--collection", default=os.getenv("WEAVIATE_COLLECTION", DEFAULT_COLLECTION))
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("report", type=Path)
    import_parser.add_argument("--document-id")
    import_parser.add_argument("--user-id", default="local")
    import_parser.add_argument("--batch-size", type=int, default=16)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("question")
    search_parser.add_argument("--document-id")
    search_parser.add_argument("--user-id")
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()
    if args.command == "import":
        count, document_id = import_chunks(
            args.report,
            collection_name=args.collection,
            document_id=args.document_id,
            user_id=args.user_id,
            batch_size=args.batch_size,
        )
        print(f"Imported {count} chunks into {args.collection}; document_id={document_id}")
    else:
        results = search_chunks(
            args.question,
            collection_name=args.collection,
            document_id=args.document_id,
            user_id=args.user_id,
            limit=args.limit,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
