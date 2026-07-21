"""Ask questions over PDF chunks stored in Weaviate, then ground an LLM answer in them.

Example:
    uv run python ask_weaviate.py "這份文件的重點是什麼？" --show-sources

Required environment variables (usually placed in .env):
    EMBEDDING_URL=http://localhost:11434/api/embed
    EMBEDDING_MODEL=nomic-embed-text
    LLM_URL=http://localhost:11434/api/chat
    LLM_MODEL=qwen2.5:7b

``--retrieval hybrid`` is the default.  It combines Weaviate's BM25 keyword
score with the supplied embedding vector, which works better than vector-only
search when a question contains exact terms, numbers, or names.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from weaviate_pdf_chunks import DEFAULT_COLLECTION, connect_weaviate, embed_texts


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    filename: str
    page_number: int | None
    chunk_id: str
    distance: float | None = None
    score: float | None = None

    @property
    def citation(self) -> str:
        page = f"第 {self.page_number} 頁" if self.page_number else "頁碼未知"
        return f"{self.filename}，{page}"


def retrieve_chunks(
    question: str,
    *,
    collection_name: str,
    document_id: str | None,
    user_id: str | None,
    limit: int,
    retrieval: str,
    alpha: float,
) -> list[RetrievedChunk]:
    """Query Weaviate using vector-only or its hybrid BM25 + vector search."""
    base_url = os.getenv("EMBEDDING_URL") or os.getenv("EMBEDDING_BASE_URL") or os.environ["OLLAMA_BASE_URL"]
    embedding_model = os.environ["EMBEDDING_MODEL"]
    timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
    vector = embed_texts([question], base_url=base_url, model=embedding_model, timeout=timeout)[0]

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
        if retrieval == "hybrid":
            response = collection.query.hybrid(
                query=question,
                vector=vector,
                alpha=alpha,
                limit=limit,
                filters=filters,
                return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
            )
        else:
            response = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                filters=filters,
                return_metadata=MetadataQuery(distance=True),
            )
        return [
            RetrievedChunk(
                content=str(item.properties.get("content") or ""),
                filename=str(item.properties.get("filename") or "未知文件"),
                page_number=item.properties.get("page_number"),
                chunk_id=str(item.properties.get("chunk_id") or ""),
                distance=getattr(item.metadata, "distance", None),
                score=getattr(item.metadata, "score", None),
            )
            for item in response.objects
            if item.properties.get("content")
        ]
    finally:
        client.close()


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[來源 {index}] {chunk.citation}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "你是以檢索內容為依據的助理。只根據提供的來源回答；"
                "若來源不足，直接說明無法從文件確認。以繁體中文回答，"
                "並在每個重要陳述後標註 [來源編號]。不要編造來源。"
            ),
        },
        {"role": "user", "content": f"問題：{question}\n\n檢索來源：\n{context}"},
    ]


def ask_llm(messages: list[dict[str, str]], *, base_url: str, model: str, timeout: float) -> str:
    """Call either an OpenAI-compatible chat endpoint or Ollama's native API."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = f"{root}/chat/completions"
    if root.endswith("/v1/chat/completions"):
        response = requests.post(
            root,
            json={"model": model, "messages": messages, "temperature": 0.2},
            timeout=timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    if not root.endswith("/api/chat"):
        root = f"{root}/api/chat"
    response = requests.post(
        root,
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2}},
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["message"]["content"]).strip()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieve PDF chunks from Weaviate and ask an LLM to answer.")
    parser.add_argument("question", help="要詢問文件的問題")
    parser.add_argument("--collection", default=os.getenv("WEAVIATE_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--document-id", help="只檢索此文件 ID")
    parser.add_argument("--user-id", help="只檢索此使用者的資料")
    parser.add_argument("--limit", type=int, default=5, help="送入 LLM 的來源片段數（預設 5）")
    parser.add_argument("--retrieval", choices=("hybrid", "vector"), default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.65, help="hybrid 的向量權重，0=BM25、1=純向量")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL"), help="覆蓋 .env 的 LLM_MODEL")
    parser.add_argument("--show-sources", action="store_true", help="列出檢索片段與分數")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit 必須大於 0")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha 必須介於 0 到 1")
    if not args.llm_model:
        parser.error("請在 .env 設定 LLM_MODEL，或使用 --llm-model")

    try:
        chunks = retrieve_chunks(
            args.question,
            collection_name=args.collection,
            document_id=args.document_id,
            user_id=args.user_id,
            limit=args.limit,
            retrieval=args.retrieval,
            alpha=args.alpha,
        )
        if not chunks:
            print("找不到相關來源，請確認 collection、篩選條件或是否已匯入資料。")
            return
        if args.show_sources:
            print("檢索來源：")
            for index, chunk in enumerate(chunks, start=1):
                metric = f"score={chunk.score:.4f}" if chunk.score is not None else f"distance={chunk.distance:.4f}"
                preview = " ".join(chunk.content.split())[:180]
                print(f"[{index}] {chunk.citation} ({metric})\n    {preview}")
            print()

        answer = ask_llm(
            build_messages(args.question, chunks),
            base_url=os.getenv("LLM_URL") or os.getenv("LLM_BASE_URL") or os.environ["OLLAMA_BASE_URL"],
            model=args.llm_model,
            timeout=float(os.getenv("LLM_TIMEOUT", "180")),
        )
        print(answer)
    except (KeyError, requests.RequestException, RuntimeError) as error:
        print(f"執行失敗：{error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
