from datetime import datetime, timezone
from ipaddress import ip_address
from socket import gethostbyname
from urllib.parse import urlparse
from uuid import uuid4

import openai
import requests
import weaviate
from bs4 import BeautifulSoup
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery

from services.config import Settings


RAG_COLLECTION = "FeedbackLoopDocuments"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 5
RAG_PROPERTIES = [
    Property(name="chunk_id", data_type=DataType.TEXT),
    Property(name="document_id", data_type=DataType.TEXT),
    Property(name="source_type", data_type=DataType.TEXT),
    Property(name="url", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="content", data_type=DataType.TEXT),
    Property(name="created_at", data_type=DataType.DATE),
]


class RagServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def weaviate_client(settings: Settings):
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def weaviate_status(settings: Settings) -> dict:
    client = weaviate_client(settings)
    try:
        return {"ready": client.is_ready(), "live": client.is_live()}
    finally:
        client.close()


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RagServiceError("網址必須是有效的 http 或 https URL。", 400)
    try:
        address = ip_address(gethostbyname(parsed.hostname))
    except OSError as error:
        raise RagServiceError("無法解析網址主機名稱。", 400) from error
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise RagServiceError("不允許讀取內網、保留或本機 IP 網址。", 400)
    return url


def load_web_page(url: str) -> dict:
    validate_public_url(url)
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "FeedbackLoop-AI/0.1"}, allow_redirects=False)
        if response.is_redirect:
            raise RagServiceError("不支援重新導向網址，請輸入最終公開網址。", 400)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RagServiceError("無法讀取此網址，請確認網址可公開存取。", 502) from error
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    content = soup.get_text(" ", strip=True)
    if not content:
        raise RagServiceError("網址頁面沒有可用的文字內容。", 400)
    title = soup.title.get_text(strip=True) if soup.title else url
    return {"url": url, "title": title, "content": content}


def chunk_text(content: str) -> list[str]:
    text = " ".join(content.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def embedding(text: str, settings: Settings) -> list[float]:
    client = openai.OpenAI(base_url=settings.embedding_base_url, api_key="EMPTY", timeout=settings.embedding_timeout)
    try:
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        vector = response.data[0].embedding
        if not isinstance(vector, list) or not vector:
            raise RagServiceError("Embedding 服務沒有回傳向量。", 503)
        return vector
    except Exception as error:
        raise RagServiceError("Embedding 服務目前無法使用。", 503) from error


def rag_collection(client):
    if not client.collections.exists(RAG_COLLECTION):
        client.collections.create(
            RAG_COLLECTION,
            vector_config=Configure.Vectors.self_provided(),
            properties=RAG_PROPERTIES,
        )
    collection = client.collections.get(RAG_COLLECTION)
    existing = {property.name for property in collection.config.get().properties}
    for property in RAG_PROPERTIES:
        if property.name not in existing:
            collection.config.add_property(property)
    return collection


def ingest_web_url(url: str, settings: Settings) -> dict:
    page = load_web_page(url)
    document_id = uuid4().hex
    chunks = chunk_text(page["content"])
    client = weaviate_client(settings)
    try:
        collection = rag_collection(client)
        for index, content in enumerate(chunks, start=1):
            collection.data.insert(
                properties={
                    "chunk_id": uuid4().hex,
                    "document_id": document_id,
                    "source_type": "web",
                    "url": page["url"],
                    "title": page["title"],
                    "chunk_index": index,
                    "content": content,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                vector=embedding(content, settings),
            )
    except RagServiceError:
        raise
    except Exception as error:
        raise RagServiceError("網址來源建立失敗，請確認 Weaviate 服務。", 503) from error
    finally:
        client.close()
    return {"id": document_id, "name": page["title"], "source_type": "web", "url": page["url"], "chunk_count": len(chunks)}


def retrieve_chunks(question: str, document_id: str, settings: Settings) -> list[dict]:
    client = weaviate_client(settings)
    try:
        response = rag_collection(client).query.near_vector(
            near_vector=embedding(question, settings),
            filters=Filter.by_property("document_id").equal(document_id),
            limit=TOP_K,
            return_metadata=MetadataQuery(distance=True),
        )
        return [{**item.properties, "score": 1 - (item.metadata.distance or 0)} for item in response.objects]
    except RagServiceError:
        raise
    except Exception as error:
        raise RagServiceError("無法從 Weaviate 取得相關內容。", 503) from error
    finally:
        client.close()


def answer_from_chunks(question: str, chunks: list[dict], settings: Settings) -> str:
    contexts = "\n\n".join(f"[{item['title']} | {item['url']}]\n{item['content']}" for item in chunks)
    prompt = (
        "請根據下列來源，以繁體中文清楚回答問題。\n"
        "若來源內容不足以完整回答，請務必明確回覆：「根據目前提供的資料，我無法回答這個問題。」，絕對不可以回傳空白。\n\n"
        f"問題：{question}\n\n來源：\n{contexts}"
    )
    client = openai.OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=settings.llm_timeout)
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        content = response.choices[0].message.content
        return content.strip()
    except openai.APITimeoutError as error:
        raise RagServiceError("模型回應逾時，請稍後再試。", 504) from error
    except Exception as error:
        raise RagServiceError("模型目前無法產生回答。", 503) from error


def answer_from_history(messages: list[dict], settings: Settings) -> str:
    client = openai.OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=settings.llm_timeout)
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        content = response.choices[0].message.content
        return content.strip()
    except openai.APITimeoutError as error:
        raise RagServiceError("模型回應逾時，請稍後再試。", 504) from error
    except Exception as error:
        raise RagServiceError("模型目前無法回應，請稍後再試。", 503) from error


def delete_document(document_id: str, settings: Settings) -> None:
    client = weaviate_client(settings)
    try:
        if client.collections.exists(RAG_COLLECTION):
            collection = client.collections.get(RAG_COLLECTION)
            collection.data.delete_many(Filter.by_property("document_id").equal(document_id))
    except Exception as error:
        raise RagServiceError("無法從 Weaviate 刪除內容。", 503) from error
    finally:
        client.close()