from datetime import datetime, timezone
from ipaddress import ip_address
from socket import gethostbyname
from urllib.parse import urlparse
from uuid import uuid4
import requests
from bs4 import BeautifulSoup
from services.config import Settings
from services.vectordb import (CHUNK_OVERLAP,CHUNK_SIZE,RagServiceError,embedding,rag_collection,weaviate_client,)

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