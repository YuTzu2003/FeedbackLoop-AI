from datetime import datetime, timezone
from ipaddress import ip_address
from socket import gethostbyname
from urllib.parse import urlparse
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from services.config import Settings
from services.vectordb import CHUNK_OVERLAP, CHUNK_SIZE, RagServiceError, delete_document, index_chunks


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RagServiceError("A public http or https URL is required.", 400)
    try:
        address = ip_address(gethostbyname(parsed.hostname))
    except OSError as error:
        raise RagServiceError("The URL host could not be resolved.", 400) from error
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        raise RagServiceError("Private network URLs are not allowed.", 400)
    return url


def load_web_page(url: str) -> dict:
    validate_public_url(url)
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "FeedbackLoop-AI/0.1"}, allow_redirects=False)
        if response.is_redirect:
            raise RagServiceError("Redirecting URLs are not supported.", 400)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RagServiceError("The web page could not be loaded.", 502) from error
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    content = soup.get_text(" ", strip=True)
    if not content:
        raise RagServiceError("No readable text was found at this URL.", 400)
    return {"url": url, "title": soup.title.get_text(strip=True) if soup.title else url, "content": content}


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
    return chunks


def ingest_web_url(url: str, settings: Settings) -> dict:
    page = load_web_page(url)
    document_id = uuid4().hex
    chunks = chunk_text(page["content"])
    try:
        index_chunks(document_id, [
            {
                "chunk_id": uuid4().hex,
                "source_type": "web",
                "url": page["url"],
                "title": page["title"],
                "chunk_index": index,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for index, content in enumerate(chunks, start=1)
        ], settings)
    except Exception:
        try:
            delete_document(document_id, settings)
        except RagServiceError:
            pass
        raise
    return {"id": document_id, "name": page["title"], "source_type": "web", "url": page["url"], "chunk_count": len(chunks)}
