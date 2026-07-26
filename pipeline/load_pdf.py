from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from pypdf import PdfReader
from services.config import Settings
from services.vectordb import RagServiceError, index_chunks

def ragflow_plain_sections(pdf_path: Path) -> list[dict]:
    """The RAGFlow PlainParser behavior: extract each page into text lines."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as error:
        raise RagServiceError("PDF parsing failed.", 400) from error

    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise RagServiceError(f"Could not extract text from PDF page {page_number}.", 400) from error
        sections.extend({"page_number": page_number, "content": line.strip()} for line in text.splitlines() if line.strip())
    return sections


def _token_count(text: str) -> int:
    # RAGFlow merges by tokens; whitespace words are a safe local approximation
    # that does not require its Infinity tokenizer runtime.
    return len(text.split())


def ragflow_naive_merge(sections: list[dict], chunk_token_num: int) -> list[dict]:
    """Merge consecutive PlainParser sections using RAGFlow naive chunk semantics."""
    chunks, current, token_count = [], [], 0
    for section in sections:
        content = section["content"]
        section_tokens = _token_count(content)
        if current and token_count + section_tokens > chunk_token_num:
            chunks.append({
                "page_number": current[0]["page_number"],
                "page_start": current[0]["page_number"],
                "page_end": current[-1]["page_number"],
                "content": "\n".join(item["content"] for item in current),
            })
            current, token_count = [], 0
        current.append(section)
        token_count += section_tokens
    if current:
        chunks.append({
            "page_number": current[0]["page_number"],
            "page_start": current[0]["page_number"],
            "page_end": current[-1]["page_number"],
            "content": "\n".join(item["content"] for item in current),
        })
    return chunks


def parse_pdf(pdf_path: Path, *, chunk_token_num: int | None = None) -> dict:
    token_limit = chunk_token_num or int(os.getenv("RAG_CHUNK_TOKEN_NUM", "512"))
    if token_limit <= 0:
        raise ValueError("RAG_CHUNK_TOKEN_NUM must be positive.")
    sections = ragflow_plain_sections(pdf_path)
    return {"source": str(pdf_path), "sections": sections, "chunks": ragflow_naive_merge(sections, token_limit)}


def write_pdf_chunk_report(report: dict, report_dir: Path, document_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{document_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def ingest_pdf(pdf_path: Path, *, document_id: str, filename: str, settings: Settings, report_dir: Path | None = None) -> dict:
    report = parse_pdf(pdf_path)
    chunks = report["chunks"]
    if not chunks:
        raise RagServiceError("No readable text was found in this PDF.", 400)
    if report_dir:
        write_pdf_chunk_report(report, report_dir, document_id)
    index_chunks(document_id, [
        {
            "chunk_id": uuid4().hex,
            "source_type": "pdf",
            "url": "",
            "title": filename,
            "chunk_index": index,
            "block_type": "general_text",
            "field_code": "",
            "field_name": "",
            "field_english_name": "",
            "detail_type": "",
            "parent_chunk_id": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ], settings)
    return {"source_type": "pdf", "chunk_count": len(chunks), "processed_pages": len({item["page_number"] for item in report["sections"]}), "ocr_pages": 0}
