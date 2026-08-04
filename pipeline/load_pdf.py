from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import camelot
from pypdf import PdfReader
from services.config import Settings
from services.vectordb import RagServiceError, index_chunks

def _markdown_table(rows: list[list[str]]) -> str:
    escaped_rows = [[cell.replace("|", "\\|").replace("\n", "<br>") for cell in row] for row in rows]
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in (escaped_rows[0], ["---"] * len(escaped_rows[0]), *escaped_rows[1:]))

def camelot_table_sections(pdf_path: Path) -> list[dict]:
    tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="hybrid")
    sections: list[dict] = []
    previous_rows: list[list[str]] | None = None
    previous_page: int | None = None
    table_start_page: int | None = None
    for table in tables:
        rows = [[str(value).strip() for value in row] for row in table.df.fillna("").values.tolist()]
        if not rows:
            continue
        page_number = int(table.page)
        same_header = bool(previous_rows and previous_rows[0] == rows[0] and any(rows[0]))
        if previous_rows and previous_page == page_number - 1 and same_header:
            previous_rows.extend(rows[1:])
            previous_page = page_number
            continue
        if previous_rows:
            sections.append({
                "page_number": table_start_page,
                "page_end": previous_page,
                "rows": previous_rows,
            })
        previous_rows, previous_page, table_start_page = rows, page_number, page_number
    if previous_rows:
        sections.append({"page_number": table_start_page, "page_end": previous_page, "rows": previous_rows})
    return sections

def ragflow_plain_sections(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            raise RagServiceError(f"Could not extract text from PDF page {page_number}.", 400) from error
        sections.extend({"page_number": page_number, "content": line.strip()} for line in text.splitlines() if line.strip())
    return sections

def ragflow_naive_merge(sections: list[dict], chunk_token_num: int) -> list[dict]:
    chunks, current, token_count = [], [], 0
    for section in sections:
        content = section["content"]
        section_tokens = len(content.split())
        if current and token_count + section_tokens > chunk_token_num:
            chunks.append({
                "page_number": current[0]["page_number"],
                "page_start": current[0]["page_number"],
                "page_end": current[-1]["page_number"],
                "content": "\n".join(item["content"] for item in current),
                "block_type": "general_text",
                "parent_chunk_id": "",
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
            "block_type": "general_text",
            "parent_chunk_id": "",
        })
    return chunks

def table_chunks(table: dict, table_id: str, chunk_token_num: int) -> list[dict]:
    rows = table["rows"]
    header = rows[0]
    metadata = {key: table[key] for key in ("page_number", "page_end")}
    chunks, current = [], [header]
    token_count = len(_markdown_table(current).split())
    for row in rows[1:]:
        row_tokens = len(" ".join(row).split())
        if len(current) > 1 and token_count + row_tokens > chunk_token_num:
            chunks.append({**metadata, "content": _markdown_table(current), "block_type": "table", "parent_chunk_id": table_id})
            current, token_count = [header], len(_markdown_table([header]).split())
        current.append(row)
        token_count += row_tokens
    if len(current) > 1 or len(rows) == 1:
        chunks.append({**metadata, "content": _markdown_table(current), "block_type": "table", "parent_chunk_id": table_id})
    return chunks

def table_context_chunks(text_sections: list[dict], table: dict, table_id: str, chunk_token_num: int) -> list[dict]:
    related = [section for section in text_sections if table["page_number"] <= section["page_number"] <= table["page_end"]]
    chunks = ragflow_naive_merge(related, chunk_token_num)
    for chunk in chunks:
        chunk["block_type"] = "table_context"
        chunk["parent_chunk_id"] = table_id
    return chunks


def parse_pdf(pdf_path: Path, *, chunk_token_num: int | None = None) -> dict:
    token_limit = chunk_token_num or int(os.getenv("RAG_CHUNK_TOKEN_NUM", "512"))
    if token_limit <= 0:
        raise ValueError("RAG_CHUNK_TOKEN_NUM must be positive.")
    text_sections = ragflow_plain_sections(pdf_path)
    tables = camelot_table_sections(pdf_path)
    chunks = ragflow_naive_merge(text_sections, token_limit)
    for index, table in enumerate(tables, start=1):
        table_id = f"table-{index}"
        chunks.extend(table_chunks(table, table_id, token_limit))
        chunks.extend(table_context_chunks(text_sections, table, table_id, token_limit))
    return {"source": str(pdf_path), "sections": text_sections, "chunks": chunks}

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
            "field_code": "",
            "field_name": "",
            "field_english_name": "",
            "detail_type": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ], settings)
    return {"source_type": "pdf", "chunk_count": len(chunks), "processed_pages": len({item["page_number"] for item in report["sections"]}), "ocr_pages": 0}