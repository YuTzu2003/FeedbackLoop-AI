from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from services.config import Settings
from services.vectordb import CHUNK_SIZE, RagServiceError, index_chunks
from services.spreadsheet_analysis import workbook_frames

def spreadsheet_sections(path: Path) -> list[str]:
    try:
        sheets = workbook_frames(path)
    except (OSError, ValueError, ImportError) as error:
        raise RagServiceError("Spreadsheet parsing failed.", 400) from error
    if not isinstance(sheets, dict):
        sheets = {"Sheet1": sheets}
    sections = []
    for sheet_name, frame in sheets.items():
        for _, row in frame.fillna("").iterrows():
            fields = [f"{column}: {value}" for column, value in row.items() if str(value).strip()]
            if fields:
                sections.append("; ".join(fields) + f"; Sheet: {sheet_name}")
    return sections

def merge_sections(sections: list[str]) -> list[str]:
    chunks, current, length = [], [], 0
    for section in sections:
        extra = len(section) + (1 if current else 0)
        if current and length + extra > CHUNK_SIZE:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(section)
        length += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def definition_chunks(path: Path) -> list[dict]:
    chunks = []
    for sheet_name, frame in workbook_frames(path).items():
        if not any(word in sheet_name.casefold() for word in ("definition", "dictionary", "欄位", "代碼")):
            continue
        for _, row in frame.fillna("").iterrows():
            fields = {str(column).strip(): str(value).strip() for column, value in row.items() if str(value).strip()}
            if not fields:
                continue
            code = fields.get("field_code", fields.get("code", ""))
            chunks.append({
                "content": "; ".join(f"{column}: {value}" for column, value in fields.items()) + f"; Sheet: {sheet_name}",
                "block_type": "code_definition" if code else "field_definition",
                "field_name": fields.get("field_name", fields.get("欄位名稱", "")),
                "field_code": code,
            })
    return chunks

def ingest_spreadsheet(path: Path, *, document_id: str, filename: str, settings: Settings) -> dict:
    chunks = [{"content": content, "block_type": "general_text", "field_name": "", "field_code": ""} for content in merge_sections(spreadsheet_sections(path))]
    chunks.extend(definition_chunks(path))
    if not chunks:
        raise RagServiceError("No readable rows were found in this spreadsheet.", 400)
    index_chunks(document_id, [
        {
            "chunk_id": uuid4().hex,
            "source_type": "spreadsheet",
            "url": "",
            "title": filename,
            "chunk_index": index,
            **chunk,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for index, chunk in enumerate(chunks, start=1)
    ], settings)
    return {"source_type": "spreadsheet", "chunk_count": len(chunks)}
