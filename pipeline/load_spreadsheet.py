from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from services.config import Settings
from services.vectordb import CHUNK_SIZE, RagServiceError, index_chunks

def spreadsheet_sections(path: Path) -> list[str]:
    try:
        sheets = pd.read_csv(path, dtype=str, keep_default_na=False) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
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

def ingest_spreadsheet(path: Path, *, document_id: str, filename: str, settings: Settings) -> dict:
    chunks = merge_sections(spreadsheet_sections(path))
    if not chunks:
        raise RagServiceError("No readable rows were found in this spreadsheet.", 400)
    index_chunks(document_id, [
        {
            "chunk_id": uuid4().hex,
            "source_type": "spreadsheet",
            "url": "",
            "title": filename,
            "chunk_index": index,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for index, content in enumerate(chunks, start=1)
    ], settings)
    return {"source_type": "spreadsheet", "chunk_count": len(chunks)}