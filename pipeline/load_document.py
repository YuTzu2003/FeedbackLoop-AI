from pathlib import Path
from pipeline.load_pdf import ingest_pdf
from pipeline.load_spreadsheet import ingest_spreadsheet
from services.config import Settings
from services.vectordb import RagServiceError

def ingest_document(path: Path, *, document_id: str, filename: str, settings: Settings, report_dir: Path | None = None) -> dict:
    if path.suffix.lower() == ".pdf":
        return ingest_pdf(path, document_id=document_id, filename=filename, settings=settings, report_dir=report_dir)
    if path.suffix.lower() in {".csv", ".xls", ".xlsx"}:
        return ingest_spreadsheet(path, document_id=document_id, filename=filename, settings=settings)
    raise RagServiceError("Unsupported document type.", 400)
