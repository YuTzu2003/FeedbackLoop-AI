import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from services.db import get_conn

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf"}

def read_jsonl(log_path: Path | str, label: str) -> list[dict]:
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))

def append_jsonl(log_path: Path | str, lock: Lock, record: dict) -> None:
    log_path = Path(log_path)
    with lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

def notebook_history(history_log: Path | str, notebook_id: str) -> list[dict]:
    return [item for item in read_jsonl(Path(history_log), "notebook history") if item.get("notebook_id") == notebook_id]

def notebook_data_dir(data_root: Path | str, user_id: str, notebook_id: str) -> Path:
    root = Path(data_root).resolve()
    notebook_dir = (root / str(user_id) / notebook_id).resolve()
    if root not in notebook_dir.parents:
        raise ValueError("Invalid notebook path.")
    return notebook_dir

def notebook_history_path(data_root: Path | str, user_id: str, notebook_id: str) -> Path:
    return notebook_data_dir(data_root, user_id, notebook_id) / "history.jsonl"

def save_upload(file: FileStorage, notebook_dir: Path | str, notebook_id: str) -> dict:
    notebook_dir = Path(notebook_dir)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    original_filename = file.filename or ""
    if "." not in original_filename or original_filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("只支援 CSV、Excel、PDF、TXT 與 DOCX 文件。")
    safe_filename = secure_filename(original_filename) or f"document{Path(original_filename).suffix.lower()}"
    stored_filename = safe_filename
    file_path = notebook_dir / stored_filename
    file.save(file_path)
    return {
        "id": notebook_id,
        "name": original_filename,
        "stored_filename": stored_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size": file_path.stat().st_size,}

def delete_notebook_data(data_root: Path | str, user_id: str, notebook_id: str) -> None:
    notebook_dir = notebook_data_dir(data_root, user_id, notebook_id)
    if notebook_dir.exists():
        shutil.rmtree(notebook_dir)

def _notebook_from_row(row) -> dict:
    created_at = row.CreatedAt.replace(tzinfo=timezone.utc).isoformat()
    filename_prefix = f"{row.NotebookId}_"
    fallback_name = row.StoreFilename.removeprefix(filename_prefix) if row.StoreFilename else row.Url
    return {
        "id": row.NotebookId,
        "name": row.Title or fallback_name or "Untitled notebook",
        "stored_filename": row.StoreFilename,
        "source_type": row.SourceType,
        "url": row.Url,
        "chunk_count": row.ChunkCount,
        "created_at": created_at,
        "owner_user_id": str(row.UserID),}

def list_notebooks(owner_user_id: str) -> list[dict]:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""SELECT NotebookId, UserID, Title, StoreFilename, SourceType, Url, ChunkCount, CreatedAt FROM dbo.Notebooks WHERE UserID = ? ORDER BY CreatedAt DESC""",(owner_user_id,) )
        return [_notebook_from_row(row) for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

def get_notebook(notebook_id: str, owner_user_id: str) -> dict | None:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""SELECT NotebookId, UserID, Title, StoreFilename, SourceType, Url, ChunkCount, CreatedAt FROM dbo.Notebooks WHERE NotebookId = ? AND UserID = ?""", (notebook_id, owner_user_id))
        row = cursor.fetchone()
        return _notebook_from_row(row) if row else None
    finally:
        cursor.close()
        conn.close()

def create_notebook(owner_user_id: str, notebook: dict) -> None:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO dbo.Notebooks (NotebookId, UserID, Title, StoreFilename, SourceType, Url, ChunkCount) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (notebook["id"],owner_user_id,notebook["name"],notebook.get("stored_filename"),notebook.get("source_type"),notebook.get("url"),notebook.get("chunk_count")))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def delete_notebook(notebook_id: str, owner_user_id: str) -> bool:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.Notebooks WHERE NotebookId = ? AND UserID = ?",(notebook_id, owner_user_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()