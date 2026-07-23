from datetime import timezone
from services.db import get_conn

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