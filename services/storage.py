import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "txt", "docx"}

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

def get_notebook(notebook_log: Path | str, notebook_id: str) -> dict | None:
    return next((item for item in read_jsonl(Path(notebook_log), "notebook") if item.get("id") == notebook_id), None)

def notebook_history(history_log: Path | str, notebook_id: str) -> list[dict]:
    return [item for item in read_jsonl(Path(history_log), "notebook history") if item.get("notebook_id") == notebook_id]

def save_upload(file: FileStorage, upload_dir: Path | str) -> dict:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = file.filename or ""
    if "." not in original_filename or original_filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("只支援 CSV、Excel、PDF、TXT 與 DOCX 文件。")
    notebook_id = uuid4().hex
    safe_filename = secure_filename(original_filename) or f"document{Path(original_filename).suffix.lower()}"
    stored_filename = f"{notebook_id}_{safe_filename}"
    file_path = upload_dir / stored_filename
    file.save(file_path)
    return {
        "id": notebook_id,
        "name": original_filename,
        "stored_filename": stored_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size": file_path.stat().st_size,}

def save_feedback(log_path: Path | str, lock: Lock, payload: dict) -> None:
    log_path = Path(log_path)
    history_id = payload.get("history_id", "").strip()
    if payload.get("score") not in {"good", "bad"}:
        raise ValueError("Invalid feedback score.")
    if not history_id:
        raise ValueError("缺少回答識別碼。")
    if payload["score"] == "bad" and not payload.get("note", "").strip():
        raise ValueError("請說明需要改善的地方")
    with lock:
        if any(item.get("history_id") == history_id for item in read_jsonl(log_path, "feedback")):
            raise FileExistsError("此回答已經有回饋紀錄。")
        record = {
            "score": payload["score"],
            "note": payload.get("note", "").strip(),
            "question": payload.get("question", "").strip(),
            "answer": payload.get("answer", "").strip(),
            "history_id": history_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")


def delete_notebook_records(notebook_log: Path | str, history_log: Path | str, notebook_id: str, lock_notebook: Lock, lock_history: Lock) -> None:
    notebook_log = Path(notebook_log)
    history_log = Path(history_log)
    with lock_notebook:
        if notebook_log.exists():
            records = []
            for line in notebook_log.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    if record.get("id") != notebook_id:
                        records.append(line)
                except json.JSONDecodeError:
                    continue
            notebook_log.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    
    with lock_history:
        if history_log.exists():
            records = []
            for line in history_log.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    if record.get("notebook_id") != notebook_id:
                        records.append(line)
                except json.JSONDecodeError:
                    continue
            history_log.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")