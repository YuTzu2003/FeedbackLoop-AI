from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
from dotenv import load_dotenv
import logging
import sys
from flask import Flask, jsonify, render_template, request, session
from pipeline.retrieve_answer import answer_from_chunks, answer_from_history, retrieve_chunks
from pipeline.load_url import ingest_web_url
from pipeline.load_pdf import ingest_pdf
from services.config import load_settings
from services.api import load_llm_settings, get_system_prompt
from services.vectordb import RagServiceError, delete_document, weaviate_status
from services.notebook_repositories import (append_jsonl,delete_notebook_data,notebook_data_dir,notebook_history,notebook_history_path,save_upload, create_notebook, delete_notebook as delete_notebook_record, get_notebook, list_notebooks as list_notebook_records)
import os
from services.auth import auth_bp, login_required, admin_required
from feedback import feedback_bp
from feedback.profile import load_profile, preferences_instruction

load_dotenv()
settings = load_settings()
llm_settings = load_llm_settings()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = f"{BASE_DIR}/tasks/historys"
Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["NOTEBOOK_DATA_ROOT"] = BASE_DIR / "tasks/notebooks"
app.config["PDF_CHUNK_REPORT_DIR"] = f"{BASE_DIR}/tmp/pdf_chunks"
app.config["LLM_SETTINGS"] = llm_settings
notebook_history_log_lock = Lock()

logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S',handlers=[logging.StreamHandler(sys.stdout)])
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.handlers = []
werkzeug_logger.propagate = True
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

app.register_blueprint(auth_bp)
app.register_blueprint(feedback_bp)

def current_user_notebook(notebook_id: str) -> dict | None:
    return get_notebook(notebook_id, str(session["id"]))

def current_notebook_dir(notebook_id: str) -> Path:
    return notebook_data_dir(app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]), notebook_id)

def current_notebook_history_path(notebook_id: str) -> Path:
    return notebook_history_path(app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]), notebook_id)


@app.get("/")
@login_required
def index():
    return render_template("index.html", llm_model=llm_settings.model)


@app.get("/connection")
@admin_required
def connection_page():
    return render_template("connection.html")


@app.get("/api/notebooks")
@login_required
def list_notebooks_api():
    return jsonify(items=list_notebook_records(str(session["id"])))


@app.get("/api/notebooks/<notebook_id>/history")
@login_required
def list_notebook_history(notebook_id: str):
    if not current_user_notebook(notebook_id):
        return jsonify(error="找不到此筆記本。"), 404
    return jsonify(items=list(reversed(notebook_history(current_notebook_history_path(notebook_id), notebook_id))))


@app.delete("/api/notebooks/<notebook_id>")
@login_required
def delete_notebook_api(notebook_id: str):
    notebook = current_user_notebook(notebook_id)
    if not notebook:
        return jsonify(error="找不到此筆記本。"), 404
    
    try:
        delete_document(notebook_id, settings)
    except RagServiceError as e:
        app.logger.warning(f"Failed to delete document from Weaviate: {e}")
        
    delete_notebook_data(app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]), notebook_id)
    delete_notebook_record(notebook_id, str(session["id"]))
                
    return jsonify(status="deleted")



@app.get("/api/weaviate/status")
@login_required
def check_weaviate_status():
    try:
        return jsonify(weaviate_status(settings))
    except Exception:
        app.logger.exception("Weaviate connection check failed")
        return jsonify(ready=False, live=False, error="無法連線至 Weaviate。"), 503


@app.post("/api/upload")
@login_required
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="請選擇要上傳的檔案。"), 400
    try:
        notebook_id = uuid4().hex
        notebook = save_upload(file, current_notebook_dir(notebook_id), notebook_id)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    if Path(notebook["stored_filename"]).suffix.lower() == ".pdf":
        try:
            notebook.update(
                ingest_pdf(
                    current_notebook_dir(notebook["id"]) / notebook["stored_filename"],
                    document_id=notebook["id"],
                    filename=notebook["name"],
                    settings=settings,
                    report_dir=Path(app.config["PDF_CHUNK_REPORT_DIR"]),
                )
            )
        except Exception as error:
            try:
                delete_document(notebook["id"], settings)
            except Exception:
                app.logger.exception("Failed to clean up PDF vectors after indexing failure")
            delete_notebook_data(app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]), notebook["id"])
            
            if isinstance(error, RagServiceError):
                return jsonify(error=str(error)), error.status_code
            
            app.logger.exception("Unexpected error during PDF ingestion")
            return jsonify(error="後台處理文件時發生非預期錯誤，已取消儲存。"), 500
    create_notebook(str(session["id"]), notebook)
    return jsonify(notebook_id=notebook["id"], filename=notebook["name"], size=notebook["size"], chunk_count=notebook.get("chunk_count"))


@app.post("/api/upload_url")
@login_required
def upload_url():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify(error="請輸入網址。"), 400
    try:
        notebook = ingest_web_url(url, settings)
    except Exception as error:
        if isinstance(error, RagServiceError):
            return jsonify(error=str(error)), error.status_code
        app.logger.exception("Unexpected error during URL ingestion")
        return jsonify(error="後台處理網址時發生非預期錯誤，已取消儲存。"), 500
    notebook["created_at"] = datetime.now(timezone.utc).isoformat()
    current_notebook_dir(notebook["id"]).mkdir(parents=True, exist_ok=True)
    create_notebook(str(session["id"]), notebook)
    return jsonify(notebook_id=notebook["id"], filename=notebook["name"], chunk_count=notebook["chunk_count"])


@app.post("/api/ask")
@login_required
def ask():
    payload = request.json or {}
    question = payload.get("question", "").strip()
    notebook_id = payload.get("notebook_id", "").strip()
    search_mode = str(payload.get("search_mode") or "near_vector").strip()
    if not question:
        return jsonify(error="請輸入問題。"), 400
    if not notebook_id:
        return jsonify(error="請選擇筆記本。"), 400
    if search_mode not in {"hybrid", "near_vector"}:
        return jsonify(error="Invalid search mode."), 400
    notebook = current_user_notebook(notebook_id)
    if not notebook:
        return jsonify(error="找不到此筆記本。"), 404
    profile = load_profile(app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]))
    personal_instruction = preferences_instruction(profile)
    try:
        if notebook.get("source_type") in {"web", "pdf"}:
            chunks = retrieve_chunks(question, notebook_id, settings, llm_settings, search_mode)
            if not chunks:
                return jsonify(error="找不到此網址來源的相關內容。"), 404
            answer = answer_from_chunks(question, chunks, llm_settings, personal_instruction)
            sources = [
                {
                    key: item.get(key)
                    for key in ("source_type", "title", "url", "page_number", "chunk_index", "score")
                }
                for item in chunks
            ]
        else:
            messages = [get_system_prompt(personal_instruction)]
            for item in reversed(notebook_history(current_notebook_history_path(notebook_id), notebook_id)[:3]):
                messages.extend([{"role": "user", "content": item["question"]}, {"role": "assistant", "content": item["answer"]}])
            messages.append({"role": "user", "content": question})
            answer = answer_from_history(messages, llm_settings)
            sources = ["AI 文字分析"]
    except RagServiceError as error:
        return jsonify(error=str(error)), error.status_code

    record = {
        "id": uuid4().hex,
        "notebook_id": notebook_id,
        "owner_user_id": str(session["id"]),
        "question": question,
        "answer": answer,
        "sources": sources,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(current_notebook_history_path(notebook_id), notebook_history_log_lock, record)
    return jsonify(answer=answer, sources=sources, history_id=record["id"], notebook_id=notebook_id)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
