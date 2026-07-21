from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pipeline.retrieve_answer import answer_from_chunks, answer_from_history, retrieve_chunks
from pipeline.load_url import ingest_web_url
from pipeline.load_pdf import ingest_pdf
from services.config import load_settings
from services.api import load_llm_settings, get_system_prompt
from services.vectordb import RagServiceError, delete_document, weaviate_status
from services.storage import (append_jsonl,delete_notebook_records,get_notebook,notebook_history,read_jsonl,save_feedback,save_upload,)

load_dotenv()

settings = load_settings()
llm_settings = load_llm_settings()
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
HISTORY_DIR = BASE_DIR / "tasks" / "historys"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
app.config["FEEDBACK_LOG"] = HISTORY_DIR / "feedbacks.jsonl"
app.config["NOTEBOOK_LOG"] = HISTORY_DIR / "notebooks.jsonl"
app.config["NOTEBOOK_HISTORY_LOG"] = HISTORY_DIR / "notebook_history.jsonl"
app.config["UPLOAD_FOLDER"] = BASE_DIR / "uploads"
app.config["PDF_CHUNK_REPORT_DIR"] = BASE_DIR / "tmp" / "pdf_chunks"
feedback_log_lock = Lock()
notebook_log_lock = Lock()
notebook_history_log_lock = Lock()


def current_notebook(notebook_id: str) -> dict | None:
    return get_notebook(app.config["NOTEBOOK_LOG"], notebook_id)


@app.get("/")
def index():
    return render_template("index.html", llm_model=llm_settings.model)


@app.get("/feedback")
@app.get("/feedbacks")
def feedback_page():
    return render_template("feedback.html")


@app.get("/connection")
def connection_page():
    return render_template("connection.html")


@app.get("/api/feedbacks")
def list_feedbacks():
    return jsonify(items=read_jsonl(app.config["FEEDBACK_LOG"], "feedback"))


@app.get("/api/notebooks")
def list_notebooks():
    return jsonify(items=read_jsonl(app.config["NOTEBOOK_LOG"], "notebook"))


@app.get("/api/notebooks/<notebook_id>/history")
def list_notebook_history(notebook_id: str):
    if not current_notebook(notebook_id):
        return jsonify(error="找不到此筆記本。"), 404
    return jsonify(items=list(reversed(notebook_history(app.config["NOTEBOOK_HISTORY_LOG"], notebook_id))))


@app.delete("/api/notebooks/<notebook_id>")
def delete_notebook(notebook_id: str):
    notebook = current_notebook(notebook_id)
    if not notebook:
        return jsonify(error="找不到此筆記本。"), 404
    
    try:
        delete_document(notebook_id, settings)
    except RagServiceError as e:
        app.logger.warning(f"Failed to delete document from Weaviate: {e}")
        
    delete_notebook_records(app.config["NOTEBOOK_LOG"], app.config["NOTEBOOK_HISTORY_LOG"], notebook_id, notebook_log_lock, notebook_history_log_lock)
    
    if notebook.get("stored_filename"):
        file_path = Path(app.config["UPLOAD_FOLDER"]) / notebook["stored_filename"]
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                app.logger.warning(f"Failed to delete file {file_path}: {e}")
                
    return jsonify(status="deleted")



@app.get("/api/weaviate/status")
def check_weaviate_status():
    try:
        return jsonify(weaviate_status(settings))
    except Exception:
        app.logger.exception("Weaviate connection check failed")
        return jsonify(ready=False, live=False, error="無法連線至 Weaviate。"), 503


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="請選擇要上傳的檔案。"), 400
    try:
        notebook = save_upload(file, Path(app.config["UPLOAD_FOLDER"]))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    if Path(notebook["stored_filename"]).suffix.lower() == ".pdf":
        try:
            notebook.update(
                ingest_pdf(
                    Path(app.config["UPLOAD_FOLDER"]) / notebook["stored_filename"],
                    document_id=notebook["id"],
                    filename=notebook["name"],
                    settings=settings,
                    report_dir=Path(app.config["PDF_CHUNK_REPORT_DIR"]),
                )
            )
        except RagServiceError as error:
            try:
                delete_document(notebook["id"], settings)
            except RagServiceError:
                app.logger.exception("Failed to clean up PDF vectors after indexing failure")
            (Path(app.config["UPLOAD_FOLDER"]) / notebook["stored_filename"]).unlink(missing_ok=True)
            return jsonify(error=str(error)), error.status_code
    append_jsonl(app.config["NOTEBOOK_LOG"], notebook_log_lock, notebook)
    return jsonify(notebook_id=notebook["id"], filename=notebook["name"], size=notebook["size"], chunk_count=notebook.get("chunk_count"))


@app.post("/api/upload_url")
def upload_url():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify(error="請輸入網址。"), 400
    try:
        notebook = ingest_web_url(url, settings)
    except RagServiceError as error:
        return jsonify(error=str(error)), error.status_code
    notebook["created_at"] = datetime.now(timezone.utc).isoformat()
    append_jsonl(app.config["NOTEBOOK_LOG"], notebook_log_lock, notebook)
    return jsonify(notebook_id=notebook["id"], filename=notebook["name"], chunk_count=notebook["chunk_count"])


@app.post("/api/ask")
def ask():
    payload = request.json or {}
    question = payload.get("question", "").strip()
    notebook_id = payload.get("notebook_id", "").strip()
    if not question:
        return jsonify(error="請輸入問題。"), 400
    if not notebook_id:
        return jsonify(error="請選擇筆記本。"), 400
    notebook = current_notebook(notebook_id)
    if not notebook:
        return jsonify(error="找不到此筆記本。"), 404
    try:
        if notebook.get("source_type") in {"web", "pdf"}:
            chunks = retrieve_chunks(question, notebook_id, settings)
            if not chunks:
                return jsonify(error="找不到此網址來源的相關內容。"), 404
            answer = answer_from_chunks(question, chunks, llm_settings)
            sources = [
                {
                    key: item.get(key)
                    for key in ("source_type", "title", "url", "page_number", "chunk_index", "score")
                }
                for item in chunks
            ]
        else:
            messages = [get_system_prompt()]
            for item in reversed(notebook_history(app.config["NOTEBOOK_HISTORY_LOG"], notebook_id)[:3]):
                messages.extend([{"role": "user", "content": item["question"]}, {"role": "assistant", "content": item["answer"]}])
            messages.append({"role": "user", "content": question})
            answer = answer_from_history(messages, llm_settings)
            sources = ["AI 文字分析"]
    except RagServiceError as error:
        return jsonify(error=str(error)), error.status_code

    record = {
        "id": uuid4().hex,
        "notebook_id": notebook_id,
        "question": question,
        "answer": answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(app.config["NOTEBOOK_HISTORY_LOG"], notebook_history_log_lock, record)
    return jsonify(answer=answer, sources=sources, history_id=record["id"], notebook_id=notebook_id)


@app.post("/api/feedback")
def feedback():
    try:
        save_feedback(app.config["FEEDBACK_LOG"], feedback_log_lock, request.json or {})
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except FileExistsError as error:
        return jsonify(error=str(error)), 409
    return jsonify(status="saved"), 201


if __name__ == "__main__":
    app.run(debug=True)
