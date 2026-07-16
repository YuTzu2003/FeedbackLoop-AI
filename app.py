import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

import openai
import weaviate
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["FEEDBACK_LOG"] = Path("feedbacks.jsonl")
app.config["NOTEBOOK_LOG"] = Path("notebooks.jsonl")
app.config["NOTEBOOK_HISTORY_LOG"] = Path("notebook_history.jsonl")
app.config["UPLOAD_FOLDER"] = Path("uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "txt", "docx"}
feedback_log_lock = Lock()
notebook_log_lock = Lock()
notebook_history_log_lock = Lock()


def weaviate_client():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
        port=int(os.getenv("WEAVIATE_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )


def weaviate_status() -> dict:
    client = weaviate_client()
    try:
        return {"ready": client.is_ready(), "live": client.is_live()}
    finally:
        client.close()


llm_client = openai.OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://120.113.70.236:8002/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    timeout=180,
)
llm_model = os.getenv("LLM_MODEL", "google/gemma-4-31B-it-qat-w4a16-ct")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_jsonl(log_path: Path, label: str) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            app.logger.warning("Skipped malformed %s log line", label)
    return list(reversed(records))


def append_jsonl(log_path: Path, lock: Lock, record: dict) -> None:
    with lock, log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_notebook(notebook_id: str) -> dict | None:
    return next((item for item in read_jsonl(app.config["NOTEBOOK_LOG"], "notebook") if item.get("id") == notebook_id), None)


def notebook_history(notebook_id: str) -> list[dict]:
    return [item for item in read_jsonl(app.config["NOTEBOOK_HISTORY_LOG"], "notebook history") if item.get("notebook_id") == notebook_id]


@app.get("/")
def index():
    return render_template("index.html", llm_model=llm_model)


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
    if not get_notebook(notebook_id):
        return jsonify(error="找不到此筆記本。"), 404
    return jsonify(items=list(reversed(notebook_history(notebook_id))))


@app.get("/api/weaviate/status")
def check_weaviate_status():
    try:
        return jsonify(weaviate_status())
    except Exception:
        app.logger.exception("Weaviate connection check failed")
        return jsonify(ready=False, live=False, error="無法連線至 Weaviate。"), 503


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="請選擇要上傳的檔案。"), 400
    if not allowed_file(file.filename):
        return jsonify(error="支援 CSV、Excel、PDF、TXT 與 DOCX 檔案。"), 400

    notebook_id = uuid4().hex
    original_filename = file.filename
    safe_filename = secure_filename(original_filename) or f"document{Path(original_filename).suffix.lower()}"
    stored_filename = f"{notebook_id}_{safe_filename}"
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(exist_ok=True)
    file_path = upload_folder / stored_filename
    file.save(file_path)
    notebook = {
        "id": notebook_id,
        "name": original_filename,
        "stored_filename": stored_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(app.config["NOTEBOOK_LOG"], notebook_log_lock, notebook)
    return jsonify(notebook_id=notebook_id, filename=original_filename, size=file_path.stat().st_size)


@app.post("/api/ask")
def ask():
    payload = request.json or {}
    question = payload.get("question", "").strip()
    notebook_id = payload.get("notebook_id", "").strip()
    if not question:
        return jsonify(error="請輸入問題。"), 400
    if not notebook_id:
        return jsonify(error="請先選擇一個文件筆記本。"), 400
    if not get_notebook(notebook_id):
        return jsonify(error="找不到此筆記本。"), 404

    recent_history = list(reversed(notebook_history(notebook_id)[:5]))
    messages = [{"role": "system", "content": "Answer clearly and concisely in Traditional Chinese."}]
    for item in recent_history:
        messages.extend([
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ])
    messages.append({"role": "user", "content": question})

    try:
        response = llm_client.chat.completions.create(
            model=llm_model, messages=messages, max_tokens=800, temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
    except openai.APITimeoutError:
        return jsonify(error="模型回應逾時，請稍後再試。"), 504
    except Exception:
        app.logger.exception("LLM request failed")
        return jsonify(error="模型服務暫時無法使用。"), 503

    record = {
        "id": uuid4().hex,
        "notebook_id": notebook_id,
        "question": question,
        "answer": answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(app.config["NOTEBOOK_HISTORY_LOG"], notebook_history_log_lock, record)
    return jsonify(answer=answer, sources=["AI 文字分析"], history_id=record["id"], notebook_id=notebook_id)


@app.post("/api/feedback")
def save_feedback():
    payload = request.json or {}
    if payload.get("score") not in {"good", "bad"}:
        return jsonify(error="Invalid feedback score."), 400
    record = {
        "score": payload["score"], "note": payload.get("note", "").strip(),
        "question": payload.get("question", "").strip(), "answer": payload.get("answer", "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(app.config["FEEDBACK_LOG"], feedback_log_lock, record)
    return jsonify(status="saved"), 201


if __name__ == "__main__":
    app.run(debug=True)
