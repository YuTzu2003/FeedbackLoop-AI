import os
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import openai
import weaviate
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UPLOAD_FOLDER = Path("uploads")
app.config["FEEDBACK_LOG"] = Path("feedbacks.jsonl")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "txt", "docx"}
feedback_log_lock = Lock()


def weaviate_client():
    return weaviate.connect_to_local(
        host=os.getenv("WEAVIATE_HOST", "127.0.0.1"),
        port=int(os.getenv("WEAVIATE_PORT", "8080")),
        grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    )


def weaviate_status() -> dict:
    """Check the Weaviate connection without creating or querying any data."""
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


@app.get("/")
def index():
    return render_template("index.html", llm_model=llm_model)


@app.get("/feedbacks")
def feedbacks_page():
    return render_template("feedbacks.html")


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
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    filename = secure_filename(file.filename)
    file.save(UPLOAD_FOLDER / filename)
    return jsonify(filename=filename, size=(UPLOAD_FOLDER / filename).stat().st_size)


@app.post("/api/ask")
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify(error="請輸入問題。"), 400
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "Answer clearly and concisely in Traditional Chinese."},
                {"role": "user", "content": question},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
    except openai.APITimeoutError:
        return jsonify(error="模型回應逾時，請稍後再試。"), 504
    except Exception:
        app.logger.exception("LLM request failed")
        return jsonify(error="模型服務暫時無法使用。"), 503
    return jsonify(answer=answer, sources=["AI 文字分析"])


@app.post("/api/feedback")
def save_feedback():
    payload = request.json or {}
    if payload.get("score") not in {"good", "bad"}:
        return jsonify(error="Invalid feedback score."), 400
    record = {
        "score": payload["score"],
        "note": payload.get("note", "").strip(),
        "question": payload.get("question", "").strip(),
        "answer": payload.get("answer", "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with feedback_log_lock, app.config["FEEDBACK_LOG"].open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonify(status="saved"), 201


if __name__ == "__main__":
    app.run(debug=True)
