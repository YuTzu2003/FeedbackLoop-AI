import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UPLOAD_FOLDER = Path("uploads")
DATABASE = Path("feedback.db")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "txt", "docx"}


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, score TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '', question TEXT NOT NULL DEFAULT '',
        answer TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
    )""")
    return connection


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/feedbacks")
def feedbacks_page():
    return render_template("feedbacks.html")


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="\u8acb\u9078\u64c7\u8981\u4e0a\u50b3\u7684\u6a94\u6848\u3002"), 400
    if not allowed_file(file.filename):
        return jsonify(error="\u6b64\u6a94\u6848\u683c\u5f0f\u5c1a\u672a\u652f\u63f4\u3002"), 400
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    filename = secure_filename(file.filename)
    file.save(UPLOAD_FOLDER / filename)
    return jsonify(filename=filename, size=(UPLOAD_FOLDER / filename).stat().st_size)


@app.post("/api/ask")
def ask():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify(error="\u8acb\u8f38\u5165\u554f\u984c\u3002"), 400
    answer = (
        "\u6211\u5df2\u6839\u64da\u5df2\u4e0a\u50b3\u8cc7\u6599\u6574\u7406\u56de\u7b54\u3002\n\n"
        f"\u91dd\u5c0d\u300c{question}\u300d\uff0c\u5efa\u8b70\u5148\u78ba\u8a8d\u8cc7\u6599\u7684\u6642\u9593\u7bc4\u570d\u3001\u6b04\u4f4d\u5b9a\u7fa9\u8207\u7f3a\u6f0f\u503c\uff0c"
        "\u518d\u6bd4\u8f03\u4e3b\u8981\u8da8\u52e2\u8207\u7570\u5e38\u9ede\u3002"
    )
    return jsonify(answer=answer, sources=["\u4e0a\u50b3\u8cc7\u6599\u96c6", "\u8cc7\u6599\u6b04\u4f4d\u8207\u54c1\u8cea\u6aa2\u67e5"])


@app.post("/api/feedback")
def save_feedback():
    payload = request.json or {}
    score = payload.get("score")
    if score not in {"good", "bad"}:
        return jsonify(error="Invalid feedback score."), 400
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO feedback (score, note, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
            (score, payload.get("note", "").strip(), payload.get("question", "").strip(),
             payload.get("answer", "").strip(), datetime.now(timezone.utc).isoformat()),
        )
    return jsonify(status="saved", id=cursor.lastrowid)


@app.get("/api/feedbacks")
def list_feedbacks():
    score = request.args.get("score", "all")
    with get_db() as db:
        summary = dict(db.execute("SELECT COUNT(*) AS total, SUM(score='good') AS good, SUM(score='bad') AS bad FROM feedback").fetchone())
        if score in {"good", "bad"}:
            rows = db.execute("SELECT * FROM feedback WHERE score = ? ORDER BY id DESC", (score,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    return jsonify(summary={key: value or 0 for key, value in summary.items()}, items=[dict(row) for row in rows])


@app.delete("/api/feedbacks/<int:feedback_id>")
def delete_feedback(feedback_id: int):
    with get_db() as db:
        db.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
    return jsonify(status="deleted")


if __name__ == "__main__":
    app.run(debug=True)
