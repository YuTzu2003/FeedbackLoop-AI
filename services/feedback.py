from pathlib import Path
from datetime import datetime, timezone
from flask import Blueprint, current_app, jsonify, render_template, request, session
from services.auth import login_required
from services.notebook_repositories import read_jsonl
from uuid import uuid4
from services.db import get_conn

feedback_bp = Blueprint("feedback", __name__)

def find_user_history(data_root: Path | str, user_id: str, history_id: str) -> dict | None:
    root = Path(data_root).resolve()
    user_dir = (root / str(user_id)).resolve()
    if root not in user_dir.parents or not user_dir.exists():
        return None

    for history_path in user_dir.glob("*/history.jsonl"):
        for record in read_jsonl(history_path, "notebook history"):
            if record.get("id") == history_id:
                return record
    return None

def create_feedback(user_id: str, history: dict, score: str, note: str) -> str:
    feedback_id = uuid4().hex
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dbo.FeedBack (FeedbackID, UserID, HistoryID, Question, Answer, Score, Note, CreatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            feedback_id, user_id, history["id"], history["question"], history["answer"], score, note, datetime.now(),
        )
        conn.commit()
        return feedback_id
    finally:
        conn.close()


def list_feedback(user_id: str) -> list[dict]:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT FeedbackID, HistoryID, Question, Answer, Score, Note, CreatedAt
            FROM dbo.FeedBack
            WHERE UserID = ?
            ORDER BY CreatedAt DESC
            """,
            user_id,
        )
        return [
            {
                "feedback_id": row.FeedbackID,
                "history_id": row.HistoryID,
                "question": row.Question,
                "answer": row.Answer,
                "score": row.Score,
                "note": row.Note,
                "created_at": row.CreatedAt.replace(tzinfo=timezone.utc).isoformat(),
            }
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


@feedback_bp.get("/feedback")
@login_required
def feedback_page():
    return render_template("feedback.html")


@feedback_bp.get("/api/feedbacks")
@login_required
def list_feedbacks():
    return jsonify(items=list_feedback(str(session["id"])))

@feedback_bp.post("/api/feedback")
@login_required
def submit_feedback():
    payload = request.json or {}
    history_id = str(payload.get("history_id") or "").strip()
    score = str(payload.get("score") or "").strip()
    note = str(payload.get("note") or "").strip()
    if not history_id:
        return jsonify(error="A history ID is required."), 400
    if score not in {"good", "bad"}:
        return jsonify(error="Invalid feedback score."), 400
    if score == "bad" and not note:
        return jsonify(error="A note is required for negative feedback."), 400

    user_id = str(session["id"])
    history = find_user_history(current_app.config["NOTEBOOK_DATA_ROOT"], user_id, history_id)
    if not history:
        return jsonify(error="Answer not found."), 404
    feedback_id = create_feedback(user_id, history, score, note)
    return jsonify(status="saved", feedback_id=feedback_id), 201
