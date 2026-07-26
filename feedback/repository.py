from datetime import datetime, timezone
from uuid import uuid4

from services.db import get_conn


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
        return [_feedback_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def latest_negative_feedback(user_id: str, limit: int = 3) -> list[dict]:
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT TOP (?) FeedbackID, HistoryID, Question, Answer, Score, Note, CreatedAt
            FROM dbo.FeedBack
            WHERE UserID = ? AND Score = 'bad'
            ORDER BY CreatedAt DESC
            """,
            limit, user_id,
        )
        return [_feedback_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _feedback_row(row) -> dict:
    return {
        "feedback_id": row.FeedbackID,
        "history_id": row.HistoryID,
        "question": row.Question,
        "answer": row.Answer,
        "score": row.Score,
        "note": row.Note,
        "created_at": row.CreatedAt.replace(tzinfo=timezone.utc).isoformat(),
    }
