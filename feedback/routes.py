from flask import Blueprint, current_app, jsonify, render_template, request, session

from feedback.analyzer import propose_patch
from feedback.history import find_user_history
from feedback.profile import apply_patch, has_applied_feedback_group, load_profile, rollback_profile, validate_patch
from feedback.repository import create_feedback, latest_negative_feedback, list_feedback
from services.auth import login_required

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.get("/feedback")
@login_required
def feedback_page():
    return render_template("feedback.html")


@feedback_bp.get("/api/feedbacks")
@login_required
def list_feedbacks():
    return jsonify(items=list_feedback(str(session["id"])))


@feedback_bp.get("/api/feedback/profile")
@login_required
def get_profile():
    return jsonify(load_profile(current_app.config["NOTEBOOK_DATA_ROOT"], str(session["id"])))


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
    profile_updated = score == "bad" and _apply_patch_if_ready(user_id)
    return jsonify(status="saved", feedback_id=feedback_id, profile_updated=profile_updated), 201


@feedback_bp.post("/api/feedback/profile/rollback/<int:version>")
@login_required
def rollback_feedback_profile(version: int):
    profile = rollback_profile(current_app.config["NOTEBOOK_DATA_ROOT"], str(session["id"]), version)
    if not profile:
        return jsonify(error="Profile version not found."), 404
    return jsonify(profile)


def _apply_patch_if_ready(user_id: str) -> bool:
    negative_feedback = latest_negative_feedback(user_id, 3)
    if len(negative_feedback) != 3:
        return False
    profile = load_profile(current_app.config["NOTEBOOK_DATA_ROOT"], user_id)
    feedback_ids = [item["feedback_id"] for item in negative_feedback]
    if has_applied_feedback_group(profile, feedback_ids):
        return False
    try:
        patch = propose_patch(negative_feedback, profile["preferences"], current_app.config["LLM_SETTINGS"])
        validate_patch(patch)
        apply_patch(current_app.config["NOTEBOOK_DATA_ROOT"], user_id, feedback_ids, patch)
        return True
    except Exception:
        current_app.logger.exception("Unable to apply feedback profile patch")
        return False
