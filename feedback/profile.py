import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

PROFILE_FILENAME = "system_prompt.json"
PROFILE_LOCK = Lock()
ALLOWED_VALUES = {
    "/response_length": {"concise", "balanced", "detailed"},
    "/response_format": {"structured", "paragraphs"},
    "/tone": {"professional", "friendly", "direct"},
    "/correction_focus": {"accuracy", "citations", "directness", "clarity", "completeness", "avoid_repetition"},
}
DEFAULT_PREFERENCES = {
    "response_length": "balanced",
    "response_format": "structured",
    "use_bullets": True,
    "include_references": True,
    "tone": "professional",
    "correction_focus": [],
}


def load_profile(data_root: Path | str, user_id: str) -> dict:
    path = profile_path(data_root, user_id)
    if not path.exists():
        return _new_profile()
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _new_profile()
    return _normalize_profile(profile)


def profile_path(data_root: Path | str, user_id: str) -> Path:
    root = Path(data_root).resolve()
    user_dir = (root / str(user_id)).resolve()
    if root not in user_dir.parents:
        raise ValueError("Invalid user profile path.")
    return user_dir / PROFILE_FILENAME


def validate_patch(patch: object) -> list[dict]:
    if not isinstance(patch, list) or not patch:
        raise ValueError("Patch must be a non-empty JSON array.")
    paths = set()
    for item in patch:
        if not isinstance(item, dict) or set(item) != {"op", "path", "value"}:
            raise ValueError("Patch operation is invalid.")
        path = item["path"]
        if not isinstance(path, str) or item["op"] != "replace" or path in paths or path.removeprefix("/") not in DEFAULT_PREFERENCES:
            raise ValueError("Patch path is not allowed.")
        paths.add(path)
        value = item["value"]
        if path in {"/use_bullets", "/include_references"}:
            if not isinstance(value, bool):
                raise ValueError("Patch value must be a boolean.")
        elif path == "/correction_focus":
            if not isinstance(value, list) or len(value) > 6 or any(item not in ALLOWED_VALUES[path] for item in value):
                raise ValueError("Correction focus is not allowed.")
        elif value not in ALLOWED_VALUES[path]:
            raise ValueError("Patch value is not allowed.")
    return patch


def apply_patch(data_root: Path | str, user_id: str, source_feedback_ids: list[str], patch: object) -> dict:
    profile = load_profile(data_root, user_id)
    valid_patch = validate_patch(patch)
    previous = deepcopy(profile["preferences"])
    for operation in valid_patch:
        profile["preferences"][operation["path"].removeprefix("/")] = operation["value"]
    profile["versions"].append({"version": profile["version"], "preferences": previous, "applied_at": _now()})
    profile["version"] += 1
    profile["applied_feedback_groups"].append({"source_feedback_ids": source_feedback_ids, "applied_at": _now()})
    _save_profile(data_root, user_id, profile)
    return profile


def has_applied_feedback_group(profile: dict, feedback_ids: list[str]) -> bool:
    return any(set(item.get("source_feedback_ids", [])) == set(feedback_ids) for item in profile["applied_feedback_groups"])


def rollback_profile(data_root: Path | str, user_id: str, version: int) -> dict | None:
    profile = load_profile(data_root, user_id)
    snapshot = next((item for item in profile["versions"] if item["version"] == version), None)
    if not snapshot:
        return None
    profile["versions"].append({"version": profile["version"], "preferences": deepcopy(profile["preferences"]), "applied_at": _now()})
    profile["preferences"] = snapshot["preferences"]
    profile["version"] += 1
    _save_profile(data_root, user_id, profile)
    return profile


def preferences_instruction(profile: dict) -> str:
    preferences = profile["preferences"]
    bullets = "Use bullet points when helpful." if preferences["use_bullets"] else "Use paragraphs instead of bullet points."
    references = "Include source references when available." if preferences["include_references"] else "Do not add extra source references unless requested."
    focus = preferences["correction_focus"]
    focus_text = f" Prioritize: {', '.join(focus)}." if focus else ""
    return (
        f"Personal response preferences: {preferences['response_length']} detail, "
        f"{preferences['response_format']} format, {preferences['tone']} tone. {bullets} {references}{focus_text}"
    )


def _new_profile() -> dict:
    return {"version": 1, "preferences": deepcopy(DEFAULT_PREFERENCES), "applied_feedback_groups": [], "versions": []}


def _normalize_profile(profile: object) -> dict:
    if not isinstance(profile, dict):
        return _new_profile()
    normalized = _new_profile()
    normalized["version"] = profile.get("version") if isinstance(profile.get("version"), int) else 1
    preferences = profile.get("preferences") if isinstance(profile.get("preferences"), dict) else {}
    for key, default in DEFAULT_PREFERENCES.items():
        value = preferences.get(key, default)
        try:
            validate_patch([{ "op": "replace", "path": f"/{key}", "value": value }])
            normalized["preferences"][key] = value
        except ValueError:
            pass
    normalized["applied_feedback_groups"] = profile.get("applied_feedback_groups") if isinstance(profile.get("applied_feedback_groups"), list) else []
    normalized["versions"] = profile.get("versions") if isinstance(profile.get("versions"), list) else []
    return normalized


def _save_profile(data_root: Path | str, user_id: str, profile: dict) -> None:
    path = profile_path(data_root, user_id)
    with PROFILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
