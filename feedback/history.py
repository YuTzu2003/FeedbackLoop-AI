from pathlib import Path

from services.notebook_repositories import read_jsonl


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
