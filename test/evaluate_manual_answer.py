"""Evaluate saved answers for the fixed manual RAG cases without calling an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFUSAL_MARKERS = (
    "無法回答",
    "沒有提供",
    "資訊不足",
    "cannot answer",
    "do not have enough information",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True, help="JSON list containing id and answer")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_manual_evaluation.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = {item["id"]: item for item in json.loads(args.cases.read_text(encoding="utf-8"))}
    answers = {item["id"]: str(item.get("answer") or "") for item in json.loads(args.answers.read_text(encoding="utf-8"))}
    evaluations = []
    for case_id, case in cases.items():
        answer = answers.get(case_id, "")
        normalized = answer.casefold()
        evaluations.append(
            {
                "id": case_id,
                "answer_present": bool(answer.strip()),
                "keyword_coverage": all(keyword.casefold() in normalized for keyword in case["expected_keywords"]),
                "refusal_expected": bool(case.get("expect_refusal")),
                "refusal_present": any(marker in normalized for marker in REFUSAL_MARKERS),
            }
        )
    report = {"cases": evaluations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
