"""Run the fixed manual retrieval set against one indexed RAG document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_text(chunk: dict) -> str:
    return str(chunk.get("content") or "").casefold()


def case_match(case: dict, chunks: list[dict]) -> bool:
    if not case["expected_pages"]:
        return False
    pages = {chunk.get("page_number") for chunk in chunks}
    keywords = case["expected_keywords"]
    combined_text = "\n".join(chunk_text(chunk) for chunk in chunks)
    return set(case["expected_pages"]).issubset(pages) and all(keyword.casefold() in combined_text for keyword in keywords)


def reciprocal_rank(case: dict, chunks: list[dict]) -> float:
    expected_pages = set(case["expected_pages"])
    if not expected_pages:
        return 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get("page_number") in expected_pages:
            return 1 / rank
    return 0.0


def evaluate_cases(cases: list[dict], results_by_id: dict[str, list[dict]]) -> dict:
    answerable = [case for case in cases if case["expected_pages"]]
    evaluations = []
    for case in cases:
        chunks = results_by_id.get(case["id"], [])
        evaluations.append(
            {
                "id": case["id"],
                "type": case["type"],
                "hit_at_5": case_match(case, chunks[:5]),
                "hit_at_10": case_match(case, chunks[:10]),
                "mrr": reciprocal_rank(case, chunks),
                "top_chunks": [
                    {
                        key: chunk.get(key)
                        for key in ("chunk_id", "page_number", "score", "rrf_score", "rerank_score")
                    }
                    for chunk in chunks[:10]
                ],
            }
        )
    answerable_evaluations = [item for item in evaluations if item["id"] in {case["id"] for case in answerable}]
    count = len(answerable_evaluations)
    return {
        "metrics": {
            "hit_at_5": sum(item["hit_at_5"] for item in answerable_evaluations) / count if count else 0,
            "hit_at_10": sum(item["hit_at_10"] for item in answerable_evaluations) / count if count else 0,
            "mrr": sum(item["mrr"] for item in answerable_evaluations) / count if count else 0,
        },
        "cases": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True, help="Indexed PDF notebook/document ID")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("rag_manual_evaluation.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-mode", choices=("hybrid", "near_vector"), default="hybrid")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from pipeline.retrieve_answer import retrieve_chunks
    from services.api import load_llm_settings
    from services.config import load_settings

    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings()
    llm_settings = load_llm_settings()
    cases = load_cases(args.cases)
    results = {
        case["id"]: retrieve_chunks(case["question"], args.document_id, settings, llm_settings, args.search_mode)
        for case in cases
    }
    report = {
        "document_id": args.document_id,
        "search_mode": args.search_mode,
        "evaluation": evaluate_cases(cases, results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["evaluation"]["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
