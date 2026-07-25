"""Run the fixed manual retrieval set against one indexed RAG document."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_text(chunk: dict) -> str:
    return str(chunk.get("content") or "").casefold()


def case_match(case: dict, chunks: list[dict]) -> bool:
    expected_pages = case.get("expected_pages", [])
    expected_field_code = case.get("expected_field_code")
    if not expected_pages and not expected_field_code:
        return False
    pages = {chunk.get("page_number") for chunk in chunks}
    keywords = case["expected_keywords"]
    combined_text = "\n".join(chunk_text(chunk) for chunk in chunks)
    page_match = not expected_pages or set(expected_pages).issubset(pages)
    field_match = not expected_field_code or any(chunk.get("field_code") == expected_field_code for chunk in chunks)
    return page_match and field_match and all(keyword.casefold() in combined_text for keyword in keywords)


def reciprocal_rank(case: dict, chunks: list[dict]) -> float:
    expected_pages = set(case.get("expected_pages", []))
    expected_field_code = case.get("expected_field_code")
    if not expected_pages and not expected_field_code:
        return 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if (not expected_pages or chunk.get("page_number") in expected_pages) and (
            not expected_field_code or chunk.get("field_code") == expected_field_code
        ):
            return 1 / rank
    return 0.0


def evaluate_cases(cases: list[dict], results_by_id: dict[str, list[dict]]) -> dict:
    answerable = [case for case in cases if not case.get("expect_refusal")]
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
                        for key in ("chunk_id", "page_number", "block_type", "field_code", "detail_type", "score", "rrf_score", "rerank_score")
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
    parser.add_argument("--case-id", action="append", help="Evaluate only this case ID; repeat for multiple cases")
    parser.add_argument("--no-query-rewrite", action="store_true", help="Use only the original question while validating retrieval")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from pipeline.retrieve_answer import retrieve_chunks
    from services.api import load_llm_settings
    from services.config import load_settings

    load_dotenv(PROJECT_ROOT / ".env")
    settings = load_settings()
    llm_settings = load_llm_settings()
    cases = load_cases(args.cases)
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected_ids]
        missing_ids = selected_ids - {case["id"] for case in cases}
        if missing_ids:
            parser.error(f"Unknown case IDs: {', '.join(sorted(missing_ids))}")
    rewrite_context = patch("pipeline.retrieve_answer.build_search_queries", side_effect=lambda question, _: [question]) if args.no_query_rewrite else nullcontext()
    with rewrite_context:
        results = {
            case["id"]: retrieve_chunks(case["question"], args.document_id, settings, llm_settings, args.search_mode)
            for case in cases
        }
    report = {
        "document_id": args.document_id,
        "search_mode": args.search_mode,
        "query_rewrite": not args.no_query_rewrite,
        "evaluation": evaluate_cases(cases, results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["evaluation"]["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
