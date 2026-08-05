import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.spreadsheet_analysis import answer_statistics_question, execute_plan, is_statistics_question, workbook_code_mappings
from services.vectordb import RagServiceError


def test_group_count_uses_pandas_not_retrieval():
    frame = pd.DataFrame({"cancer": ["lung", "breast", "lung"], "age": ["40", "50", "60"]})

    result = execute_plan(frame, {"operation": "group_count", "target_column": None, "filters": [], "group_by": ["cancer"], "limit": 100})

    assert result["matched_row_count"] == 3
    assert result["value"] == [{"cancer": "breast", "count": 1}, {"cancer": "lung", "count": 2}]


def test_plan_rejects_unknown_column():
    with pytest.raises(RagServiceError, match="Unknown spreadsheet column"):
        execute_plan(pd.DataFrame({"age": ["40"]}), {"operation": "mean", "target_column": "missing", "filters": [], "group_by": [], "limit": 10})


def test_chinese_statistics_question_is_routed_to_pandas():
    assert is_statistics_question("肺癌有多少筆資料？")
    assert not is_statistics_question("癌症欄位代表什麼？")


def test_null_plan_limit_uses_the_safe_default():
    result = execute_plan(pd.DataFrame({"age": ["40", "50"]}), {"operation": "count", "target_column": None, "filters": [], "group_by": [], "limit": None})

    assert result["value"] == 2


def test_total_count_is_returned_as_a_readable_answer(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("name\nA\nB\n", encoding="utf-8")

    result = answer_statistics_question(path, "資料總筆數為多少", object())

    assert result["answer"] == "Sheet1 的資料總筆數為 2 筆。"


def test_definition_sheet_decodes_filter_codes(tmp_path, monkeypatch):
    path = tmp_path / "records.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"\u6027\u5225": ["1", "2"]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame({"\u4e2d\u6587\u6b04\u4f4d\u540d\u7a31": ["\u6027\u5225"], "define": ["1:\u7537\u6027\u30022:\u5973\u6027\u3002"]}).to_excel(writer, sheet_name="Definitions", index=False)
    monkeypatch.setattr("services.spreadsheet_analysis.spreadsheet_plan", lambda *_: {"operation": "count", "target_column": None, "filters": [{"column": "\u6027\u5225", "operator": "eq", "value": "1"}], "group_by": [], "limit": 100})

    result = answer_statistics_question(path, "\u6027\u5225\u70ba\u7537\u6027\u7684\u591a\u5c11", object())

    assert workbook_code_mappings(path)["\u6027\u5225"]["1"] == "\u7537\u6027"
    assert result["answer"] == "Sheet1 的\u6027\u5225為\u7537\u6027筆數為 1 筆。"
