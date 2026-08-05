from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import openai

from services.api import LLMSettings, llm_client
from services.vectordb import RagServiceError

ALLOWED_OPERATIONS = {"count", "distinct_count", "sum", "mean", "median", "min", "max", "group_count", "percentage", "top_n"}
ALLOWED_FILTERS = {"eq", "ne", "contains", "gt", "gte", "lt", "lte", "in", "not_in", "between", "is_null", "not_null"}


def workbook_frames(path: Path) -> dict[str, pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            frames = {"Sheet1": pd.read_csv(path, dtype=str, keep_default_na=False)}
        else:
            frames = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    except (OSError, ValueError, ImportError) as error:
        raise RagServiceError("Spreadsheet parsing failed.", 400) from error
    return {name: frame.dropna(axis=0, how="all").dropna(axis=1, how="all") for name, frame in frames.items()}


def data_sheet_names(frames: dict[str, pd.DataFrame]) -> list[str]:
    names = [name for name in frames if not re.search(r"definition|dictionary|欄位|代碼", name, re.I)]
    return names or list(frames)


def workbook_code_mappings(path: Path) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    field_columns = ("\u4e2d\u6587\u6b04\u4f4d\u540d\u7a31", "\u53f0\u5927\u96f2\u6797\u6b04\u4f4d\u540d\u7a31", "\u53f0\u5927\u9ad4\u7cfb\u91ab\u6574\u5eab\u6b04\u4f4d\u540d\u7a31")
    for sheet_name, frame in workbook_frames(path).items():
        if sheet_name.casefold() not in {"definitions", "definition", "dictionary"}:
            continue
        for _, row in frame.fillna("").iterrows():
            definition = str(row.get("define", "")).strip()
            pairs = re.findall(r"([^:\uff1a\s]+)\s*[:\uff1a]\s*([^\u3002;\uff1b]+)", definition)
            if not pairs:
                continue
            for field_column in field_columns:
                field_name = str(row.get(field_column, "")).strip()
                if field_name:
                    mappings.setdefault(field_name, {}).update({code.strip(): label.strip() for code, label in pairs})
    return mappings


def definition_count_plan(question: str, mappings: dict[str, dict[str, str]]) -> dict | None:
    if not re.search("\\u591a\\u5c11|\\u5e7e\\u7b46|\\u7b46\\u6578", question):
        return None
    for field_name, values in mappings.items():
        for code, label in values.items():
            if field_name in question and label in question:
                return {
                    "operation": "count",
                    "target_column": None,
                    "filters": [{"column": field_name, "operator": "eq", "value": code}],
                    "group_by": [],
                    "sort": [],
                    "limit": 100,
                }
    return None


def profile_workbook(path: Path) -> list[dict]:
    profiles = []
    for sheet_name in data_sheet_names(workbook_frames(path)):
        frame = workbook_frames(path)[sheet_name]
        profiles.append({
            "sheet_name": sheet_name,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "columns": [{"name": str(column), "dtype": str(frame[column].dtype), "null_count": int((frame[column] == "").sum()), "unique_count": int(frame[column].nunique()), "sample_values": frame[column].astype(str).head(3).tolist()} for column in frame.columns],
        })
    return profiles


def _json_response(prompt: str, llm_settings: LLMSettings) -> dict:
    try:
        response = llm_client(llm_settings).chat.completions.create(model=llm_settings.model, messages=[{"role": "user", "content": prompt}], temperature=0)
        content = (response.choices[0].message.content or "").strip()
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content))
    except (openai.OpenAIError, ValueError, json.JSONDecodeError) as error:
        raise RagServiceError("Could not create a valid spreadsheet query plan.", 400) from error


def spreadsheet_plan(question: str, profile: dict, llm_settings: LLMSettings) -> dict:
    prompt = (
        "Return JSON only. Create a spreadsheet query plan for this question. "
        f"Allowed operations: {sorted(ALLOWED_OPERATIONS)}. Allowed filters: {sorted(ALLOWED_FILTERS)}. "
        "Schema: {operation,target_column,filters:[{column,operator,value}],group_by:[],sort:[],limit}. "
        f"Workbook profile: {json.dumps(profile, ensure_ascii=False)}\nQuestion: {question}"
    )
    return _json_response(prompt, llm_settings)


def validate_plan(plan: dict, frame: pd.DataFrame) -> None:
    if plan.get("operation") not in ALLOWED_OPERATIONS:
        raise RagServiceError("Unsupported spreadsheet operation.", 400)
    columns = set(map(str, frame.columns))
    for column in [plan.get("target_column"), *(plan.get("group_by") or [])]:
        if column and column not in columns:
            raise RagServiceError(f"Unknown spreadsheet column: {column}", 400)
    try:
        limit = int(plan.get("limit") or 100)
    except (TypeError, ValueError) as error:
        raise RagServiceError("Spreadsheet result limit must be a number.", 400) from error
    if limit < 1 or limit > 1000:
        raise RagServiceError("Spreadsheet result limit must be between 1 and 1000.", 400)
    for item in plan.get("filters") or []:
        if item.get("column") not in columns or item.get("operator") not in ALLOWED_FILTERS:
            raise RagServiceError("Invalid spreadsheet filter.", 400)


def _filtered(frame: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    result = frame.copy()
    for item in filters:
        column, operator, value = item["column"], item["operator"], item.get("value")
        series = result[column].astype(str)
        if operator == "eq": result = result[series == str(value)]
        elif operator == "ne": result = result[series != str(value)]
        elif operator == "contains": result = result[series.str.contains(str(value), case=False, na=False)]
        elif operator == "in": result = result[series.isin([str(x) for x in value])]
        elif operator == "not_in": result = result[~series.isin([str(x) for x in value])]
        elif operator == "is_null": result = result[series.str.strip() == ""]
        elif operator == "not_null": result = result[series.str.strip() != ""]
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if operator == "gt": result = result[numeric > float(value)]
            elif operator == "gte": result = result[numeric >= float(value)]
            elif operator == "lt": result = result[numeric < float(value)]
            elif operator == "lte": result = result[numeric <= float(value)]
            elif operator == "between": result = result[numeric.between(float(value[0]), float(value[1]))]
    return result


def execute_plan(frame: pd.DataFrame, plan: dict) -> dict:
    validate_plan(plan, frame)
    filtered = _filtered(frame, plan.get("filters") or [])
    operation, target, limit = plan["operation"], plan.get("target_column"), int(plan.get("limit") or 100)
    if operation == "count": value = len(filtered)
    elif operation == "distinct_count": value = int(filtered[target].nunique())
    elif operation in {"sum", "mean", "median", "min", "max"}:
        values = pd.to_numeric(filtered[target], errors="coerce").dropna()
        value = getattr(values, operation)() if operation != "sum" else values.sum()
        value = None if pd.isna(value) else float(value)
    elif operation == "group_count":
        groups = plan.get("group_by") or [target]
        value = filtered.groupby(groups, dropna=False).size().reset_index(name="count").head(limit).to_dict("records")
    elif operation == "percentage": value = round(100 * len(filtered) / len(frame), 2) if len(frame) else 0
    else:
        value = filtered.sort_values(target, ascending=False).head(limit)[[target]].to_dict("records")
    return {"operation": operation, "matched_row_count": len(filtered), "value": value, "filters": plan.get("filters") or []}


def is_statistics_question(question: str) -> bool:
    return bool(re.search(r"count|sum|average|mean|median|maximum|minimum|percentage|top\s*\d+|多少|幾筆|總和|平均|中位數|最大|最小|比例|百分比|前\s*\d+", question, re.I))


def answer_statistics_question(path: Path, question: str, llm_settings: LLMSettings) -> dict:
    frames = workbook_frames(path)
    sheets = data_sheet_names(frames)
    if not sheets:
        raise RagServiceError("No data sheet was found in this spreadsheet.", 400)
    sheet_name = sheets[0]
    frame = frames[sheet_name]
    mappings = workbook_code_mappings(path)
    direct_plan = definition_count_plan(question, mappings)
    if direct_plan:
        result = execute_plan(frame, direct_plan)
        filter_item = direct_plan["filters"][0]
        label = mappings[filter_item["column"]][filter_item["value"]]
        return {
            "answer": f"{sheet_name} \u7684{filter_item['column']}\u70ba{label}\u7b46\u6578\u70ba {result['value']:,} \u7b46\u3002",
            "plan": direct_plan,
            "result": result,
            "sheet_name": sheet_name,
        }
    profile = profile_workbook(path)[0]
    plan = {"operation": "count", "target_column": None, "filters": [], "group_by": [], "sort": [], "limit": 100} if re.search(r"資料總筆數|總筆數|總共多少筆", question) else spreadsheet_plan(question, profile, llm_settings)
    result = execute_plan(frame, plan)
    mappings = workbook_code_mappings(path)
    filters = result["filters"]
    filter_text = "、".join(
        f"{item['column']}為{mappings.get(item['column'], {}).get(str(item.get('value')), item.get('value'))}"
        for item in filters
        if item.get("operator") == "eq"
    )
    if result["operation"] == "count":
        answer = f"{sheet_name} 的{filter_text or '資料總'}筆數為 {result['value']:,} 筆。"
    else:
        answer = json.dumps({"sheet_name": sheet_name, **result}, ensure_ascii=False, indent=2)
    return {
        "answer": answer,
        "plan": plan,
        "result": result,
        "sheet_name": sheet_name,
    }
