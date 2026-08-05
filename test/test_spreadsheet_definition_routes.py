from services.spreadsheet_analysis import definition_count_plan


def test_definition_count_plan_uses_the_code_mapping_without_an_llm():
    mappings = {"\u6027\u5225": {"9": "\u4e0d\u8a73"}}

    plan = definition_count_plan("\u6027\u5225\u70ba\u4e0d\u8a73\u7684\u8cc7\u6599\u6709\u591a\u5c11\u7b46", mappings)

    assert plan["filters"] == [{"column": "\u6027\u5225", "operator": "eq", "value": "9"}]
