from train.make_preferences import rejection_reason


def test_malformed_tool_name_is_a_safe_negative():
    gold = {"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}
    assert rejection_reason(gold, '{"calls":[{"tool":[],"arguments":{}}]}') == (
        "hallucinated_tool"
    )
