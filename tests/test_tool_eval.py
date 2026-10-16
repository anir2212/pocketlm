from eval.tool_eval import bootstrap, item_metrics, summarize


def test_structured_metrics_detect_wrong_or_invented_tool():
    row = {"target": {"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}}
    correct = item_metrics(row, '{"calls":[{"tool":"set_dpi","arguments":{"dpi":800}}]}')
    invented = item_metrics(row, '{"calls":[{"tool":"invented","arguments":{"dpi":800}}]}')
    invalid = item_metrics(row, 'not json')
    scores = summarize([correct, invented, invalid])
    assert scores["json_validity"] == 2 / 3
    assert scores["full_call_exact_match"] == 1 / 3
    assert scores["hallucinated_tool_rate"] == 1 / 3
    assert correct["argument_tp"] == 1
    assert bootstrap([correct], repetitions=10)["full_call_exact_match"] == [1.0, 1.0]


def test_malformed_calls_are_scored_without_crashing():
    row = {"target": {"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}}
    malformed = item_metrics(row, '{"calls":[{"tool":[],"arguments":{}},null]}')
    assert malformed["json_valid"]
    assert not malformed["target_valid"]
    assert not malformed["tool_exact"]
    assert malformed["hallucinated_tool"]
