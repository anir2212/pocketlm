from quant.build_grammar import build
from tools_api.executor import SCHEMAS


def test_grammar_has_every_tool_and_only_known_output_kinds():
    grammar = build()
    for tool in SCHEMAS:
        assert f"tool-{tool.replace('_', '-')} ::= " in grammar
    assert "call-output" in grammar
    assert "refusal-output" in grammar
    assert "clarification-output" in grammar
