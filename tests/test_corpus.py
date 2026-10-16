import copy

from data.build_mix import convert_glaive
from data.build_synthetic import generate
from data.clean import clean_row
from data.dedup import deduplicate
from tools_api.executor import MockDevice, validate_target

CONFIG = {
    "seed": 17, "domain_examples": 1000, "out_of_scope_fraction": 0.15,
    "clarification_fraction": 0.10, "multi_call_fraction": 0.15,
    "heldout_tools": ["close_app", "set_sleep_timer"],
}


def test_schema_rejects_unknown_tool_and_bad_args():
    assert validate_target({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]})
    assert not validate_target({"calls": [{"tool": "made_up", "arguments": {}}]})
    assert not validate_target({"calls": [{"tool": "set_dpi", "arguments": {"dpi": "800"}}]})
    assert not validate_target({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800,
                                                                               "surprise": 1}}]})


def test_mock_device_executes_only_validated_calls():
    device = MockDevice()
    assert device.execute({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 1200}}]})[
        "dpi"] == 1200
    try:
        device.execute({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 999}}]})
        raise AssertionError("bad call was executed")
    except ValueError:
        pass


def test_splits_hold_out_two_tools_and_validate_every_target():
    rows = generate(CONFIG)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    assert train and test
    assert all(validate_target(row["target"]) for row in rows)
    assert all(call["tool"] not in CONFIG["heldout_tools"]
               for row in train if "calls" in row["target"] for call in row["target"]["calls"])
    assert any(call["tool"] in CONFIG["heldout_tools"]
               for row in test if "calls" in row["target"] for call in row["target"]["calls"])


def test_clean_and_near_dedup_preserves_test_row():
    row = generate({**CONFIG, "domain_examples": 1})[0]
    bad = copy.deepcopy(row)
    bad["prompt"] = "email me at person@example.com"
    assert clean_row(bad)[1] == "pii_like"
    test = {**row, "split": "test", "id": "test"}
    train = {**row, "split": "train", "id": "train"}
    kept, counts = deduplicate([train, test])
    assert len(kept) == 1 and kept[0]["id"] == "test"
    assert counts["exact_removed"] == 1


def test_public_function_call_is_checked_against_its_own_schema():
    raw = {
        "system": 'SYSTEM: tools {"name":"get_weather","parameters":'
                  '{"type":"object","properties":{"city":{"type":"string"}},'
                  '"required":["city"],"additionalProperties":false}}',
        "chat": 'USER: weather in Singapore?\n\nASSISTANT: <functioncall> '
                '{"name": "get_weather", "arguments": \'{"city":"Singapore"}\'} '
                '<|endoftext|>',
    }
    row = convert_glaive(raw, 1)
    assert row is not None
    assert clean_row(row)[1] is None
    wrong = copy.deepcopy(row)
    wrong["target"]["calls"][0]["arguments"] = {"city": 4}
    assert clean_row(wrong)[1] == "schema"
