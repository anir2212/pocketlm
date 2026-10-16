import unittest
from pocketlm import MockDevice, corpus, dedup, evaluate_baseline, parse_request, validate


class PocketTests(unittest.TestCase):
    def test_schema(self):
        self.assertTrue(validate({"tool": "set_dpi", "arguments": {"dpi": 800}}))
        self.assertFalse(validate({"tool": "set_dpi", "arguments": {"dpi": "800"}}))
        self.assertFalse(validate({"tool": "invented", "arguments": {}}))

    def test_corpus(self):
        rows = corpus()
        self.assertEqual(len(rows), len(dedup(rows + rows)))

    def test_mock_tool(self):
        call = parse_request("make my keyboard purple")
        self.assertEqual(MockDevice().execute(call)["lighting"]["keyboard"], "purple")
        self.assertIn("refusal", parse_request("order lunch"))
        self.assertEqual(evaluate_baseline(corpus())["total"], len(corpus()))


if __name__ == "__main__":
    unittest.main()
