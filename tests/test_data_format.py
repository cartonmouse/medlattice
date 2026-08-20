from __future__ import annotations

import json
import unittest
from pathlib import Path


class SampleDataFormatTest(unittest.TestCase):
    def test_sample_records_have_required_fields(self) -> None:
        data_path = Path(__file__).parents[1] / "data" / "sample_medical_qa.jsonl"
        required = {"id", "question", "reference_answer", "source", "license"}
        with data_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            self.assertTrue(required.issubset(row))
            self.assertTrue(row["question"])
            self.assertTrue(row["reference_answer"])


if __name__ == "__main__":
    unittest.main()

