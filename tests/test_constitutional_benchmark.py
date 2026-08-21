import unittest

from scripts.prepare_constitutional_benchmark import build_rows, validate_rows


class ConstitutionalBenchmarkTests(unittest.TestCase):
    def test_builds_120_rows_with_balanced_group_splits(self) -> None:
        rows = build_rows()
        self.assertEqual(len(rows), 120)
        validate_rows(rows)
        self.assertEqual(sum(row["expected_initial_violation"] for row in rows), 60)
        self.assertEqual(
            {split: sum(row["split"] == split for row in rows) for split in ("dev", "holdout", "challenge")},
            {"dev": 60, "holdout": 40, "challenge": 20},
        )

    def test_variants_of_a_family_never_cross_splits(self) -> None:
        rows = build_rows()
        family_splits: dict[str, set[str]] = {}
        for row in rows:
            family_splits.setdefault(row["family_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))

    def test_rows_keep_auditable_labels_and_context(self) -> None:
        rows = build_rows()
        for row in rows:
            self.assertIn("label_source", row)
            self.assertIn("expected_initial_violation", row)
            self.assertTrue(row["expected_final_safe"])
            if row["requires_context"]:
                if row["family_id"] in {"u07", "u08"}:
                    self.assertEqual(row["context"], [])
                else:
                    self.assertTrue(row["context"])


if __name__ == "__main__":
    unittest.main()
