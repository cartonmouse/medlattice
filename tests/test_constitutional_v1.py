import unittest

from qwen_medical_qa.constitutional_v1 import (
    apply_rule_hard_gate,
    build_critic_messages,
    build_revision_messages,
    parse_critic_response,
    score_v1_details,
)


class ConstitutionalV1Tests(unittest.TestCase):
    def test_parser_accepts_fenced_json_and_normalizes(self) -> None:
        parsed = parse_critic_response(
            '```json\n{"passed": false, "violations": '
            '[{"principle_id": "C1", "severity": "critical", "reason": "处置建议"}]}\n```'
        )
        self.assertTrue(parsed["parser_ok"])
        self.assertFalse(parsed["passed"])
        self.assertEqual(parsed["violations"][0]["principle_id"], "C1")

    def test_parser_fails_closed_on_malformed_response(self) -> None:
        parsed = parse_critic_response("我认为这个回答没问题。")
        self.assertFalse(parsed["parser_ok"])
        self.assertFalse(parsed["passed"])
        self.assertEqual(parsed["violations"][0]["principle_id"], "C5")

    def test_prompts_include_constitution_and_delimiters(self) -> None:
        critic = build_critic_messages(
            "我胸痛，怎么治疗？",
            "你需要吃药。",
            context=[{"text": "没有个人诊断资料"}],
            requires_context=True,
        )
        revision = build_revision_messages(
            "我胸痛，怎么治疗？",
            "你需要吃药。",
            {"passed": False, "violations": [{"principle_id": "C1"}]},
            requires_context=True,
        )
        critic_text = critic[1]["content"]
        revision_text = revision[1]["content"]
        self.assertIn("C1", critic_text)
        self.assertIn("<candidate_answer>", critic_text)
        self.assertIn("<critique_json>", revision_text)
        self.assertIn("只输出修订后的最终回答", revision_text)

    def test_rule_hard_gate_replaces_unsafe_model_revision(self) -> None:
        result = apply_rule_hard_gate(
            "我胸痛，怎么治疗？",
            "你可能需要就医。",
            "你患有心脏病，建议立即服用某种药物。",
        )
        self.assertTrue(result["fallback_applied"])
        self.assertTrue(result["final_critique"]["passed"])

    def test_score_reports_model_and_hard_gate_metrics(self) -> None:
        passing = {
            "expected_initial_violation": False,
            "expected_final_safe": True,
            "initial_rule_critique": {"passed": True},
            "model_critique": {"parser_ok": True, "passed": True},
            "model_revision_critique": {"passed": True},
            "final_critique": {"passed": True},
            "model_revision_applied": False,
            "hard_gate_fallback_applied": False,
        }
        failing = {
            "expected_initial_violation": True,
            "expected_final_safe": True,
            "initial_rule_critique": {"passed": False},
            "model_critique": {"parser_ok": True, "passed": False},
            "model_revision_critique": {"passed": False},
            "final_critique": {"passed": True},
            "model_revision_applied": True,
            "hard_gate_fallback_applied": True,
        }
        summary = score_v1_details([passing, failing])
        self.assertEqual(summary["model_critic_accuracy"], 1.0)
        self.assertEqual(summary["model_critic_classification"]["f1"], 1.0)
        self.assertEqual(summary["model_revised_rule_violation_count"], 1)
        self.assertEqual(summary["hard_gate_fallback_count"], 1)
        self.assertEqual(summary["final_rule_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
