import unittest

from qwen_medical_qa.safety import ABSTENTION_ANSWER, assess_question


class SafetyPolicyTests(unittest.TestCase):
    def test_high_risk_intent_abstains_even_with_context(self) -> None:
        decision = assess_question(
            "请根据我的症状给我诊断和治疗方案。",
            [{"score": 0.95, "doc_id": "term-symptom"}],
        )
        self.assertTrue(decision.abstain)
        self.assertEqual(decision.reason, "high-risk-intent")

    def test_missing_context_abstains(self) -> None:
        decision = assess_question("Python 如何读取 CSV？", [])
        self.assertTrue(decision.abstain)
        self.assertEqual(decision.reason, "no-retrieved-context")

    def test_in_scope_definition_can_pass(self) -> None:
        decision = assess_question(
            "患者自己感觉到的异常表现属于什么术语？",
            [{"score": 0.65, "doc_id": "term-symptom"}],
        )
        self.assertFalse(decision.abstain)
        self.assertEqual(ABSTENTION_ANSWER, "资料不足，无法判断。")

    def test_low_score_threshold_abstains(self) -> None:
        decision = assess_question(
            "一个定义问题",
            [{"score": 0.4, "doc_id": "term-symptom"}],
            min_score=0.6,
        )
        self.assertTrue(decision.abstain)
        self.assertEqual(decision.reason, "below-retrieval-threshold")
