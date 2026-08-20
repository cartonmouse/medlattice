import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_cmb_rag import build_exam_rag_prompt


class CmbRagPromptTests(unittest.TestCase):
    def test_prompt_contains_context_and_letter_constraint(self) -> None:
        prompt = build_exam_rag_prompt(
            "当前题目",
            [
                {
                    "chunk_id": "cmb-train-00000#chunk-000",
                    "title": "训练例题",
                    "text": "题目内容\n参考答案：A",
                }
            ],
        )
        self.assertIn("cmb-train-00000#chunk-000", prompt)
        self.assertIn("当前题目", prompt)
        self.assertIn("只输出当前问题的一个选项字母", prompt)

    def test_prompt_handles_empty_context(self) -> None:
        self.assertIn("[无相关训练例题]", build_exam_rag_prompt("问题", []))
