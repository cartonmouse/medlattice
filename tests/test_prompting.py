from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.prompting import build_messages, format_user_prompt


class PromptingTest(unittest.TestCase):
    record = {
        "id": "demo-1",
        "question": "下列哪项正确？",
        "choices": {"A": "错误", "B": "正确"},
    }

    def test_formats_choice_prompt(self) -> None:
        prompt = format_user_prompt(self.record)
        self.assertIn("A. 错误", prompt)
        self.assertIn("只输出一个选项字母", prompt)

    def test_builds_training_messages(self) -> None:
        messages = build_messages(self.record, assistant_content="B")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], "B")


if __name__ == "__main__":
    unittest.main()
