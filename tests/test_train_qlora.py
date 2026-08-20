from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from train_qlora import tokenize_rows


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "PROMPT" if len(messages) == 2 else "PROMPTANSWER"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [1, 2, 3] if text == "PROMPT" else [1, 2, 3, 4]}


class TrainQLoRATest(unittest.TestCase):
    def test_masks_prompt_and_keeps_assistant_target(self) -> None:
        rows = [{"id": "demo-1", "messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}]
        dataset = tokenize_rows(rows, FakeTokenizer(), max_seq_length=8)
        self.assertEqual(dataset[0]["input_ids"], [1, 2, 3, 4])
        self.assertEqual(dataset[0]["labels"], [-100, -100, -100, 4])


if __name__ == "__main__":
    unittest.main()
