from __future__ import annotations

import unittest


class PreferenceEvalTests(unittest.TestCase):
    def test_mean_completion_logprob_masks_prompt_tokens(self) -> None:
        import torch

        from qwen_medical_qa.preference_eval import mean_completion_logprob

        # At position 1 the target is token 2; at position 2 the target is token 3.
        logits = torch.full((1, 4, 4), -10.0)
        logits[0, 0, 1] = 10.0  # prompt token prediction, must be ignored
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        input_ids = torch.tensor([[0, 1, 2, 3]])
        score = mean_completion_logprob(logits, input_ids, prompt_length=2)
        self.assertGreater(score, -0.01)


if __name__ == "__main__":
    unittest.main()
