import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_common import behavior_settings
from common import load_config
from generate_behavior import audit_prompt_lengths, prompt_forbidden_token_ids


class FakeTokenizer:
    all_special_ids = [0, 1, 2, 3]
    unk_token_id = 3

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, prompts, **kwargs):
        return {"input_ids": [list(self.token_ids) for _ in prompts]}


class BehaviorTokenProtocolTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        return load_config(root / "configs" / "xglm_1b7_behavior_v1.json")

    def test_formal_config_uses_plain_text_prompt_encoding(self):
        settings = behavior_settings(self.config())
        self.assertFalse(settings["decoding"]["prompt_add_special_tokens"])

    def test_special_token_prompt_encoding_is_rejected(self):
        cfg = self.config()
        cfg["behavior_v1"]["decoding"]["prompt_add_special_tokens"] = True
        with self.assertRaisesRegex(ValueError, "must not add tokenizer special tokens"):
            behavior_settings(cfg)

    def test_prompt_unknown_token_is_audited_but_not_rejected(self):
        tokenizer = FakeTokenizer([10, 3, 11])
        forbidden = prompt_forbidden_token_ids(tokenizer)
        summary, lengths = audit_prompt_lengths(
            tokenizer,
            [{"task_id": "task-1", "prompt": "text"}],
            {"prompt_add_special_tokens": False, "max_prompt_tokens": 16},
            forbidden,
        )
        self.assertEqual(forbidden, {0, 1, 2})
        self.assertEqual(summary["prompt_unknown_token_count"], 1)
        self.assertEqual(summary["prompt_unknown_token_fraction"], 1 / 3)
        self.assertEqual(summary["tasks_with_prompt_unknown_tokens"], 1)
        self.assertEqual(lengths["task-1"], 3)

    def test_prompt_control_token_is_rejected(self):
        tokenizer = FakeTokenizer([10, 2, 11])
        with self.assertRaisesRegex(ValueError, "prompt contains tokenizer control tokens"):
            audit_prompt_lengths(
                tokenizer,
                [{"task_id": "task-1", "prompt": "text"}],
                {"prompt_add_special_tokens": False, "max_prompt_tokens": 16},
                prompt_forbidden_token_ids(tokenizer),
            )


if __name__ == "__main__":
    unittest.main()
