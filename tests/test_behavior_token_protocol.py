import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_common import behavior_settings
from common import load_config


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


if __name__ == "__main__":
    unittest.main()
