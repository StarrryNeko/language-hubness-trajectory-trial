import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from import_archived_outputs import import_outputs


class ImportOutputsTests(unittest.TestCase):
    def test_copy_import_refuses_to_overwrite_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "archive" / "model_a"
            source.mkdir(parents=True)
            (source / "artifact.txt").write_text("ok", encoding="utf-8")
            destination = root / "workspace" / "model_a"
            config = root / "model.json"
            config.write_text(json.dumps({"output_dir": str(destination)}), encoding="utf-8")
            suite = root / "suite.json"
            suite.write_text(json.dumps({"configs": [config.name]}), encoding="utf-8")
            records = import_outputs(source.parent, suite, "copy")
            self.assertEqual(records[0]["status"], "COPY")
            self.assertEqual((destination / "artifact.txt").read_text(encoding="utf-8"), "ok")
            with self.assertRaises(FileExistsError):
                import_outputs(source.parent, suite, "copy")


if __name__ == "__main__":
    unittest.main()

