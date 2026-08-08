import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_behavior import memory_guided_batch_size


class BehaviorGpuBatchingTests(unittest.TestCase):
    def test_memory_measurement_grows_batch_toward_target(self):
        gib = 1024**3
        selected = memory_guided_batch_size(
            current_batch_size=16,
            minimum_batch_size=2,
            maximum_batch_size=128,
            baseline_memory_bytes=16 * gib,
            peak_memory_bytes=24 * gib,
            target_memory_bytes=72 * gib,
            maximum_growth_factor=2.0,
        )
        self.assertEqual(selected, 32)

    def test_memory_measurement_respects_maximum(self):
        selected = memory_guided_batch_size(
            current_batch_size=64,
            minimum_batch_size=8,
            maximum_batch_size=96,
            baseline_memory_bytes=10,
            peak_memory_bytes=20,
            target_memory_bytes=880,
            maximum_growth_factor=2.0,
        )
        self.assertEqual(selected, 96)

    def test_memory_measurement_does_not_shrink_successful_batch(self):
        selected = memory_guided_batch_size(
            current_batch_size=32,
            minimum_batch_size=4,
            maximum_batch_size=128,
            baseline_memory_bytes=60,
            peak_memory_bytes=88,
            target_memory_bytes=88,
            maximum_growth_factor=2.0,
        )
        self.assertEqual(selected, 32)

    def test_reserved_memory_above_target_shrinks_next_batch(self):
        gib = 1024**3
        selected = memory_guided_batch_size(
            current_batch_size=32,
            minimum_batch_size=4,
            maximum_batch_size=128,
            baseline_memory_bytes=16 * gib,
            peak_memory_bytes=80 * gib,
            target_memory_bytes=72 * gib,
            maximum_growth_factor=2.0,
        )
        self.assertEqual(selected, 28)


if __name__ == "__main__":
    unittest.main()
