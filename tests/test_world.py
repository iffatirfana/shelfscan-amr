import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webots/tools"))

from make_world import CAM_Z, ITEM, make, wbt  # noqa: E402
from shelfscan.robot_utils import Driver        # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
import aggregate_results                         # noqa: E402


class WorldTests(unittest.TestCase):
    def test_world_needs_no_remote_proto(self):
        text = wbt(make(1), "R2025a")
        self.assertNotIn("EXTERNPROTO", text)
        self.assertNotIn("webots://", text)
        self.assertTrue(text.startswith("#VRML_SIM R2025a utf8"))

    def test_robot_devices_match_driver(self):
        text = wbt(make(1), "R2025a", "data_capture")
        for dev in ('"left wheel motor"', '"right wheel motor"', '"gps"', '"imu"',
                    '"camera"', "DEF CAM Camera", 'controller "data_capture"'):
            self.assertIn(dev, text)

    def test_camera_at_item_height(self):
        lay = make(1)
        z = next(iter(lay["slots"].values()))["z"]
        self.assertLess(abs(CAM_Z - z), ITEM[2] / 2)

    def test_layout_deterministic_per_seed(self):
        self.assertEqual(make(7), make(7))
        self.assertNotEqual(make(7)["slots"], make(8)["slots"])

    def test_start_pose_in_free_aisle(self):
        lay = make(1)
        self.assertEqual(lay["start_pose"][0], Driver.AISLE_X)
        xs = [s["x"] for s in lay["slots"].values()]
        self.assertLess(Driver.AISLE_X, min(xs) - 0.45)


class AggregateTests(unittest.TestCase):
    def test_summary_from_run_files(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            runs.mkdir()
            for pol, acc in (("V1_fixed", 0.5), ("V4_active", 1.0)):
                (runs / f"{pol}_seed1.json").write_text(json.dumps(dict(
                    policy=pol, seed=1, n_slots=2, correct=int(acc * 2), slot_accuracy=acc,
                    decided=2, decided_accuracy=acc, unverified=0, unverified_rate=0.0,
                    n_truly_not_empty=2, false_empty=0, false_empty_rate=0.0, n_truly_empty=0,
                    empty_caught=0, true_discrepancies=1, caught=1, detection_rate=1.0,
                    false_alarms=0, audit_time_s=10.0, distance_m=1.0, frames=2)))
            aggregate_results.main(str(runs), d)
            rows = (Path(d) / "results.csv").read_text().splitlines()
            self.assertEqual(len(rows), 3)
            md = (Path(d) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("100.0 ± 0.0%", md)
            self.assertIn("50.0 ± 0.0%", md)


if __name__ == "__main__":
    unittest.main()
