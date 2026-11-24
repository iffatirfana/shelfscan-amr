import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webots/tools"))

from make_world import CAM_Z, ITEM, LIFT_POS, make, wbt  # noqa: E402
from shelfscan import scissor as sc             # noqa: E402
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


class ScissorTests(unittest.TestCase):
    """The scissor stack is driven, not animated by hand, so its geometry has
    to close exactly at every lift height the audit uses."""

    def heights(self):
        return [k / 100.0 for k in range(0, int(LIFT_POS[-1] * 100) + 1)]

    def test_links_close_and_carry_the_camera(self):
        for h in self.heights():
            theta, spread, dz, stack = sc.geometry(h)
            # every bar is exactly LINK_L long and ends on the row pins
            for i in range(sc.N_STAGES):
                for (x0, z0), (x1, z1) in sc.bar_ends(h, i):
                    self.assertAlmostEqual(math.hypot(x1 - x0, z1 - z0),
                                           sc.LINK_L, places=12)
                    for z in (z0, z1):
                        self.assertAlmostEqual((z - sc.BASE_Z) / dz,
                                               round((z - sc.BASE_Z) / dz),
                                               places=9)
            # one stage rises LINK_L*sin(theta) and spreads LINK_L*cos(theta)
            self.assertAlmostEqual(dz, sc.LINK_L * math.sin(theta), places=12)
            self.assertAlmostEqual(spread, sc.LINK_L * math.cos(theta), places=12)
            # the top pivot row puts the camera exactly where the lift does
            self.assertAlmostEqual(sc.BASE_Z + stack + sc.TOP_DROP,
                                   CAM_Z + h, places=9)

    def test_stack_folds_and_stays_behind_the_camera(self):
        lo, hi = sc.geometry(0.0), sc.geometry(LIFT_POS[-1])
        self.assertLess(lo[0], hi[0])            # folds flat low, steep high
        self.assertGreater(lo[1], hi[1])         # and so takes more room low
        self.assertLess(math.degrees(lo[0]), 15.0)
        self.assertGreater(math.degrees(hi[0]), 60.0)
        for h in self.heights():
            for _, (x, _, _), _ in sc.poses(h):
                self.assertLessEqual(x + sc.CROSS_R, sc.X_MAX)


if __name__ == "__main__":
    unittest.main()
