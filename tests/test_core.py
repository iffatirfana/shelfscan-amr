import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfscan.reasoning import classify               # noqa: E402
from shelfscan.robot_utils import (results_dir,        # noqa: E402
                                   slot_matches, under_root)
from shelfscan.twin import DigitalTwin                 # noqa: E402
from shelfscan.types import EMPTY_CLASS, Observation, Status  # noqa: E402
from shelfscan.viewpoint import choose_viewpoint       # noqa: E402


def obs(label, conf=0.9, vis=0.95, side=None, vp="center", sid="S0-0"):
    return Observation(sid, label, conf, vis, side, vp)


class TestClassify(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(classify(obs("sku_A"), "sku_A"), Status.OK)

    def test_misplaced(self):
        self.assertEqual(classify(obs("sku_B"), "sku_A"), Status.MISPLACED)

    def test_empty_needs_visibility(self):
        self.assertEqual(classify(obs(EMPTY_CLASS), "sku_A"), Status.EMPTY)
        self.assertEqual(classify(obs(EMPTY_CLASS, vis=0.3), "sku_A"), Status.UNVERIFIED)

    def test_nothing_detected_is_not_empty(self):
        self.assertEqual(classify(obs(None, conf=0.0), "sku_A"), Status.UNVERIFIED)

    def test_low_confidence_unverified(self):
        self.assertEqual(classify(obs("sku_A", conf=0.4), "sku_A"), Status.UNVERIFIED)

    def test_expected_empty_slot(self):
        self.assertEqual(classify(obs(EMPTY_CLASS), None), Status.OK)
        self.assertEqual(classify(obs("sku_A"), None), Status.MISPLACED)


class TestTwin(unittest.TestCase):
    def test_decisive_replaces_unverified(self):
        t = DigitalTwin({"S0-0": "sku_A"})
        t.update(obs("sku_A", conf=0.3))
        self.assertEqual(t.slots["S0-0"].status, Status.UNVERIFIED)
        t.update(obs("sku_A", conf=0.9, vp="left"))
        self.assertEqual(t.slots["S0-0"].status, Status.OK)

    def test_later_unverified_keeps_better_status(self):
        t = DigitalTwin({"S0-0": "sku_A"})
        t.update(obs("sku_B", conf=0.9))
        t.update(obs("sku_B", conf=0.2))
        self.assertEqual(t.slots["S0-0"].status, Status.MISPLACED)

    def test_summary_and_report(self):
        t = DigitalTwin({"S0-0": "sku_A", "S0-1": "sku_B"})
        t.update(obs("sku_A", sid="S0-0"))
        self.assertEqual(t.summary()["OK"], 1)
        self.assertEqual(t.summary()["UNVERIFIED"], 1)


class TestViewpoint(unittest.TestCase):
    def test_opposite_of_occluder(self):
        t = DigitalTwin({"S0-0": "sku_A"})
        t.update(obs("sku_A", conf=0.3, vis=0.3, side="left"))
        rec = t.slots["S0-0"]
        self.assertEqual(choose_viewpoint(rec, ["center"]), "right")

    def test_close_when_no_occluder(self):
        t = DigitalTwin({"S0-0": "sku_A"})
        t.update(obs("sku_A", conf=0.3, vis=0.9, side=None))
        self.assertEqual(choose_viewpoint(t.slots["S0-0"], ["center"]), "close")

    def test_never_repeats_and_exhausts(self):
        t = DigitalTwin({"S0-0": "sku_A"})
        t.update(obs(None, conf=0.0, vis=0.2, side="right"))
        rec = t.slots["S0-0"]
        tried = ["center"]
        seen = []
        while True:
            vp = choose_viewpoint(rec, tried)
            if vp is None:
                break
            self.assertNotIn(vp, tried)
            tried.append(vp)
            seen.append(vp)
        self.assertEqual(set(seen), {"left", "right", "close"})


class TestResultsDir(unittest.TestCase):
    """Webots runs a controller from the controller's OWN folder, so a relative
    SHELFSCAN_RESULTS resolved against the working directory lands nowhere the
    caller looks - a whole batch reports FAILED while every run actually
    succeeded. It is resolved against the repo root instead."""

    ROOT = Path("/repo")

    def test_a_relative_path_hangs_off_the_repo_root(self):
        self.assertEqual(results_dir("results_v3/qr", self.ROOT),
                         self.ROOT / "results_v3/qr")

    def test_an_absolute_path_is_left_alone(self):
        p = Path(tempfile.gettempdir()).resolve() / "elsewhere"
        self.assertEqual(results_dir(str(p), self.ROOT), p)

    def test_unset_and_empty_both_give_the_v2_default(self):
        for value in (None, ""):
            self.assertEqual(results_dir(value, self.ROOT),
                             self.ROOT / "results_v2")

    def test_the_capture_output_dir_resolves_the_same_way(self):
        """--out datasets_v4 wrote 364 frames under the controller's folder
        before this was shared, which looked exactly like capturing nothing."""
        self.assertEqual(under_root("datasets_v4", self.ROOT, "dataset"),
                         self.ROOT / "datasets_v4")
        self.assertEqual(under_root(None, self.ROOT, "dataset"),
                         self.ROOT / "dataset")

class TestSlotMatches(unittest.TestCase):
    """The capture subset predicate. The capture controller and the batch
    runner both use it - the runner to decide a seed is worth starting Webots
    for, the controller to decide which slots to drive to - so if they ever
    disagreed a seed would launch and capture nothing."""

    @staticmethod
    def slot(actual=None, blocker=None):
        return {"actual": actual, "blocker_kind": blocker}

    def test_no_subset_keeps_every_slot(self):
        for sl in (self.slot(), self.slot("sku_A"), self.slot(None, "pallet")):
            self.assertTrue(slot_matches(sl, ""))
            self.assertTrue(slot_matches(sl, None))

    def test_empty_blocked_needs_both_empty_and_a_blocker(self):
        self.assertTrue(slot_matches(self.slot(None, "pallet"), "empty_blocked"))
        self.assertFalse(slot_matches(self.slot(None, None), "empty_blocked"))
        self.assertFalse(slot_matches(self.slot("sku_A", "pallet"), "empty_blocked"))
        self.assertFalse(slot_matches(self.slot("sku_A", None), "empty_blocked"))

    def test_an_unknown_subset_is_refused_not_silently_empty(self):
        with self.assertRaises(ValueError):
            slot_matches(self.slot(), "nonsense")


if __name__ == "__main__":
    unittest.main()
