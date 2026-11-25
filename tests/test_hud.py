"""HUD text-building tests. No Webots: the renderer takes any object with
setLabel, so a recording stub stands in for the robot."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfscan import hud as H                                   # noqa: E402
from shelfscan.types import Status                               # noqa: E402


def slot(rack="A", level=1, bay=3, expected="sku_A"):
    return dict(rack=rack, level=level, bay=bay, expected=expected)


class FakeRobot:
    """Records setLabel calls the way Webots would receive them."""

    def __init__(self):
        self.labels = {}
        self.calls = []

    def setLabel(self, lid, text, x, y, size, colour, transparency, font):
        self.labels[lid] = text
        self.calls.append((lid, text, x, y, size, colour, transparency, font))


class TestPieces(unittest.TestCase):
    def test_sku_label_numbers_the_four_skus(self):
        self.assertEqual(H.sku_label("sku_A"), "SKU_1")
        self.assertEqual(H.sku_label("sku_C"), "SKU_3")
        self.assertEqual(H.sku_label("sku_D"), "SKU_4")

    def test_sku_label_passes_other_names_through(self):
        self.assertEqual(H.sku_label("slot_empty"), "EMPTY")
        self.assertEqual(H.sku_label("cardboard_box"), "cardboard_box")
        self.assertEqual(H.sku_label(None), "-")

    def test_slot_line(self):
        line = H.slot_line(slot(rack="A", level=1, bay=3), 7, 30)
        self.assertIn("Side A | L1 | Bay 3", line)
        self.assertIn("slot 7/30", line)

    def test_detector_line_counts_and_best_confidence(self):
        boxes = [((0, 0, 1, 1), "cardboard_box", 0.91),
                 ((0, 0, 1, 1), "cardboard_box", 0.55),
                 ((0, 0, 1, 1), "label", 0.88),
                 ((0, 0, 1, 1), "blocker", 0.74)]
        line = H.detector_line(boxes)
        self.assertIn("cardboard_box x2 0.91", line)
        self.assertIn("label x1 0.88", line)
        self.assertIn("blocker x1 0.74", line)

    def test_detector_line_uses_a_fixed_class_order(self):
        boxes = [((0, 0, 1, 1), "blocker", 0.5),
                 ((0, 0, 1, 1), "label", 0.5),
                 ((0, 0, 1, 1), "cardboard_box", 0.5),
                 ((0, 0, 1, 1), "slot_empty", 0.5)]
        line = H.detector_line(boxes)
        order = [line.index(c) for c in
                 ("cardboard_box", "label", "slot_empty", "blocker")]
        self.assertEqual(order, sorted(order))

    def test_detector_line_with_no_boxes(self):
        self.assertIn("no boxes", H.detector_line([]))

    def test_decode_line(self):
        self.assertEqual(H.decode_line("qr", "SKU=sku_C;INV=1", "sku_C"),
                         "QR -> SKU_3")
        self.assertEqual(H.decode_line("barcode", "2101013", "sku_B"),
                         "Code128 -> SKU_2")

    def test_decode_line_failure(self):
        self.assertIn("DECODE FAIL", H.decode_line("qr", None, None))
        self.assertIn("DECODE FAIL", H.decode_line("barcode"))

    def test_decode_line_without_label_mode_is_not_a_failure(self):
        line = H.decode_line(None)
        self.assertNotIn("FAIL", line)
        self.assertIn("printed pattern", line)

    def test_twin_line(self):
        self.assertEqual(H.twin_line("sku_A", "sku_C"),
                         "Twin  expected SKU_1  |  read SKU_3")

    def test_twin_line_expected_empty(self):
        self.assertIn("(empty)", H.twin_line(None, None))

    def test_status_entry_colours(self):
        self.assertEqual(H.status_entry(Status.OK),
                         ("OK", H.STATUS_COLOUR[Status.OK]))
        for st in Status:
            text, colour = H.status_entry(st)
            self.assertEqual(text, st.value)
            self.assertIsInstance(colour, int)

    def test_every_status_has_its_own_colour(self):
        self.assertEqual(len(set(H.STATUS_COLOUR.values())), len(Status))

    def test_tally_line(self):
        line = H.tally_line({"OK": 12, "MISPLACED": 2, "EMPTY": 1,
                             "UNVERIFIED": 3})
        for part in ("OK 12", "MISPLACED 2", "EMPTY 1", "UNVERIFIED 3"):
            self.assertIn(part, line)

    def test_tally_line_defaults_missing_counts_to_zero(self):
        self.assertIn("EMPTY 0", H.tally_line({"OK": 1}))


class TestNextAction(unittest.TestCase):
    def test_first_slot(self):
        self.assertEqual(H.next_action(None, slot(bay=0)), "Moving to Bay 0")

    def test_same_slot_is_a_re_look(self):
        s = slot()
        self.assertEqual(H.next_action(s, dict(s)),
                         "Re-look from new viewpoint")

    def test_bay_change(self):
        self.assertEqual(H.next_action(slot(bay=3), slot(bay=4)),
                         "Moving to Bay 4")

    def test_level_change(self):
        self.assertEqual(H.next_action(slot(level=1), slot(level=0)),
                         "Lifting to L0")

    def test_side_change_wins_over_level_and_bay(self):
        self.assertEqual(
            H.next_action(slot(rack="A", level=2, bay=4),
                          slot(rack="B", level=0, bay=0)),
            "Switching to Side B")

    def test_level_change_wins_over_bay(self):
        self.assertEqual(
            H.next_action(slot(level=2, bay=4), slot(level=1, bay=0)),
            "Lifting to L1")

    def test_end_of_scan(self):
        self.assertEqual(H.next_action(slot(), None), "Scan complete")


class TestHudRender(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.steps = []
        self.hud = H.Hud(self.robot, total=30, label_type="qr", hold_steps=3,
                         step_fn=lambda: self.steps.append(1))

    def test_begin_draws_slot_and_next_action(self):
        self.hud.begin("S0-1-3", slot())
        self.assertIn("Side A | L1 | Bay 3", self.robot.labels[H.SLOT])
        self.assertIn("Moving to Bay 3", self.robot.labels[H.NEXT])

    def test_scan_index_counts_distinct_slots_and_is_stable_on_re_look(self):
        self.hud.begin("S0-1-0", slot(bay=0))
        self.hud.begin("S0-1-1", slot(bay=1))
        self.assertIn("slot 2/30", self.robot.labels[H.SLOT])
        self.hud.begin("S0-1-0", slot(bay=0))       # a V4 re-look
        self.assertIn("slot 1/30", self.robot.labels[H.SLOT])

    def test_begin_blanks_the_previous_slot_readings(self):
        """The slot line names the slot being driven to; the readings below it
        still belong to the one just left until the robot gets there."""
        self.hud.begin("S0-2-4", slot(level=2, bay=4))
        self.hud.detection([((0, 0, 1, 1), "label", 0.9)], "SKU=sku_C", "sku_C")
        self.hud.status(Status.OK, {"OK": 1})
        self.hud.begin("S0-1-4", slot(level=1, bay=4))
        for lid in (H.DET, H.DECODE, H.TWIN, H.STATE):
            self.assertEqual(self.robot.labels[lid], "")
        self.assertIn("L1", self.robot.labels[H.SLOT])

    def test_readings_come_back_on_arrival(self):
        self.hud.begin("S0-1-4", slot(level=1, bay=4))
        self.hud.detection([((0, 0, 1, 1), "blocker", 0.9)], None, None)
        self.assertIn("blocker", self.robot.labels[H.DET])
        self.assertIn("DECODE FAIL", self.robot.labels[H.DECODE])

    def test_detection_draws_detector_decode_and_twin(self):
        self.hud.begin("S0-1-3", slot(expected="sku_A"))
        self.hud.detection([((0, 0, 1, 1), "label", 0.88)],
                           "SKU=sku_C;INV=1", "sku_C")
        self.assertIn("label x1 0.88", self.robot.labels[H.DET])
        self.assertEqual(self.robot.labels[H.DECODE], "QR -> SKU_3")
        self.assertIn("expected SKU_1", self.robot.labels[H.TWIN])
        self.assertIn("read SKU_3", self.robot.labels[H.TWIN])

    def test_status_clears_the_stale_action(self):
        """At a decision the previous move is over and the next is not chosen
        yet, so the line must not still claim the move that brought us here."""
        self.hud.begin("S0-1-3", slot(level=1, bay=4))
        self.assertIn("Bay 4", self.robot.labels[H.NEXT])
        self.hud.status(Status.OK, {"OK": 1})
        self.assertNotIn("Bay 4", self.robot.labels[H.NEXT])
        self.assertIn(H.DECIDING, self.robot.labels[H.NEXT])

    def test_the_next_move_replaces_deciding(self):
        self.hud.begin("S0-1-3", slot(level=1, bay=3))
        self.hud.status(Status.OK, {"OK": 1})
        self.hud.begin("S0-1-4", slot(level=1, bay=4))
        self.assertIn("Moving to Bay 4", self.robot.labels[H.NEXT])

    def test_panel_clears_the_camera_overlay(self):
        """Webots draws the camera device over the TOP-LEFT of the view, at
        the camera's native size. Every line must miss that rectangle."""
        ox, oy = H.CAM_OVERLAY
        for lid, (x, y, _, _) in H.LAYOUT.items():
            self.assertTrue(x > ox or y > oy,
                            f"line {lid} at ({x}, {y}) is under the camera "
                            f"overlay ({ox}, {oy})")

    def test_status_sets_the_state_colour_and_tally(self):
        self.hud.begin("S0-1-3", slot())
        self.hud.status(Status.MISPLACED, {"OK": 1, "MISPLACED": 1})
        # the LAST write: begin() blanks the line before status() fills it
        colour = [c[5] for c in self.robot.calls if c[0] == H.STATE][-1]
        self.assertEqual(colour, H.STATUS_COLOUR[Status.MISPLACED])
        self.assertIn("MISPLACED 1", self.robot.labels[H.TALLY])

    def test_status_holds_the_simulation(self):
        self.hud.begin("S0-1-3", slot())
        self.assertEqual(self.steps, [])          # drawing alone costs nothing
        self.hud.status(Status.OK, {"OK": 1})
        self.assertEqual(len(self.steps), 3)

    def test_finish_announces_scan_complete(self):
        self.hud.begin("S0-1-3", slot())
        self.hud.finish({"OK": 30})
        self.assertIn("Scan complete", self.robot.labels[H.NEXT])

    def test_lines_do_not_overlap_and_stay_inside_the_view(self):
        ys = sorted(y for _, y, _, _ in H.LAYOUT.values())
        for a, b in zip(ys, ys[1:]):
            self.assertGreater(b - a, 0.02)
        for x, y, size, _ in H.LAYOUT.values():
            self.assertTrue(0.0 <= x < 1.0 and 0.0 <= y < 1.0)
            self.assertTrue(0.0 < size < 1.0)

    def test_every_line_has_its_own_label_id(self):
        ids = (H.SLOT, H.DET, H.DECODE, H.TWIN, H.STATE, H.NEXT, H.TALLY)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(H.LAYOUT), set(ids))

    def test_shot_fn_fires_after_the_hold_at_each_decided_slot(self):
        shots = []
        h = H.Hud(self.robot, total=30, hold_steps=2,
                  step_fn=lambda: self.steps.append(1),
                  shot_fn=shots.append)
        h.begin("S0-1-3", slot())
        # one at the move, showing the action line, and one at the decision
        self.assertEqual(shots, ["S0-1-3_move"])
        h.status(Status.MISPLACED, {"MISPLACED": 1})
        self.assertEqual(shots, ["S0-1-3_move", "S0-1-3_MISPLACED"])
        self.assertEqual(len(self.steps), 2)      # the hold ran first
        h.finish({"MISPLACED": 1})
        self.assertEqual(shots[-1], "complete")

    def test_no_shot_fn_is_fine(self):
        self.hud.begin("S0-1-3", slot())
        self.hud.status(Status.OK, {"OK": 1})     # must not raise

    def test_hud_without_a_step_fn_never_blocks(self):
        h = H.Hud(FakeRobot(), total=1, hold_steps=5)
        h.begin("S0-0-0", slot())
        h.status(Status.OK, {"OK": 1})            # must not raise


class FakeTwin:
    """Enough of DigitalTwin for the attach() wrapper."""

    def __init__(self, status=Status.OK):
        self.status = status
        self.calls = []

    def update(self, obs, evidence_path=None):
        self.calls.append((obs.slot_id, evidence_path))
        return self.status

    def summary(self):
        return {"OK": len(self.calls)}


class Obs:
    def __init__(self, slot_id):
        self.slot_id = slot_id


class TestAttach(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.steps = []
        self.hud = H.Hud(self.robot, total=30, hold_steps=2,
                         step_fn=lambda: self.steps.append(1))
        self.twin = FakeTwin()
        H.attach(self.twin, self.hud)

    def test_wrapper_forwards_the_call_and_its_result(self):
        out = self.twin.update(Obs("S0-1-3"), "evidence.jpg")
        self.assertEqual(out, Status.OK)
        self.assertEqual(self.twin.calls, [("S0-1-3", "evidence.jpg")])

    def test_wrapper_redraws_for_the_observed_slot(self):
        self.hud.begin("S0-1-3", slot())
        self.twin.update(Obs("S0-1-3"))
        self.assertEqual(self.robot.labels[H.STATE], "OK")
        self.assertEqual(len(self.steps), 2)

    def test_wrapper_ignores_updates_for_other_slots(self):
        self.hud.begin("S0-1-3", slot())
        self.twin.update(Obs("S0-1-4"))           # a code read for a neighbour
        # begin() blanked the line; the neighbour's status must not fill it
        self.assertEqual(self.robot.labels[H.STATE], "")
        self.assertEqual(self.steps, [])          # and spends no hold

    def test_wrapper_is_inert_before_the_first_slot(self):
        self.twin.update(Obs("S0-1-3"))
        self.assertNotIn(H.STATE, self.robot.labels)
        self.assertEqual(self.steps, [])


if __name__ == "__main__":
    unittest.main()
