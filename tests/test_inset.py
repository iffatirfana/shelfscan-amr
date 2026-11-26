"""Camera-inset overlay tests (HUD_DISPLAY=1).

Pure image work: no Webots. The Display is a fake exposing only the three
calls shelfscan.inset.push makes, which is the whole of its contract.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfscan import inset as I                                  # noqa: E402

PLATE = (100, 100, 160, 150)
CARTON = (80, 80, 240, 260)
PAYLOAD = "SKU=sku_C;INV=INV-013-07"


def frame(w=640, h=480):
    return np.zeros((h, w, 3), np.uint8)


def sku_of(text):
    """Stand-in for the controller's code_sku: the real one is in a module
    that cannot be imported without Webots."""
    for part in (text or "").split(";"):
        k, _, v = part.partition("=")
        if k.strip().upper() == "SKU":
            return v.strip() or None
    return None


def centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class FakeDisplay:
    """Records what the controller would have pushed to Webots."""

    def __init__(self):
        self.pasted = []
        self.live = set()
        self.next_ref = 1

    def imageNew(self, data, fmt, w, h):
        ref, self.next_ref = self.next_ref, self.next_ref + 1
        self.live.add(ref)
        self.last = (bytes(data), fmt, w, h)
        return ref

    def imagePaste(self, ref, x, y, blend):
        assert ref in self.live, "pasted an image that was never created"
        self.pasted.append((ref, x, y, blend))

    def imageDelete(self, ref):
        self.live.discard(ref)


class TestPointIn(unittest.TestCase):
    def test_inside_and_outside(self):
        self.assertTrue(I.point_in((110, 110), PLATE))
        self.assertFalse(I.point_in((300, 110), PLATE))

    def test_the_edge_counts_as_inside(self):
        self.assertTrue(I.point_in((100, 100), PLATE))
        self.assertTrue(I.point_in((160, 150), PLATE))


class TestCodeIn(unittest.TestCase):
    def test_finds_the_code_whose_centre_is_in_the_box(self):
        codes = [(PAYLOAD, centre(PLATE))]
        self.assertEqual(I.code_in(PLATE, codes), PAYLOAD)

    def test_a_code_elsewhere_in_the_frame_is_not_claimed(self):
        self.assertIsNone(I.code_in(PLATE, [(PAYLOAD, (500.0, 400.0))]))

    def test_no_codes_at_all(self):
        self.assertIsNone(I.code_in(PLATE, []))


class TestCaption(unittest.TestCase):
    def test_a_decoded_payload_becomes_the_short_sku_name(self):
        self.assertEqual(I.caption(PAYLOAD, sku_of), "SKU_3")

    def test_no_payload_is_the_failure_text(self):
        self.assertEqual(I.caption(None, sku_of), I.FAIL_TEXT)
        self.assertEqual(I.caption("", sku_of), I.FAIL_TEXT)

    def test_a_payload_that_carries_no_known_sku_also_reads_as_a_failure(self):
        """A code that decodes but means nothing is not a successful read:
        saying SKU_? would claim more than the audit got."""
        self.assertEqual(I.caption("HELLO=world", sku_of), I.FAIL_TEXT)

    def test_without_a_sku_function_the_raw_payload_is_shown(self):
        self.assertEqual(I.caption("301307"), "301307")

    def test_a_long_raw_payload_is_clipped_to_fit(self):
        got = I.caption("X" * 80)
        self.assertEqual(len(got), I.MAX_TEXT)
        self.assertTrue(got.endswith("…"))


class TestOverlayItems(unittest.TestCase):
    def test_a_decoded_plate_is_green_with_its_sku(self):
        boxes = [(PLATE, "label", 0.9)]
        codes = [(PAYLOAD, centre(PLATE))]
        items = I.overlay_items(boxes, codes, sku_of)
        self.assertEqual(items, [(PLATE, I.GREEN, "SKU_3")])

    def test_a_plate_that_did_not_decode_is_amber(self):
        items = I.overlay_items([(PLATE, "label", 0.9)], [], sku_of)
        self.assertEqual(items, [(PLATE, I.AMBER, I.FAIL_TEXT)])

    def test_the_boxes_drawn_are_exactly_the_ones_the_decoder_tried(self):
        """crop_targets decides what gets decoded; the picture must not invent
        a region the readers were never given, nor omit one they were."""
        boxes = [(CARTON, "cardboard_box", 0.9), (PLATE, "label", 0.8)]
        drawn = [b for b, _c, _t in I.overlay_items(boxes, [], sku_of)]
        from shelfscan.decode import crop_targets
        self.assertEqual(drawn, [b for b, _ in crop_targets(boxes)])
        self.assertEqual(drawn, [PLATE])      # carton holds the plate: skipped

    def test_a_carton_with_no_plate_is_drawn_as_the_fallback_region(self):
        items = I.overlay_items([(CARTON, "cardboard_box", 0.9)], [], sku_of)
        self.assertEqual([b for b, _c, _t in items], [CARTON])

    def test_slot_empty_and_blocker_are_never_drawn(self):
        """They are not decode targets, so a box round one would say the
        reader looked there when it did not."""
        boxes = [((0, 0, 50, 50), "slot_empty", 0.9),
                 ((60, 0, 90, 50), "blocker", 0.9)]
        self.assertEqual(I.overlay_items(boxes, [], sku_of), [])

    def test_a_whole_frame_fallback_read_still_gets_a_marker(self):
        """The fallback finds a code the detector never boxed. Dropping it
        would show a blank frame for a slot that was in fact read."""
        items = I.overlay_items([], [(PAYLOAD, (320.0, 240.0))], sku_of)
        self.assertEqual(items, [((320.0, 240.0, 320.0, 240.0), I.GREEN,
                                  "SKU_3")])

    def test_a_code_already_shown_on_its_box_is_not_drawn_twice(self):
        boxes = [(PLATE, "label", 0.9)]
        codes = [(PAYLOAD, centre(PLATE))]
        self.assertEqual(len(I.overlay_items(boxes, codes, sku_of)), 1)

    def test_nothing_detected_and_nothing_read_draws_nothing(self):
        self.assertEqual(I.overlay_items([], [], sku_of), [])


class TestLabelTypeParity(unittest.TestCase):
    """Code128 cartons must be drawn exactly as QR cartons are.

    The two label types reach the overlay by different routes - a QR payload
    spells the SKU out, a Code128 one is six digits because it has a tenth of
    the modules to spend - so the module is pinned as label-type blind: it is
    given boxes and payloads and never asked which kind they are."""

    QR = "SKU=sku_C;INV=INV-013-07"
    BC = "301307"                     # sku_3, seed 013, slot 07

    @staticmethod
    def code_sku(text):
        """The controller's rule, which cannot be imported without Webots."""
        text = (text or "").strip()
        skus = ["sku_A", "sku_B", "sku_C", "sku_D"]
        if text.isdigit() and len(text) == 6:
            i = int(text[0]) - 1
            return skus[i] if 0 <= i < 4 else None
        for part in text.split(";"):
            k, _, v = part.partition("=")
            if k.strip().upper() == "SKU" and v.strip() in skus:
                return v.strip()
        return None

    def test_both_payload_forms_caption_to_the_same_sku(self):
        self.assertEqual(I.caption(self.QR, self.code_sku), "SKU_3")
        self.assertEqual(I.caption(self.BC, self.code_sku), "SKU_3")

    def test_a_decoded_barcode_is_green_exactly_like_a_decoded_qr(self):
        boxes = [(PLATE, "label", 0.9)]
        qr = I.overlay_items(boxes, [(self.QR, centre(PLATE))], self.code_sku)
        bc = I.overlay_items(boxes, [(self.BC, centre(PLATE))], self.code_sku)
        self.assertEqual(qr, bc)
        self.assertEqual(bc, [(PLATE, I.GREEN, "SKU_3")])

    def test_an_unread_barcode_fails_exactly_like_an_unread_qr(self):
        boxes = [(PLATE, "label", 0.9)]
        self.assertEqual(I.overlay_items(boxes, [], self.code_sku),
                         [(PLATE, I.FAIL, I.FAIL_TEXT)])

    def test_the_module_never_branches_on_label_type(self):
        """The one guarantee behind the parity: if this module cannot tell a
        QR frame from a Code128 one, it cannot treat them differently."""
        import inspect
        src = inspect.getsource(I)
        for banned in ("label_type", '"qr"', "'qr'", "barcode", "Code128"):
            self.assertNotIn(banned, src, f"inset.py must not mention {banned}")

    def test_the_barcode_whole_frame_fallback_is_drawn_too(self):
        """Code128 leans on the whole-frame reader far more than QR does, and
        those reads carry the code's own centre, not a detected box."""
        items = I.overlay_items([], [(self.BC, (320.0, 240.0))], self.code_sku)
        self.assertEqual(items, [((320.0, 240.0, 320.0, 240.0), I.GREEN,
                                  "SKU_3")])


class TestScanningState(unittest.TestCase):
    """The in-between picture: a plate is boxed, the readers have not answered.

    It is a different RENDERING of the same frame, never a different decision,
    so the same boxes appear - only the colour and caption change."""

    def test_pending_paints_every_target_cyan_and_says_scanning(self):
        items = I.overlay_items([(PLATE, "label", 0.9)], [], sku_of,
                                pending=True)
        self.assertEqual(items, [(PLATE, I.SCAN, I.SCAN_TEXT)])

    def test_pending_ignores_codes_that_have_already_arrived(self):
        """Painting SCANNING over a frame whose code is already decoded would
        be a lie about what the robot knew at that instant."""
        codes = [(PAYLOAD, centre(PLATE))]
        items = I.overlay_items([(PLATE, "label", 0.9)], codes, sku_of,
                                pending=True)
        self.assertEqual(items, [(PLATE, I.SCAN, I.SCAN_TEXT)])

    def test_pending_draws_the_same_boxes_as_the_settled_frame(self):
        boxes = [(CARTON, "cardboard_box", 0.9), (PLATE, "label", 0.8)]
        a = [b for b, _c, _t in I.overlay_items(boxes, [], sku_of, True)]
        b = [b for b, _c, _t in I.overlay_items(boxes, [], sku_of, False)]
        self.assertEqual(a, b)

    def test_pending_with_nothing_detected_draws_nothing(self):
        self.assertEqual(I.overlay_items([], [], sku_of, pending=True), [])

    def test_the_three_states_are_three_distinct_colours(self):
        self.assertEqual(len({I.SCAN, I.GREEN, I.FAIL}), 3)

    def test_scanning_paints_cyan_pixels(self):
        out = I.draw(frame(), [(PLATE, "label", 0.9)], [], sku_of,
                     pending=True)
        px = out.reshape(-1, 3)
        self.assertTrue((px == I.SCAN).all(axis=1).any())
        self.assertFalse((px == I.GREEN).all(axis=1).any())

    def test_the_settled_frame_never_paints_cyan(self):
        for codes in ([], [(PAYLOAD, centre(PLATE))]):
            out = I.draw(frame(), [(PLATE, "label", 0.9)], codes, sku_of)
            self.assertFalse((out.reshape(-1, 3) == I.SCAN).all(axis=1).any())


class TestContrast(unittest.TestCase):
    """The inset is watched once, at speed, over mid-brown cardboard."""

    def test_the_stroke_is_thick_enough_to_see_in_a_recording(self):
        self.assertGreaterEqual(I.BOX_THICK, 3)

    def test_a_dark_keyline_is_drawn_outside_the_coloured_stroke(self):
        """Cyan on a white plate washes out without it."""
        out = I.draw(frame(), [(PLATE, "label", 0.9)], [], sku_of,
                     pending=True)
        k = I.BOX_THICK // 2 + I.EDGE_THICK
        self.assertTrue((out[PLATE[1] - k, PLATE[0]:PLATE[2]] == I.EDGE)
                        .all(axis=1).any())

    def test_the_caption_sits_on_a_dark_chip(self):
        white = np.full((480, 640, 3), 255, np.uint8)
        out = I.draw(white, [(PLATE, "label", 0.9)], [], sku_of)
        self.assertTrue((out.reshape(-1, 3) == I.EDGE).all(axis=1).any(),
                        "no dark chip drawn behind the caption")

    def test_the_chip_keeps_the_caption_legible_on_a_white_plate(self):
        white = np.full((480, 640, 3), 255, np.uint8)
        out = I.draw(white, [(PLATE, "label", 0.9)], [], sku_of)
        px = out.reshape(-1, 3)
        self.assertTrue((px == I.FAIL).all(axis=1).any())


class TestDraw(unittest.TestCase):
    def test_returns_a_copy_and_leaves_the_detector_frame_alone(self):
        img = frame()
        before = img.copy()
        out = I.draw(img, [(PLATE, "label", 0.9)], [(PAYLOAD, centre(PLATE))],
                     sku_of)
        self.assertTrue(np.array_equal(img, before))
        self.assertFalse(np.array_equal(out, before))
        self.assertEqual(out.shape, img.shape)

    def test_a_decoded_plate_paints_green_pixels_and_no_amber(self):
        out = I.draw(frame(), [(PLATE, "label", 0.9)],
                     [(PAYLOAD, centre(PLATE))], sku_of)
        px = out.reshape(-1, 3)
        self.assertTrue((px == I.GREEN).all(axis=1).any())
        self.assertFalse((px == I.AMBER).all(axis=1).any())

    def test_a_failed_plate_paints_amber_pixels_and_no_green(self):
        out = I.draw(frame(), [(PLATE, "label", 0.9)], [], sku_of)
        px = out.reshape(-1, 3)
        self.assertTrue((px == I.AMBER).all(axis=1).any())
        self.assertFalse((px == I.GREEN).all(axis=1).any())

    def test_the_rectangle_lands_on_the_detected_box(self):
        """Measured on the box's own bottom edge, which nothing else touches.

        The caption cannot be included in this: 'DECODE FAIL' is wider than a
        plate box, so it legitimately runs past the right edge and bounding
        every drawn pixel would not describe the rectangle."""
        out = I.draw(frame(), [(PLATE, "label", 0.9)], [], sku_of)
        # cv2 centres a thickness-2 stroke on the coordinate, so the painted
        # span straddles each edge by up to BOX_THICK.
        row = np.argwhere((out[PLATE[3]] == I.AMBER).all(axis=1))[:, 0]
        self.assertLessEqual(abs(int(row.min()) - PLATE[0]), I.BOX_THICK)
        self.assertLessEqual(abs(int(row.max()) - PLATE[2]), I.BOX_THICK)
        col = np.argwhere((out[:, PLATE[0]] == I.AMBER).all(axis=1))[:, 0]
        self.assertLessEqual(abs(int(col.min()) - PLATE[1]), I.BOX_THICK)
        self.assertLessEqual(abs(int(col.max()) - PLATE[3]), I.BOX_THICK)

    def test_the_caption_sits_above_the_box_when_there_is_room(self):
        out = I.draw(frame(), [(PLATE, "label", 0.9)], [], sku_of)
        rows = np.argwhere((out == I.AMBER).all(axis=2))[:, 0]
        self.assertLess(int(rows.min()), PLATE[1])

    def test_nothing_is_drawn_outside_the_frame(self):
        """A box hard against an edge must not push its caption off the
        picture - the inset is the whole picture, there is no margin."""
        for box in [(0, 0, 60, 40), (580, 0, 639, 40),
                    (0, 440, 60, 479), (580, 440, 639, 479)]:
            out = I.draw(frame(), [(box, "label", 0.9)], [], sku_of)
            self.assertEqual(out.shape, (480, 640, 3))
            self.assertTrue((out != 0).any(), f"nothing drawn for {box}")

    def test_a_box_at_the_very_top_keeps_its_caption_inside(self):
        """With no room above, the caption moves inside the box rather than
        off the top of the frame."""
        top = (100, 0, 200, 60)
        out = I.draw(frame(), [(top, "label", 0.9)], [], sku_of)
        rows = np.argwhere((out == I.AMBER).all(axis=2))[:, 0]
        self.assertGreater(int(rows.max()), top[1])

    def test_a_zero_size_fallback_marker_draws_its_text_without_a_box(self):
        out = I.draw(frame(), [], [(PAYLOAD, (320.0, 240.0))], sku_of)
        self.assertTrue((out.reshape(-1, 3) == I.GREEN).all(axis=1).any())

    def test_many_boxes_all_appear(self):
        boxes = [((10 + 70 * i, 200, 60 + 70 * i, 250), "label", 0.9)
                 for i in range(5)]
        out = I.draw(frame(), boxes, [], sku_of)
        cols = np.argwhere((out == I.AMBER).all(axis=2))[:, 1]
        self.assertGreater(int(cols.max()) - int(cols.min()), 250)


class TestPush(unittest.TestCase):
    def test_pastes_one_bgra_image_at_the_origin(self):
        d = FakeDisplay()
        img = frame(64, 48)
        I.push(d, img, "BGRA")
        self.assertEqual(d.pasted, [(1, 0, 0, False)])
        data, fmt, w, h = d.last
        self.assertEqual((fmt, w, h), ("BGRA", 64, 48))
        self.assertEqual(len(data), 64 * 48 * 4)

    def test_the_image_reference_is_always_released(self):
        """Webots keeps every reference that is not deleted, and this runs
        once a frame for a whole audit."""
        d = FakeDisplay()
        I.push(d, frame(32, 32), "BGRA")
        self.assertEqual(d.live, set())

    def test_the_reference_is_released_even_when_the_paste_fails(self):
        class Boom(FakeDisplay):
            def imagePaste(self, *a):
                raise RuntimeError("display gone")

        d = Boom()
        with self.assertRaises(RuntimeError):
            I.push(d, frame(32, 32), "BGRA")
        self.assertEqual(d.live, set())

    def test_colour_survives_the_bgr_to_bgra_conversion(self):
        img = frame(8, 8)
        img[:, :] = (10, 20, 30)
        d = FakeDisplay()
        I.push(d, img, "BGRA")
        data, _fmt, _w, _h = d.last
        self.assertEqual(tuple(data[:3]), (10, 20, 30))

    def test_show_draws_and_pushes_in_one_call(self):
        d = FakeDisplay()
        I.show(d, "BGRA", frame(), [(PLATE, "label", 0.9)],
               [(PAYLOAD, centre(PLATE))], sku_of)
        self.assertEqual(len(d.pasted), 1)
        self.assertEqual(d.live, set())


class TestReadOnly(unittest.TestCase):
    def test_the_module_decides_nothing(self):
        """It must not import the decoders or the model: everything it shows
        is handed to it. crop_targets is the one thing it borrows, and that is
        a pure ordering function over boxes already detected."""
        import inspect
        src = inspect.getsource(I)
        for banned in ("read_codes", "decode_box", "read_crop", "YOLO",
                       "predict("):
            self.assertNotIn(banned, src, f"inset.py must not call {banned}")


if __name__ == "__main__":
    unittest.main()


class TestThreeWindowPipeline(unittest.TestCase):
    """Tests for the 3-window pipeline: detections, cut crop, and dashboard."""

    def test_draw_detections_draws_boxes_and_banner(self):
        img = frame()
        boxes = [(CARTON, "cardboard_box", 0.91), (PLATE, "label", 0.88)]
        codes = [(PAYLOAD, centre(PLATE))]
        out = I.draw_detections(img, boxes, codes, sku_of, banner="DETECTOR")
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))
        self.assertEqual(int(img.sum()), 0)

    def test_draw_crop_shows_decoded_and_empty_states(self):
        img = np.full((480, 640, 3), 180, np.uint8)
        boxes = [(CARTON, "cardboard_box", 0.91), (PLATE, "label", 0.88)]
        codes = [(PAYLOAD, centre(PLATE))]
        c1 = I.draw_crop(img, boxes, codes, sku_of, title="CUT QR")
        self.assertEqual(c1.shape, (260, 320, 3))
        c2 = I.draw_crop(None, [], (), sku_of, title="CUT CODE128")
        self.assertEqual(c2.shape, (260, 320, 3))

    def test_draw_dashboard_renders_twin_state(self):
        from shelfscan.twin import DigitalTwin
        from shelfscan.types import Observation
        slots_meta = {
            "S0-0-0": {"rack": "A", "level": 0, "bay": 0, "expected": "sku_A"},
            "S0-0-1": {"rack": "A", "level": 0, "bay": 1, "expected": "sku_B"},
        }
        twin = DigitalTwin({k: v["expected"] for k, v in slots_meta.items()})
        st = twin.update(Observation("S0-0-0", "sku_A", 1.0, 1.0, None, "front", 1.0))
        dash = I.draw_dashboard(
            slots_meta, twin.slots, slot_id="S0-0-0",
            decoded_sku="sku_A", status=st, action="Audited")
        self.assertEqual(dash.shape, (340, 420, 3))
        self.assertGreater(int(dash.sum()), 0)


class TestIdleCard(unittest.TestCase):
    """A Display is born black and stays black until something pastes to it.

    A world built with --display but run without HUD_DISPLAY=1 therefore shows
    a permanently black panel that is indistinguishable from a crashed overlay
    - which is exactly how it was read. The idle card makes that state state
    its own reason."""

    def test_it_is_not_black(self):
        """The whole point: a panel that says something, not a dark square."""
        card = I.idle_card(["INSET IDLE", "HUD_DISPLAY is not set"])
        self.assertEqual(card.shape, (480, 640, 3))
        self.assertGreater(int(card.max()), 0)
        # text is drawn light on dark, so some pixels must be well clear of bg
        self.assertGreater(int((card > 150).sum()), 500)

    def test_it_matches_the_camera_frame_size(self):
        """It is pasted to the same 640x480 Display the frames go to, so a
        mismatch would be cropped by Webots."""
        self.assertEqual(I.idle_card(["x"]).shape[:2], (480, 640))

    def test_it_pushes_like_any_other_frame(self):
        d = FakeDisplay()
        I.push(d, I.idle_card(["INSET IDLE", "second line"]), "BGRA")
        self.assertEqual(len(d.pasted), 1)
        data, fmt, w, h = d.last
        self.assertEqual((fmt, w, h), ("BGRA", 640, 480))
        self.assertEqual(len(data), 640 * 480 * 4)
        self.assertEqual(d.live, set(), "image refs must not leak")

    def test_a_long_message_does_not_crash_it(self):
        card = I.idle_card(["INSET IDLE", "x" * 200])
        self.assertEqual(card.shape, (480, 640, 3))
