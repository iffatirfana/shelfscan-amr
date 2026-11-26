"""Crop-guided decoding tests.

The round-trip tests render a REAL QR / Code128 with the same generators the
world uses, paste it onto a synthetic cardboard frame at the sort of size the
camera sees, and read it back through the production path. No Webots.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webots/tools"))

from shelfscan import decode as D                                # noqa: E402

CARDBOARD = (140, 175, 200)        # BGR, roughly the carton brown


def frame(w=640, h=480):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = CARDBOARD
    return img


def paste(img, patch, x0, y0):
    """Paste a greyscale patch as BGR and return its box."""
    ph, pw = patch.shape[:2]
    img[y0:y0 + ph, x0:x0 + pw] = np.dstack([patch] * 3)
    return (x0, y0, x0 + pw, y0 + ph)


def qr_patch(text, px):
    from make_world import _qr_image
    return np.array(_qr_image(text, px))


def barcode_patch(text, w, h):
    from make_world import _barcode_image
    return np.array(_barcode_image(text, w, h))


class TestPadBox(unittest.TestCase):
    def test_grows_by_a_fraction_of_the_box(self):
        self.assertEqual(D.pad_box((100, 100, 200, 200), 640, 480, 0.10),
                         (90, 90, 210, 210))

    def test_clips_to_the_frame(self):
        self.assertEqual(D.pad_box((2, 2, 50, 50), 640, 480, 0.5), (0, 0, 74, 74))
        self.assertEqual(D.pad_box((600, 440, 639, 479), 640, 480, 0.5)[2:],
                         (640, 480))

    def test_zero_padding_is_the_box_itself(self):
        self.assertEqual(D.pad_box((10, 20, 30, 40), 640, 480, 0.0),
                         (10, 20, 30, 40))


class TestPrepCrop(unittest.TestCase):
    def test_upscales_and_returns_greyscale(self):
        img = frame()
        crop = D.prep_crop(img, (100, 100, 200, 200), pad_frac=0.0, scale=3)
        self.assertEqual(crop.ndim, 2)
        self.assertEqual(crop.shape, (300, 300))

    def test_rejects_a_crop_too_small_to_hold_a_code(self):
        self.assertIsNone(D.prep_crop(frame(), (10, 10, 14, 14)))

    def test_caps_the_upscale_so_a_big_box_is_not_blown_up(self):
        crop = D.prep_crop(frame(), (0, 0, 640, 480), pad_frac=0.0, scale=3)
        self.assertLessEqual(max(crop.shape), D.MAX_CROP_PX)

    def test_sharpen_changes_pixels_but_keeps_the_shape(self):
        img = frame()
        a = D.prep_crop(img, (100, 100, 200, 200), 0.0, 2, do_sharpen=False)
        b = D.prep_crop(img, (100, 100, 200, 200), 0.0, 2, do_sharpen=True)
        self.assertEqual(a.shape, b.shape)


class TestCropTargets(unittest.TestCase):
    def test_label_boxes_come_first_most_confident_first(self):
        boxes = [((0, 0, 50, 50), "cardboard_box", 0.9),
                 ((10, 10, 30, 30), "label", 0.7),
                 ((200, 200, 240, 240), "label", 0.95)]
        got = D.crop_targets(boxes)
        self.assertEqual([c for _, c in got][:2], ["label", "label"])
        self.assertEqual(got[0][0], (200, 200, 240, 240))

    def test_carton_holding_a_label_box_is_not_decoded_twice(self):
        boxes = [((0, 0, 50, 50), "cardboard_box", 0.9),
                 ((10, 10, 30, 30), "label", 0.8)]
        self.assertEqual(D.crop_targets(boxes), [((10, 10, 30, 30), "label")])

    def test_carton_without_a_label_box_is_the_fallback(self):
        boxes = [((0, 0, 50, 50), "cardboard_box", 0.9)]
        self.assertEqual(D.crop_targets(boxes),
                         [((0, 0, 50, 50), "cardboard_box")])

    def test_other_classes_are_never_decoded(self):
        boxes = [((0, 0, 50, 50), "slot_empty", 0.9),
                 ((60, 0, 90, 50), "blocker", 0.9)]
        self.assertEqual(D.crop_targets(boxes), [])

    def test_no_boxes_gives_no_targets(self):
        self.assertEqual(D.crop_targets([]), [])


class TestQrRoundTrip(unittest.TestCase):
    PAYLOAD = "SKU=sku_C;INV=INV-013-07"

    def test_reads_a_pasted_qr_through_the_production_path(self):
        img = frame()
        box = paste(img, qr_patch(self.PAYLOAD, 110), 240, 180)
        self.assertEqual(D.decode_box(img, box, "qr"), self.PAYLOAD)

    def test_reads_a_small_qr_the_full_frame_reader_would_struggle_with(self):
        img = frame()
        box = paste(img, qr_patch(self.PAYLOAD, 70), 300, 220)
        self.assertEqual(D.decode_box(img, box, "qr"), self.PAYLOAD)

    def test_read_codes_returns_the_payload_and_the_box_centre(self):
        img = frame()
        box = paste(img, qr_patch(self.PAYLOAD, 110), 240, 180)
        boxes = [(box, "label", 0.95)]
        got = D.read_codes(img, boxes, "qr")
        self.assertEqual(len(got), 1)
        text, (cx, cy) = got[0]
        self.assertEqual(text, self.PAYLOAD)
        self.assertAlmostEqual(cx, (box[0] + box[2]) / 2)
        self.assertAlmostEqual(cy, (box[1] + box[3]) / 2)

    def test_blank_cardboard_decodes_to_nothing(self):
        self.assertIsNone(D.decode_box(frame(), (200, 200, 320, 320), "qr"))

    def test_two_codes_in_one_frame_both_come_back(self):
        img = frame()
        a = paste(img, qr_patch("SKU=sku_A;INV=INV-013-01", 100), 60, 180)
        b = paste(img, qr_patch("SKU=sku_D;INV=INV-013-02", 100), 400, 180)
        got = D.read_codes(img, [(a, "label", 0.9), (b, "label", 0.9)], "qr")
        self.assertEqual({t for t, _ in got},
                         {"SKU=sku_A;INV=INV-013-01", "SKU=sku_D;INV=INV-013-02"})

    def test_carton_fallback_reads_a_plate_the_detector_missed(self):
        """Only a cardboard_box box: the crop is the whole carton face."""
        img = frame()
        carton = (200, 150, 420, 370)
        paste(img, qr_patch(self.PAYLOAD, 150), 235, 185)
        got = D.read_codes(img, [(carton, "cardboard_box", 0.9)], "qr")
        self.assertEqual([t for t, _ in got], [self.PAYLOAD])


class TestCode128RoundTrip(unittest.TestCase):
    PAYLOAD = "301307"

    def setUp(self):
        try:
            import zxingcpp                                      # noqa: F401
        except ImportError:
            self.skipTest("zxing-cpp not installed")

    def test_reads_a_pasted_code128(self):
        img = frame()
        box = paste(img, barcode_patch(self.PAYLOAD, 300, 110), 170, 200)
        self.assertEqual(D.decode_box(img, box, "barcode"), self.PAYLOAD)

    def test_blank_cardboard_decodes_to_nothing(self):
        self.assertIsNone(D.decode_box(frame(), (200, 200, 400, 290), "barcode"))


class TestSkuVocabulary(unittest.TestCase):
    """code_sku must read SKU names, not whatever the detector's classes are.

    The controller cannot be imported without Webots, so the rule it depends on
    is pinned here: the v3 class set shares no name with the SKU vocabulary, so
    indexing the wrong list is never a silent near-miss."""

    def test_v3_classes_hold_no_sku_name(self):
        from shelfscan.robot_utils import class_set
        self.assertEqual(set(class_set("v3")) & set(class_set("v2")[:4]), set())

    def test_sku_vocabulary_is_the_v2_head(self):
        from shelfscan.robot_utils import class_set
        self.assertEqual(class_set("v2")[:4],
                         ["sku_A", "sku_B", "sku_C", "sku_D"])


class TestScaleCascade(unittest.TestCase):
    def test_scaled_never_exceeds_the_cap(self):
        img = np.zeros((900, 900), np.uint8)
        self.assertLessEqual(max(D.scaled(img, 3).shape), D.MAX_CROP_PX)

    def test_scaled_leaves_a_unit_scale_alone(self):
        img = np.zeros((40, 40), np.uint8)
        self.assertIs(D.scaled(img, 1), img)

    def test_cascade_is_three_scales(self):
        self.assertEqual(D.SCALES, (1, 2, 3))

    def test_read_crop_stops_at_the_first_scale_that_reads(self):
        tried = []

        def fake(crop):
            tried.append(crop.shape[0])
            return "PAYLOAD" if len(tried) == 2 else None

        crop = np.zeros((50, 50), np.uint8)
        import unittest.mock as mock
        with mock.patch.object(D, "read_qr_crop", fake):
            self.assertEqual(D.read_crop(crop, "qr"), "PAYLOAD")
        self.assertEqual(len(tried), 2)        # stopped, did not try scale 3

    def test_read_crop_tries_every_scale_before_giving_up(self):
        tried = []
        import unittest.mock as mock
        with mock.patch.object(D, "read_qr_crop",
                               lambda c: tried.append(c.shape[0]) or None):
            self.assertIsNone(D.read_crop(np.zeros((50, 50), np.uint8), "qr"))
        self.assertEqual(len(tried), len(D.SCALES))

    def test_read_crop_of_none_is_none(self):
        self.assertIsNone(D.read_crop(None, "qr"))


class TestWholeFrameFallback(unittest.TestCase):
    PAYLOAD = "SKU=sku_B;INV=INV-013-04"

    def test_fallback_runs_when_no_crop_reads(self):
        called = []

        def fallback(img):
            called.append(True)
            return [(self.PAYLOAD, (320.0, 240.0))]

        # a detected carton with nothing readable on it
        got = D.read_codes(frame(), [((100, 100, 300, 300), "cardboard_box", 0.9)],
                           "qr", fallback=fallback)
        self.assertEqual(called, [True])
        self.assertEqual(got, [(self.PAYLOAD, (320.0, 240.0))])

    def test_fallback_runs_when_there_are_no_boxes_at_all(self):
        called = []
        D.read_codes(frame(), [], "qr",
                     fallback=lambda im: called.append(True) or [])
        self.assertEqual(called, [True])

    def test_fallback_is_skipped_once_a_crop_reads(self):
        called = []
        img = frame()
        box = paste(img, qr_patch(self.PAYLOAD, 110), 240, 180)
        got = D.read_codes(img, [(box, "label", 0.95)], "qr",
                           fallback=lambda im: called.append(True) or [])
        self.assertEqual(called, [])
        self.assertEqual([t for t, _ in got], [self.PAYLOAD])

    def test_no_fallback_supplied_is_still_fine(self):
        self.assertEqual(D.read_codes(frame(), [], "qr"), [])


class TestAnnotate(unittest.TestCase):
    def test_returns_a_copy_and_leaves_the_frame_alone(self):
        img = frame()
        before = img.copy()
        out = D.annotate(img, [((10, 10, 60, 60), "label", 0.9)],
                         [("SKU=sku_A;INV=1", (35, 35))])
        self.assertTrue(np.array_equal(img, before))
        self.assertFalse(np.array_equal(out, before))


if __name__ == "__main__":
    unittest.main()
