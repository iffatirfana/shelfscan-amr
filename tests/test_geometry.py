import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfscan.geometry import (project_box, project_point, to_yolo,   # noqa: E402
                                visibility_and_side)

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
W, H, FOV = 640, 480, math.radians(90)


class TestProjection(unittest.TestCase):
    def test_point_ahead_is_image_centre(self):
        u, v = project_point((0, 0, 0), I3, (2, 0, 0), FOV, W, H)
        self.assertAlmostEqual(u, W / 2)
        self.assertAlmostEqual(v, H / 2)

    def test_left_object_is_left_in_image(self):
        u, _ = project_point((0, 0, 0), I3, (2, 0.5, 0), FOV, W, H)   # +y = left
        self.assertLess(u, W / 2)

    def test_up_object_is_up_in_image(self):
        _, v = project_point((0, 0, 0), I3, (2, 0, 0.5), FOV, W, H)   # +z = up
        self.assertLess(v, H / 2)

    def test_behind_camera_none(self):
        self.assertIsNone(project_point((0, 0, 0), I3, (-1, 0, 0), FOV, W, H))
        self.assertIsNone(project_box((0, 0, 0), I3, (-1, 0, 0), (0.2, 0.2, 0.2), FOV, W, H))

    def test_box_projects_and_yolo_normalised(self):
        bb = project_box((0, 0, 0), I3, (2, 0, 0), (0.4, 0.4, 0.4), FOV, W, H)
        self.assertIsNotNone(bb)
        cx, cy, w, h = to_yolo(bb, W, H)
        self.assertAlmostEqual(cx, 0.5, places=2)
        self.assertAlmostEqual(cy, 0.5, places=2)
        self.assertTrue(0 < w < 1 and 0 < h < 1)

    def test_nearer_box_is_bigger(self):
        near = project_box((0, 0, 0), I3, (1.5, 0, 0), (0.4, 0.4, 0.4), FOV, W, H)
        far = project_box((0, 0, 0), I3, (3.0, 0, 0), (0.4, 0.4, 0.4), FOV, W, H)
        self.assertGreater(to_yolo(near, W, H)[2], to_yolo(far, W, H)[2])


class TestVisibility(unittest.TestCase):
    slot = (100, 100, 200, 200)

    def test_no_blockers_fully_visible(self):
        self.assertEqual(visibility_and_side(self.slot, []), (1.0, None))

    def test_own_content_not_a_blocker(self):
        vis, side = visibility_and_side(self.slot, [(110, 110, 190, 190)])
        self.assertEqual((vis, side), (1.0, None))

    def test_left_blocker(self):
        vis, side = visibility_and_side(self.slot, [(60, 90, 150, 210)])
        self.assertEqual(side, "left")
        self.assertLess(vis, 0.7)

    def test_right_blocker(self):
        _, side = visibility_and_side(self.slot, [(150, 90, 240, 210)])
        self.assertEqual(side, "right")


if __name__ == "__main__":
    unittest.main()
