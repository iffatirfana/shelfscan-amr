"""Camera projection and visibility estimation (pure Python, no Webots import).

Webots camera frame (R2022+ FLU convention) is ASSUMED to be:
    +x forward, +y left, +z up   (image right = -y, image up = +z)
Verify this once with webots/tools/overlay_check.py before trusting any labels.
"""
import math
from typing import Optional, Sequence, Tuple

Vec = Sequence[float]
Mat = Sequence[Sequence[float]]   # 3x3 row-major, world_from_camera rotation


def _world_to_cam(cam_pos: Vec, cam_rot: Mat, p: Vec):
    d = [p[i] - cam_pos[i] for i in range(3)]
    # cam_rot is world_from_camera, so camera coords = R^T * d
    return [sum(cam_rot[r][c] * d[r] for r in range(3)) for c in range(3)]


def project_point(cam_pos, cam_rot, p, fov_h, width, height):
    """Return (u, v) pixel coords or None if behind the camera."""
    x, y, z = _world_to_cam(cam_pos, cam_rot, p)
    if x <= 1e-6:
        return None
    f = (width / 2.0) / math.tan(fov_h / 2.0)
    u = width / 2.0 - f * (y / x)
    v = height / 2.0 - f * (z / x)
    return u, v


def project_box(cam_pos, cam_rot, centre, size_xyz, fov_h, width, height
                ) -> Optional[Tuple[float, float, float, float]]:
    """Project an axis-aligned box (world frame) to a clipped pixel bbox
    (x_min, y_min, x_max, y_max), or None if not visible."""
    sx, sy, sz = (s / 2.0 for s in size_xyz)
    us, vs = [], []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                pt = project_point(cam_pos, cam_rot,
                                   (centre[0] + dx, centre[1] + dy, centre[2] + dz),
                                   fov_h, width, height)
                if pt is None:
                    return None
                us.append(pt[0])
                vs.append(pt[1])
    x0, x1 = max(0.0, min(us)), min(float(width), max(us))
    y0, y1 = max(0.0, min(vs)), min(float(height), max(vs))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def to_yolo(bbox, width, height):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
            (x1 - x0) / width, (y1 - y0) / height)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def visibility_and_side(slot_bbox, other_boxes):
    """Estimate how visible a slot is, and on which side a blocker sits.

    other_boxes: detected boxes (x0,y0,x1,y1) that are NOT the slot's own content.
    A box mostly inside the slot region (>= 80% of its own area) is the slot's own
    content. A box that overlaps the slot but extends beyond it is a blocker; its
    side is where its centre lies relative to the slot centre.
    Returns (visibility in [0,1], 'left'|'right'|None).
    """
    a = _area(slot_bbox)
    if a <= 0:
        return 0.0, None
    scx = (slot_bbox[0] + slot_bbox[2]) / 2
    covered, side, best = 0.0, None, 0.0
    for b in other_boxes:
        ov = _inter(slot_bbox, b)
        bcx = (b[0] + b[2]) / 2
        own_content = _area(b) > 0 and ov / _area(b) >= 0.8
        if ov > 0 and not own_content:
            covered += ov
            if ov > best:
                best = ov
                side = "left" if bcx < scx else "right"
    return max(0.0, 1.0 - covered / a), side
