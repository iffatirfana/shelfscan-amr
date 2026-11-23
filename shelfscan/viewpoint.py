"""Active viewpoint selection.

For an UNVERIFIED slot, pick the candidate camera viewpoint most likely to give a
clear view, instead of repeating the same scan.

The rule is deliberately simple and needs no extra model:
  - if the last observation saw an occluder on one side, prefer the viewpoint on
    the opposite side (it can look around the blocker);
  - if the view was merely poor (low confidence, nothing blocking), prefer the
    closer viewpoint;
  - never pick a viewpoint already tried for this slot, unless all were tried;
  - break ties by travel cost from the robot's current position.
"""
from typing import Dict, List, Optional, Sequence

from .types import SlotRecord

VIEWPOINTS = ("center", "left", "right", "close")
OPPOSITE = {"left": "right", "right": "left"}


def preferred_viewpoint(rec: SlotRecord) -> str:
    side = rec.last_occluder_side
    if side in OPPOSITE:
        return OPPOSITE[side]
    return "close"


def choose_viewpoint(rec: SlotRecord, tried: Sequence[str],
                     travel_cost: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Return the next viewpoint to try for this slot, or None if exhausted."""
    remaining: List[str] = [v for v in VIEWPOINTS if v not in tried]
    if not remaining:
        return None
    want = preferred_viewpoint(rec)
    if want in remaining:
        return want
    if travel_cost:
        return min(remaining, key=lambda v: travel_cost.get(v, 0.0))
    return remaining[0]
