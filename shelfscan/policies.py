"""Audit policies. Each takes an `observe(slot_id, viewpoint) -> Observation`
callable, so the SAME code runs in the abstract harness and in Webots (where
`observe` drives the robot to the viewpoint and runs the detector).

V1: one fixed patrol, center viewpoint.
V3: fixed patrol, then re-observe low-confidence slots from the SAME viewpoint.
V4: fixed patrol, then for each unresolved slot choose a better viewpoint
    (active verification), up to `max_tries` extra looks.
V5: same loop as V4, but the next viewpoint comes from a small learned model
    (MVP). Falls back to the V4 rule whenever the model declines or no model
    was supplied, so V5 can never do worse than V4 for lack of a prediction.
"""
from typing import Callable, Dict, List

from .twin import DigitalTwin
from .types import Observation, Status
from .viewpoint import VIEWPOINTS, choose_viewpoint

Observe = Callable[[str, str], Observation]


def _patrol(twin: DigitalTwin, observe: Observe, order: List[str]) -> Dict[str, List[str]]:
    tried: Dict[str, List[str]] = {sid: [] for sid in order}
    for sid in order:
        twin.update(observe(sid, "center"))
        tried[sid].append("center")
    return tried


def v1_fixed_patrol(twin: DigitalTwin, observe: Observe, order: List[str]):
    _patrol(twin, observe, order)


def v3_rescan_same_view(twin: DigitalTwin, observe: Observe, order: List[str],
                        max_tries: int = 1):
    _patrol(twin, observe, order)
    scope = set(order)
    for _ in range(max_tries):
        for rec in twin.unresolved():
            if rec.slot_id in scope:
                twin.update(observe(rec.slot_id, "center"))


def v4_active_verification(twin: DigitalTwin, observe: Observe, order: List[str],
                           max_tries: int = 3, on_decision=None):
    """`on_decision(rec, tried, candidates, chosen)` is called at every
    next-best-view decision, for logging only. It cannot influence the choice:
    it is invoked after `choose_viewpoint` has already returned."""
    tried = _patrol(twin, observe, order)
    scope = set(order)
    for _ in range(max_tries):
        # Only slots this call was asked to audit. The twin may hold slots the
        # caller is handling separately (one shelf side at a time), and those
        # start out UNVERIFIED simply because they have not been visited yet.
        pending = [r for r in twin.unresolved() if r.slot_id in scope]
        if not pending:
            break
        for rec in pending:
            vp = choose_viewpoint(rec, tried[rec.slot_id])
            if on_decision is not None:
                cands = [v for v in VIEWPOINTS if v not in tried[rec.slot_id]]
                on_decision(rec, list(tried[rec.slot_id]), cands, vp)
            if vp is None:
                continue
            twin.update(observe(rec.slot_id, vp))
            tried[rec.slot_id].append(vp)


def v5_learned(twin: DigitalTwin, observe: Observe, order: List[str],
               max_tries: int = 3, picker=None, on_decision=None):
    """V4's loop with a learned viewpoint picker.

    `picker(rec, tried, candidates) -> viewpoint or None` is supplied by the
    caller, because scoring a candidate needs the robot's current pose and the
    slot map, which this module deliberately knows nothing about. Whenever the
    picker returns None - no model, no candidates, or the model failed - the
    V4 rule decides instead, so the fallback is total."""
    tried = _patrol(twin, observe, order)
    scope = set(order)
    for _ in range(max_tries):
        pending = [r for r in twin.unresolved() if r.slot_id in scope]
        if not pending:
            break
        for rec in pending:
            cands = [v for v in VIEWPOINTS if v not in tried[rec.slot_id]]
            vp = None
            if picker is not None and cands:
                vp = picker(rec, list(tried[rec.slot_id]), cands)
            if vp not in cands:                      # None, or not offered
                vp = choose_viewpoint(rec, tried[rec.slot_id])
            if on_decision is not None:
                on_decision(rec, list(tried[rec.slot_id]), cands, vp)
            if vp is None:
                continue
            twin.update(observe(rec.slot_id, vp))
            tried[rec.slot_id].append(vp)


POLICIES = {
    "V1_fixed": v1_fixed_patrol,
    "V3_rescan": v3_rescan_same_view,
    "V4_active": v4_active_verification,
    "V5_learned": v5_learned,
}
