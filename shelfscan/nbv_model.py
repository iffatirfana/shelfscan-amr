"""Features and scorer for the learned next-best-view picker (V5, MVP).

Shared by training (experiments/train_nbv.py) and inference (the audit
controller), so the two can never drift apart: one row = one CANDIDATE
viewpoint at one decision point, and `features()` is the only place that turns
a row into a vector.

Deliberately NOT given a ready-made "is this the opposite of the blocker side"
feature. That is exactly the V4 rule, and handing it over would make the tree a
one-split re-encoding of the thing it is supposed to be compared against. It
gets the raw ingredients - which side the occluder was on, which viewpoint this
candidate is - and has to find the interaction itself.

No Webots and no sklearn import at module level, so the unit tests can use it.
"""
from typing import Dict, List, Optional, Sequence

CANDIDATES = ("center", "left", "right", "close")
OCCLUDER_SIDES = ("none", "left", "right", "front")

FEATURE_NAMES = (
    ["uncertainty", "visibility", "travel_m", "blocker_present",
     "level", "level_change", "n_tried", "try_index", "side_is_B"]
    + [f"cand_{c}" for c in CANDIDATES]
    + [f"occ_{s}" for s in OCCLUDER_SIDES]
)


def features(row: Dict) -> List[float]:
    """One feature vector for one candidate viewpoint."""
    occ = str(row.get("occluder_side") or "none")
    if occ not in OCCLUDER_SIDES:
        occ = "none"
    out = [
        float(row["uncertainty"]),
        float(row["visibility"]),
        float(row["travel_m"]),
        float(row["blocker_present"]),
        float(row["level"]),
        float(row["level_change"]),
        float(row["n_tried"]),
        float(row["try_index"]),
        1.0 if str(row["side"]) == "B" else 0.0,
    ]
    out += [1.0 if str(row["candidate"]) == c else 0.0 for c in CANDIDATES]
    out += [1.0 if occ == s else 0.0 for s in OCCLUDER_SIDES]
    return out


class NBVPicker:
    """Scores the candidates of one decision and returns the best.

    Returns None when it cannot decide, so the caller falls back to the V4
    rule rather than inventing a choice."""

    def __init__(self, path):
        import joblib
        self.model = joblib.load(path)
        self.n_calls = 0
        self.n_fallback = 0

    def pick(self, rows: Sequence[Dict]) -> Optional[str]:
        if not rows:
            self.n_fallback += 1
            return None
        import numpy as np
        X = np.asarray([features(r) for r in rows], dtype=float)
        try:
            p = self.model.predict_proba(X)[:, 1]
        except Exception:
            self.n_fallback += 1
            return None
        self.n_calls += 1
        best = int(max(range(len(rows)), key=lambda i: p[i]))
        return str(rows[best]["candidate"])
