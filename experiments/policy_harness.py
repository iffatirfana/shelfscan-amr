"""Abstract policy harness (NOT Webots).

Purpose: check that the twin / reasoning / viewpoint / policy code runs end to end
and behaves sensibly, before wiring it to the simulator.

IMPORTANT: every number this prints depends on the assumptions in WORLD and
DETECTOR below (occlusion rate, how viewpoints change visibility, costs). They are
assumptions, not measurements. Do NOT report this output as a result. Report only
numbers from the Webots runs with the trained detector.

Usage:  python experiments/policy_harness.py --layouts 200 --seed 0
"""
import argparse
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelfscan.policies import POLICIES            # noqa: E402
from shelfscan.twin import DigitalTwin             # noqa: E402
from shelfscan.types import EMPTY_CLASS, Observation, Status  # noqa: E402

SKUS = ["sku_A", "sku_B", "sku_C", "sku_D"]

# ---- assumptions (edit and justify; they drive every number) ---------------
WORLD = dict(
    shelves=5, slots_per_shelf=8,
    p_expected_empty=0.05,   # slot is meant to hold nothing
    p_empty=0.08,            # item missing
    p_misplaced=0.08,        # wrong item present
    p_occluded=0.20,         # something blocks the view from the center
    p_dark=0.15,             # poor lighting at that slot
)
# visibility of the slot from each viewpoint when an occluder sits on `side`
VIS_OCCLUDED = {"center": 0.40, "same_side": 0.30, "opposite_side": 0.95, "close": 0.50}
COST = {"center": 1.0, "left": 2.0, "right": 2.0, "close": 2.5,
        "center_revisit": 1.5}                      # arbitrary time units
# ---------------------------------------------------------------------------


def make_world(rng):
    expected, actual, occ, dark = {}, {}, {}, {}
    for s in range(WORLD["shelves"]):
        for k in range(WORLD["slots_per_shelf"]):
            sid = f"S{s}-{k}"
            exp = None if rng.random() < WORLD["p_expected_empty"] else rng.choice(SKUS)
            r = rng.random()
            if exp is None:
                act = rng.choice(SKUS) if r < WORLD["p_misplaced"] else None
            elif r < WORLD["p_empty"]:
                act = None
            elif r < WORLD["p_empty"] + WORLD["p_misplaced"]:
                act = rng.choice([x for x in SKUS if x != exp])
            else:
                act = exp
            expected[sid], actual[sid] = exp, act
            occ[sid] = rng.choice(["left", "right"]) if rng.random() < WORLD["p_occluded"] else None
            dark[sid] = rng.random() < WORLD["p_dark"]
    return expected, actual, occ, dark


def true_status(exp, act):
    if act == exp:
        return Status.OK
    return Status.EMPTY if act is None else Status.MISPLACED


class Detector:
    """Stand-in for YOLO + visibility estimate."""
    def __init__(self, actual, occ, dark, rng):
        self.actual, self.occ, self.dark, self.rng = actual, occ, dark, rng

    def observe(self, sid, vp):
        side = self.occ[sid]
        if side is None:
            vis = 0.95
        elif vp == "close":
            vis = VIS_OCCLUDED["close"]
        elif vp == "center":
            vis = VIS_OCCLUDED["center"]
        else:
            vis = VIS_OCCLUDED["same_side" if vp == side else "opposite_side"]
        vis = min(1.0, max(0.0, vis + self.rng.gauss(0, 0.03)))

        good_light = (not self.dark[sid]) or vp == "close"
        if vis >= 0.7 and good_light:
            conf = min(1.0, max(0.0, self.rng.gauss(0.90, 0.05)))
            truth = self.actual[sid] or EMPTY_CLASS
            label = truth if self.rng.random() < 0.97 else self.rng.choice(SKUS + [EMPTY_CLASS])
        else:
            conf = min(1.0, max(0.0, self.rng.gauss(0.45, 0.10)))
            label = self.rng.choice(SKUS + [EMPTY_CLASS, None])
        seen_side = side if (side and vis < 0.7 and self.rng.random() < 0.9) else None
        return Observation(sid, label, conf, vis, seen_side, vp)


def run_one(policy_name, rng_seed):
    rng = random.Random(rng_seed)
    expected, actual, occ, dark = make_world(rng)
    det = Detector(actual, occ, dark, random.Random(rng_seed + 1))
    twin = DigitalTwin(expected)
    order = list(expected)
    seen_center = set()
    time_units = 0.0

    def observe(sid, vp):
        nonlocal time_units
        if vp == "center":
            time_units += COST["center"] if sid not in seen_center else COST["center_revisit"]
            seen_center.add(sid)
        else:
            time_units += COST[vp]
        return det.observe(sid, vp)

    POLICIES[policy_name](twin, observe, order)

    real_disc = [s for s in order if true_status(expected[s], actual[s]) != Status.OK]
    caught = sum(1 for s in real_disc
                 if twin.slots[s].status == true_status(expected[s], actual[s]))
    false_alarms = sum(1 for s in order
                       if twin.slots[s].status in (Status.MISPLACED, Status.EMPTY)
                       and true_status(expected[s], actual[s]) == Status.OK)
    wrong_status = sum(1 for s in real_disc
                       if twin.slots[s].status in (Status.MISPLACED, Status.EMPTY)
                       and twin.slots[s].status != true_status(expected[s], actual[s]))
    return dict(
        detect=caught / len(real_disc) if real_disc else 1.0,
        false_alarms=false_alarms,
        wrong_status=wrong_status,
        unresolved=len(twin.unresolved()) / len(order),
        time=time_units,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("ABSTRACT HARNESS - assumption-driven, not a Webots result\n")
    print(f"{'policy':10s} {'detect':>10s} {'false_alarm':>12s} {'unresolved':>11s} {'time':>8s}")
    for name in POLICIES:
        runs = [run_one(name, a.seed * 100000 + i * 7) for i in range(a.layouts)]
        d = [r["detect"] for r in runs]
        print(f"{name:10s} {st.mean(d):6.3f}±{st.pstdev(d):.3f} "
              f"{st.mean(r['false_alarms'] for r in runs):12.2f} "
              f"{st.mean(r['unresolved'] for r in runs):11.3f} "
              f"{st.mean(r['time'] for r in runs):8.1f}")


if __name__ == "__main__":
    main()
