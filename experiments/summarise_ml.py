"""Compare V4_active with the learned picker V5_learned, seed by seed.

    python experiments/summarise_ml.py results_ml/runs

Both strategies run on the same seeds, so the comparison is PAIRED: the
per-seed difference is far more informative than two independent means, and
with only 10 seeds an unpaired comparison would hide a real effect or invent
one. Reads only; writes nothing.
"""
import json
import statistics as st
import sys
from pathlib import Path

KEYS = [
    ("slot_accuracy", "slot accuracy %", 100, 1),
    ("decided_accuracy", "decided accuracy %", 100, 1),
    ("unverified", "UNVERIFIED slots", 1, 1),
    ("audit_time_s", "audit time (sim s)", 1, 0),
    ("frames", "frames", 1, 1),
    ("distance_m", "distance (m)", 1, 1),
]


def load(d, policy):
    out = {}
    for f in sorted(Path(d).glob(f"{policy}_seed*.json")):
        r = json.loads(f.read_text())
        out[r["seed"]] = r
    return out


def pm(vals, dp=1):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{dp}f}"
    return f"{st.mean(vals):.{dp}f} ± {st.pstdev(vals):.{dp}f}"


def main(d):
    a, b = load(d, "V4_active"), load(d, "V5_learned")
    seeds = sorted(set(a) & set(b))
    if not seeds:
        print("no paired runs found in", d)
        return
    missing = sorted((set(a) | set(b)) - set(seeds))
    if missing:
        print(f"WARNING: seeds without both policies, excluded: {missing}\n")

    print(f"Paired on {len(seeds)} seeds: {seeds[0]}-{seeds[-1]}\n")
    print(f"{'metric':<22}{'V4_active':>18}{'V5_learned':>18}"
          f"{'paired diff (V5-V4)':>24}")
    for key, name, scale, dp in KEYS:
        va = [a[s][key] * scale for s in seeds if a[s].get(key) is not None]
        vb = [b[s][key] * scale for s in seeds if b[s].get(key) is not None]
        if len(va) != len(seeds) or len(vb) != len(seeds):
            print(f"{name:<22}{pm(va, dp):>18}{pm(vb, dp):>18}{'(incomplete)':>24}")
            continue
        d_ = [y - x for x, y in zip(va, vb)]
        m = st.mean(d_)
        if len(d_) > 1 and st.pstdev(d_) > 0:
            se = st.stdev(d_) / len(d_) ** 0.5
            ci = f"{m:+.{dp}f}  (95% CI {m - 2 * se:+.{dp}f}..{m + 2 * se:+.{dp}f})"
        else:
            ci = f"{m:+.{dp}f}  (identical on every seed)" if m == 0 else f"{m:+.{dp}f}"
        print(f"{name:<22}{pm(va, dp):>18}{pm(vb, dp):>18}{ci:>24}")

    ca, cb = sum(a[s]["caught"] for s in seeds), sum(b[s]["caught"] for s in seeds)
    real = sum(a[s]["true_discrepancies"] for s in seeds)
    print(f"\n{'discrepancies caught':<22}{f'{ca}/{real}':>18}{f'{cb}/{real}':>18}")
    print(f"{'false alarms':<22}{sum(a[s]['false_alarms'] for s in seeds):>18}"
          f"{sum(b[s]['false_alarms'] for s in seeds):>18}")
    print(f"{'false EMPTY':<22}{sum(a[s]['false_empty'] for s in seeds):>18}"
          f"{sum(b[s]['false_empty'] for s in seeds):>18}")

    calls = sum(b[s].get("nbv_calls", 0) for s in seeds)
    fb = sum(b[s].get("nbv_fallbacks", 0) for s in seeds)
    print(f"\nV5 picker: {calls} scored decisions, {fb} fell back to the V4 rule")

    same = [s for s in seeds if a[s]["slot_accuracy"] == b[s]["slot_accuracy"]]
    up = [s for s in seeds if b[s]["slot_accuracy"] > a[s]["slot_accuracy"]]
    dn = [s for s in seeds if b[s]["slot_accuracy"] < a[s]["slot_accuracy"]]
    print(f"slot accuracy: V5 better on {len(up)} seeds {up}, "
          f"worse on {len(dn)} {dn}, equal on {len(same)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_ml/runs")
