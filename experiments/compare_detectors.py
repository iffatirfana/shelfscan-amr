"""Paired comparison of two detectors on the SAME seeds, per label type.

    python experiments/compare_detectors.py \
        --a results_qr/runs --b results_v3/qr --label qr \
        --a results_bc/runs --b results_v3/bc --label barcode

A and B must have run the same seeds with the same policy, so every seed is a
matched pair and the difference can be tested within seed. That matters here:
seed-to-seed variation (how many cartons, how many blockers) dwarfs the
difference between detectors, and an unpaired comparison would drown in it.

Reports, per label type:
  * mean +- sd per seed, for the decode rate and slot accuracy
  * the POOLED decode rate (all codes read over all cartons present), which is
    not the mean of the per-seed rates and is labelled separately
  * the paired difference B - A with a 95% CI from the t distribution
  * the per-seed win/loss/tie count, because a mean hides a split

Reads only. Writes nothing, and never touches results/ or results_v2/.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

# two-sided t, 95%, for n-1 degrees of freedom; index by n
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
       14: 2.160, 15: 2.145, 16: 2.131, 20: 2.093, 25: 2.064, 30: 2.045}


def t95(n):
    if n in T95:
        return T95[n]
    return min((v for k, v in sorted(T95.items()) if k >= n), default=1.96)


def load(run_dir, policy="V4_active"):
    out = {}
    for f in sorted(Path(run_dir).glob(f"{policy}_seed*.json")):
        d = json.loads(f.read_text())
        out[d["seed"]] = d
    return out


def ci(diffs):
    """(mean, lo, hi) of the paired difference, 95% CI."""
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else 0.0), float("nan"), float("nan")
    m = st.mean(diffs)
    half = t95(n) * st.stdev(diffs) / math.sqrt(n)
    return m, m - half, m + half


def wld(diffs, eps=1e-9):
    w = sum(1 for d in diffs if d > eps)
    losses = sum(1 for d in diffs if d < -eps)
    return w, losses, len(diffs) - w - losses


def line(name, a_vals, b_vals, unit="pp", scale=100.0, higher_is_better=True):
    """One metric. `higher_is_better` is NOT decoration: for UNVERIFIED rate,
    audit time and frames a bigger number is a worse result, and calling a
    rise an improvement would invert the conclusion."""
    da = [scale * v for v in a_vals]
    db = [scale * v for v in b_vals]
    diffs = [y - x for x, y in zip(da, db)]
    m, lo, hi = ci(diffs)
    w, l, t = wld(diffs) if higher_is_better else wld([-d for d in diffs])
    sd_a = st.pstdev(da) if len(da) > 1 else 0.0
    sd_b = st.pstdev(db) if len(db) > 1 else 0.0
    if lo <= 0 <= hi:
        verdict = "no difference"
    else:
        b_wins = (lo > 0) if higher_is_better else (hi < 0)
        verdict = "B better" if b_wins else "A better"
    print(f"  {name:<22}{st.mean(da):6.1f} +- {sd_a:4.1f}   "
          f"{st.mean(db):6.1f} +- {sd_b:4.1f}   "
          f"{m:+6.1f} {unit} (95% CI {lo:+.1f}..{hi:+.1f})   "
          f"W/L/T {w}/{l}/{t}   {verdict}")
    return m, lo, hi


def pooled_decode(rows):
    got = sum(r["qr_slots_decoded"] for r in rows)
    tot = sum(r["qr_slots_with_carton"] for r in rows)
    return got, tot, (got / tot if tot else float("nan"))


def compare(a_dir, b_dir, label, a_name="A (v2)", b_name="B (v3)"):
    A, B = load(a_dir), load(b_dir)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        print(f"\n[{label}] no seeds in common between {a_dir} and {b_dir}")
        return
    missing = sorted((set(A) | set(B)) - set(seeds))
    print(f"\n=== {label}  |  {len(seeds)} paired seeds "
          f"{seeds[0]}-{seeds[-1]} ===")
    if missing:
        print(f"  (ignored, present in only one side: {missing})")
    print(f"  {'metric':<22}{a_name:>16}{b_name:>18}")
    ra = [A[s] for s in seeds]
    rb = [B[s] for s in seeds]
    line("decode rate %", [r["qr_decode_rate"] for r in ra],
         [r["qr_decode_rate"] for r in rb])
    line("slot accuracy %", [r["slot_accuracy"] for r in ra],
         [r["slot_accuracy"] for r in rb])
    line("UNVERIFIED rate %", [r["unverified_rate"] for r in ra],
         [r["unverified_rate"] for r in rb], higher_is_better=False)
    line("audit time s", [r["audit_time_s"] for r in ra],
         [r["audit_time_s"] for r in rb], unit="s", scale=1.0,
         higher_is_better=False)
    line("frames", [r["frames"] for r in ra], [r["frames"] for r in rb],
         unit="", scale=1.0, higher_is_better=False)

    ga, ta, pa = pooled_decode(ra)
    gb, tb, pb = pooled_decode(rb)
    print(f"  {'pooled decode':<22}{100 * pa:6.1f}  ({ga}/{ta})     "
          f"{100 * pb:6.1f}  ({gb}/{tb})")
    for nm, rows in ((a_name, ra), (b_name, rb)):
        print(f"  {nm}: wrong payloads {sum(r.get('qr_wrong_payload', 0) for r in rows)}"
              f", false alarms {sum(r['false_alarms'] for r in rows)}"
              f", false EMPTY {sum(r['false_empty'] for r in rows)}"
              f", caught {sum(r['caught'] for r in rows)}"
              f"/{sum(r['true_discrepancies'] for r in rows)}")
    print("  per seed  " + "  ".join(
        f"{s}:{100 * A[s]['qr_decode_rate']:.0f}->{100 * B[s]['qr_decode_rate']:.0f}"
        for s in seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True,
                    help="baseline run dir (repeat, paired with --b/--label)")
    ap.add_argument("--b", action="append", required=True,
                    help="new run dir")
    ap.add_argument("--label", action="append", required=True,
                    help="label type name for the heading")
    ap.add_argument("--a-name", default="A (v2)")
    ap.add_argument("--b-name", default="B (v3)")
    g = ap.parse_args()
    assert len(g.a) == len(g.b) == len(g.label), "--a/--b/--label must pair up"
    for a, b, lbl in zip(g.a, g.b, g.label):
        compare(a, b, lbl, g.a_name, g.b_name)


if __name__ == "__main__":
    main()
