"""Aggregate Webots audit runs (results/runs/*.json) into results/results.csv
(one row per run) and results/summary.md (mean ± std per policy).

    python experiments/aggregate_results.py [results/runs] [results]

Only per-run JSON files written by the shelfscan_audit controller inside Webots
are read. Nothing here is simulated or estimated.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

COLS = ["policy", "seed", "n_slots", "correct", "slot_accuracy", "decided", "decided_accuracy",
        "unverified", "unverified_rate", "n_truly_not_empty", "false_empty", "false_empty_rate",
        "n_truly_empty", "empty_caught", "true_discrepancies", "caught", "detection_rate",
        "false_alarms", "audit_time_s", "distance_m", "frames"]
NAMES = {"V1_fixed": "Fixed patrol (V1)", "V3_rescan": "Same-position rescan (V3)",
         "V4_active": "Active verification (V4)"}


def ms(xs, pct=False, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    k = 100 if pct else 1
    m = st.mean(xs) * k
    sd = (st.pstdev(xs) * k) if len(xs) > 1 else 0.0
    return (fmt + " ± " + fmt).format(m, sd) + ("%" if pct else "")


def main(runs_dir="results/runs", out_dir="results"):
    runs = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*.json"))]
    runs = [r for r in runs if "policy" in r and "slot_accuracy" in r]
    if not runs:
        sys.exit("no result files found")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs.sort(key=lambda r: (r["seed"], r["policy"]))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    by = defaultdict(list)
    for r in runs:
        by[r["policy"]].append(r)
    # Only seeds where every policy finished, so the comparison is paired.
    common = set.intersection(*({r["seed"] for r in rs} for rs in by.values()))
    lines = [
        f"Webots R2025a, {len(common)} unseen seeds ({min(common)}-{max(common)}), "
        f"18 slots per seed, mean ± std over seeds.\n" if common else "",
        "| Strategy | Runs | Slot accuracy | UNVERIFIED rate | False-EMPTY rate "
        "| Discrepancies caught | False alarms / run | Audit time (sim s) | Frames / run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in sorted(by):
        rs = [r for r in by[pol] if r["seed"] in common]
        caught = sum(r["caught"] for r in rs)
        real = sum(r["true_discrepancies"] for r in rs)
        lines.append(
            f"| {NAMES.get(pol, pol)} | {len(rs)} | {ms([r['slot_accuracy'] for r in rs], True)} "
            f"| {ms([r['unverified_rate'] for r in rs], True)} "
            f"| {ms([r['false_empty_rate'] for r in rs], True)} "
            f"| {caught}/{real} | {ms([r['false_alarms'] for r in rs], fmt='{:.2f}')} "
            f"| {ms([r['audit_time_s'] for r in rs], fmt='{:.0f}')} "
            f"| {ms([r['frames'] for r in rs], fmt='{:.1f}')} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
