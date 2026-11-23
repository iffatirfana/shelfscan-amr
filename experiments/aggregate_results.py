"""Aggregate Webots result JSONs (results/*.json) into the table for the concept note.

    python experiments/aggregate_results.py results/
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main(d="results"):
    by = defaultdict(list)
    for p in Path(d).glob("*.json"):
        r = json.loads(p.read_text())
        if "policy" in r:
            by[r["policy"]].append(r)
    if not by:
        sys.exit("no result files found")
    print(f"{'policy':10s} {'runs':>4s} {'detect':>13s} {'false_alarm':>12s} "
          f"{'unresolved':>11s} {'time_s':>8s} {'dist_m':>7s}")
    for pol in sorted(by):
        rs = by[pol]
        det = [r["detection_rate"] for r in rs]
        sd = st.pstdev(det) if len(det) > 1 else 0.0
        print(f"{pol:10s} {len(rs):4d} {st.mean(det):7.3f}±{sd:.3f} "
              f"{st.mean(r['false_alarms'] for r in rs):12.2f} "
              f"{st.mean(r['unresolved_rate'] for r in rs):11.3f} "
              f"{st.mean(r['audit_time_s'] for r in rs):8.1f} "
              f"{st.mean(r['distance_m'] for r in rs):7.1f}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["results"]))
