"""Summarise the label-reading MVP runs (QR / Code128) in one table.

    python experiments/summarise_labels.py results_qr/runs [results_bc/runs ...]

Reads the per-run JSON the audit controller writes and prints slot accuracy,
UNVERIFIED rate, code decode rate and audit time, mean +- sd over the seeds
present. Reads only; writes nothing, and never touches results/ or results_v2/.
"""
import json
import statistics as st
import sys
from pathlib import Path


def load(run_dir, policy="V4_active"):
    rows = []
    for f in sorted(Path(run_dir).glob(f"{policy}_seed*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.pstdev(vals):.1f}"


def pct(rows, key):
    return ms([100 * r[key] for r in rows if r.get(key) is not None])


def main(dirs):
    print(f"{'run':<22}{'seeds':>6}{'slot acc %':>16}{'UNVERIFIED %':>16}"
          f"{'decode %':>14}{'wrong':>7}{'caught':>9}{'audit s':>14}")
    for d in dirs:
        rows = load(d)
        if not rows:
            print(f"{d:<22}{'(no runs)':>6}")
            continue
        caught = sum(r["caught"] for r in rows)
        real = sum(r["true_discrepancies"] for r in rows)
        label = rows[0].get("label_type") or ("qr" if rows[0].get("qr") else "none")
        name = f"{Path(d).parent.name} [{label}]"
        print(f"{name:<22}{len(rows):>6}{pct(rows, 'slot_accuracy'):>16}"
              f"{pct(rows, 'unverified_rate'):>16}"
              f"{pct(rows, 'qr_decode_rate'):>14}"
              f"{sum(r.get('qr_wrong_payload', 0) for r in rows):>7}"
              f"{f'{caught}/{real}':>9}{ms([r['audit_time_s'] for r in rows]):>14}")
        seeds = sorted(r["seed"] for r in rows)
        print(f"{'':<22}seeds {seeds[0]}-{seeds[-1]}, "
              f"{rows[0]['n_slots']} slots each, "
              f"decided accuracy {pct(rows, 'decided_accuracy')}%, "
              f"false EMPTY {sum(r['false_empty'] for r in rows)}, "
              f"false alarms {sum(r['false_alarms'] for r in rows)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results_qr/runs"])
