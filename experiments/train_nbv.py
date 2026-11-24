"""Train the learned next-best-view picker (V5, MVP).

    python experiments/train_nbv.py

Input : results_ml/nbv_train.csv, one row per CANDIDATE viewpoint at every V4
        next-best-view decision, logged by the audit controller.
Output: models_ml/nbv_tree.joblib + models_ml/nbv_report.txt

Seed discipline: trains on the 12 training seeds (1-12) only, reports on the
dev seeds (13-14) as a held-out check, and asserts that no row from 101-130
ever reaches the model. Evaluation seeds are never trained on.

The honest metric is per-DECISION top-1 agreement with V4, not per-row
accuracy: most rows are negatives, so a model that answered "not chosen" to
everything would score well per row and be useless.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shelfscan.nbv_model import FEATURE_NAMES, features  # noqa: E402

TRAIN_SEEDS = set(range(1, 13))
DEV_SEEDS = {13, 14}
FORBIDDEN = set(range(101, 131))


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        r["seed"] = int(r["seed"])
    return rows


def decisions(rows):
    """Group candidate rows back into decisions."""
    g = defaultdict(list)
    for r in rows:
        g[(r["seed"], r["slot_id"], r["n_tried"])].append(r)
    return g


def top1(model, groups):
    """Fraction of decisions where the model's best candidate is V4's."""
    import numpy as np
    ok = n = 0
    for _, rs in groups.items():
        truth = [r["candidate"] for r in rs if int(r["chosen"]) == 1]
        if not truth:
            continue            # V4 had no candidate left; nothing to imitate
        X = np.asarray([features(r) for r in rs], dtype=float)
        p = model.predict_proba(X)[:, 1]
        pick = rs[int(np.argmax(p))]["candidate"]
        ok += int(pick == truth[0])
        n += 1
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "results_ml/nbv_train.csv"))
    ap.add_argument("--out", default=str(ROOT / "models_ml"))
    ap.add_argument("--max-depth", type=int, default=5)
    a = ap.parse_args()

    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_text
    import joblib

    rows = load(a.table)
    seeds = sorted({r["seed"] for r in rows})
    leaked = sorted(set(seeds) & FORBIDDEN)
    assert not leaked, f"evaluation seeds present in the training table: {leaked}"

    tr = [r for r in rows if r["seed"] in TRAIN_SEEDS]
    dev = [r for r in rows if r["seed"] in DEV_SEEDS]
    assert tr, "no training rows"

    Xtr = np.asarray([features(r) for r in tr], dtype=float)
    ytr = np.asarray([int(r["chosen"]) for r in tr])

    model = DecisionTreeClassifier(max_depth=a.max_depth, random_state=0,
                                   class_weight="balanced")
    model.fit(Xtr, ytr)

    g_tr, g_dev = decisions(tr), decisions(dev)
    ok_tr, n_tr = top1(model, g_tr)
    ok_dev, n_dev = top1(model, g_dev) if dev else (0, 0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "nbv_tree.joblib")

    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                 key=lambda t: -t[1])
    lines = [
        "Learned next-best-view picker (V5) - MVP, simulation only",
        "",
        f"table            : {a.table}",
        f"seeds in table   : {seeds}",
        f"trained on       : {sorted(TRAIN_SEEDS & set(seeds))} "
        f"({len(tr)} candidate rows, {n_tr} decisions)",
        f"held out (dev)   : {sorted(DEV_SEEDS & set(seeds))} "
        f"({len(dev)} candidate rows, {n_dev} decisions)",
        "evaluation seeds 101-130: never used for training (asserted)",
        "",
        f"model            : DecisionTreeClassifier(max_depth={a.max_depth}, "
        f"class_weight='balanced')",
        f"positive rate    : {ytr.mean():.3f} of rows are V4's choice",
        "",
        f"top-1 agreement with V4, train : {ok_tr}/{n_tr} "
        f"({100 * ok_tr / max(n_tr, 1):.1f}%)",
        f"top-1 agreement with V4, dev   : {ok_dev}/{n_dev} "
        f"({100 * ok_dev / max(n_dev, 1):.1f}%)" if n_dev else
        "top-1 agreement with V4, dev   : (no dev rows)",
        "",
        "feature importances:",
    ]
    lines += [f"  {n:<16} {v:.3f}" for n, v in imp if v > 0]
    lines += ["", "tree:", export_text(model, feature_names=list(FEATURE_NAMES),
                                       max_depth=a.max_depth)]
    text = "\n".join(lines)
    (out / "nbv_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
