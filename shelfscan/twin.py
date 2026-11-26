"""Digital twin: expected vs. last observed state of every shelf slot."""
import csv
import json
from typing import Dict, Iterable, List, Optional

from .reasoning import classify
from .types import Observation, SlotRecord, Status


class DigitalTwin:
    def __init__(self, expected: Dict[str, Optional[str]]):
        """expected: slot_id -> expected SKU (None = slot should be empty)."""
        self.slots: Dict[str, SlotRecord] = {
            sid: SlotRecord(slot_id=sid, expected_sku=sku)
            for sid, sku in expected.items()
        }

    def update(self, obs: Observation, evidence_path: Optional[str] = None) -> Status:
        rec = self.slots[obs.slot_id]
        new_status = classify(obs, rec.expected_sku)
        rec.n_observations += 1

        # A decisive status replaces an earlier UNVERIFIED one. A new UNVERIFIED
        # never overwrites an earlier decisive status that had higher confidence.
        keep_old = (new_status == Status.UNVERIFIED
                    and rec.status != Status.UNVERIFIED
                    and rec.confidence >= obs.confidence)
        if not keep_old:
            rec.status = new_status
            rec.confidence = obs.confidence
            rec.observed_label = obs.label
        rec.last_checked = obs.t
        rec.last_viewpoint = obs.viewpoint
        rec.last_occluder_side = obs.occluder_side
        rec.last_visibility = obs.visibility
        if evidence_path and new_status not in (Status.OK,):
            rec.evidence.append(evidence_path)
        return rec.status

    # ---- queries -----------------------------------------------------------
    def unresolved(self) -> List[SlotRecord]:
        return [r for r in self.slots.values() if r.status == Status.UNVERIFIED]

    def discrepancies(self) -> List[SlotRecord]:
        return [r for r in self.slots.values()
                if r.status in (Status.MISPLACED, Status.EMPTY)]

    def summary(self) -> Dict[str, int]:
        out = {s.value: 0 for s in Status}
        for r in self.slots.values():
            out[r.status.value] += 1
        return out

    # ---- reporting ---------------------------------------------------------
    def to_rows(self) -> Iterable[dict]:
        for r in self.slots.values():
            yield {
                "slot_id": r.slot_id,
                "expected": r.expected_sku or "(empty)",
                "observed": r.observed_label or "-",
                "status": r.status.value,
                "confidence": round(r.confidence, 3),
                "observations": r.n_observations,
                "evidence": ";".join(r.evidence),
            }

    def write_report(self, path_csv: str, path_json: Optional[str] = None):
        rows = list(self.to_rows())
        with open(path_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if path_json:
            with open(path_json, "w") as f:
                json.dump({"summary": self.summary(), "slots": rows}, f, indent=2)
