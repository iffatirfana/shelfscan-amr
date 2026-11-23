"""Turn one observation + the expected inventory into a slot status.

Key rule: "nothing detected" is never evidence of an empty slot. EMPTY requires
the slot to be clearly visible AND the detector to positively report slot_empty.
"""
from .types import Observation, Status, EMPTY_CLASS

# Tunable thresholds (set from validation data, not guessed, before final results).
MIN_CONF = 0.60          # detector must be at least this confident to decide
MIN_VISIBILITY = 0.70    # slot must be at least this visible to decide


def classify(obs: Observation, expected_sku, min_conf=MIN_CONF,
             min_vis=MIN_VISIBILITY) -> Status:
    if obs.label is None or obs.confidence < min_conf or obs.visibility < min_vis:
        return Status.UNVERIFIED

    if obs.label == EMPTY_CLASS:
        return Status.OK if expected_sku is None else Status.EMPTY

    # An item was seen.
    if expected_sku is None:
        return Status.MISPLACED        # item present where slot should be empty
    return Status.OK if obs.label == expected_sku else Status.MISPLACED
