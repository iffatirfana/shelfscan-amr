"""Shared types for ShelfScan AMR."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    OK = "OK"
    MISPLACED = "MISPLACED"
    EMPTY = "EMPTY"
    UNVERIFIED = "UNVERIFIED"


EMPTY_CLASS = "slot_empty"


@dataclass
class Observation:
    """What the detector reports for one slot in one frame.

    label:       predicted class ("sku_A", ..., or "slot_empty"), None if nothing
                 was detected for this slot.
    confidence:  detector confidence in [0, 1] for that label.
    visibility:  estimated fraction of the slot region that is visible, [0, 1].
                 Derived from slot geometry vs. detected occluders.
    occluder_side: if something blocks the slot, which side of the slot it sits on
                 in the image ("left", "right", "front"), else None.
    viewpoint:   name of the camera viewpoint the frame was taken from.
    """
    slot_id: str
    label: Optional[str]
    confidence: float
    visibility: float
    occluder_side: Optional[str] = None
    viewpoint: str = "center"
    t: float = 0.0


@dataclass
class SlotRecord:
    slot_id: str
    expected_sku: Optional[str]          # None means the slot is expected to be empty
    status: Status = Status.UNVERIFIED
    confidence: float = 0.0
    last_checked: Optional[float] = None
    observed_label: Optional[str] = None
    last_viewpoint: Optional[str] = None
    last_occluder_side: Optional[str] = None
    n_observations: int = 0
    evidence: list = field(default_factory=list)   # e.g. saved image paths
