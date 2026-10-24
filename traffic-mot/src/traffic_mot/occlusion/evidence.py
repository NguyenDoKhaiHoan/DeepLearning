"""Observation history and conservative occlusion/exit hypotheses (no GT)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


def overlap_ioa(box: np.ndarray, others: np.ndarray) -> float:
    """Largest directed overlap over the candidate track area, excluding itself."""
    if len(others) == 0: return 0.0
    intersection = np.maximum(0, np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    return float(intersection.max() / area) if area > 0 else 0.0


def suspected_exit(box: np.ndarray, velocity: np.ndarray, shape: tuple, border_fraction: float,
                   require_outward: bool) -> bool:
    """Border contact plus outward motion; a hypothesis, not a measured exit label."""
    h, w = shape[:2]
    near = [box[0] <= w * border_fraction, box[1] <= h * border_fraction,
            box[2] >= w * (1 - border_fraction), box[3] >= h * (1 - border_fraction)]
    outward = [velocity[0] < 0, velocity[1] < 0, velocity[0] > 0, velocity[1] > 0]
    return any(a and (b or not require_outward) for a, b in zip(near, outward))


@dataclass
class EvidenceHistory:
    """Bounded observations; evidence freezes at disappearance rather than hallucinating."""
    hits: int = 0
    overlaps: deque = field(default_factory=deque)
    exit_suspected: bool = False
    last_timestamp: int | None = None

    def add_overlap(self, frame_id: int, overlaps: bool, window: int) -> None:
        self.overlaps.append((frame_id, overlaps))
        while len(self.overlaps) > window: self.overlaps.popleft()

    def occlusion_evidence(self, last_observed_frame: int, window: int, minimum: int) -> bool:
        # Include the first missed frame, which may be when overlap becomes heavy.
        return sum(bool(value) for frame, value in self.overlaps
                   if last_observed_frame - window + 1 <= frame <= last_observed_frame + 1) >= minimum
