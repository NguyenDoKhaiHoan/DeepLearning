"""Typed data at detector/video boundaries; no tracking identities in Phase 1."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def validate_frame(frame: np.ndarray) -> None:
    """Require a nonempty BGR uint8 image as returned by OpenCV."""
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
            frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError("frame must be a nonempty BGR uint8 HxWx3 array")


@dataclass(frozen=True)
class DetectionBatch:
    """Original-pixel xyxy boxes, scores and original class/postprocessor row IDs."""

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    detection_indices: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.scores)
        if self.boxes.shape != (n, 4) or any(x.shape != (n,) for x in
                (self.scores, self.class_ids, self.detection_indices)):
            raise ValueError("DetectionBatch needs Nx4 boxes and aligned N-element arrays")
        if self.boxes.dtype != np.float32 or self.scores.dtype != np.float32:
            raise ValueError("boxes and scores must be float32")
        if any(not np.issubdtype(x.dtype, np.integer) for x in (self.class_ids, self.detection_indices)):
            raise ValueError("class_ids and detection_indices must be integers")
        if (not np.isfinite(self.boxes).all() or not np.isfinite(self.scores).all() or
                (self.boxes < 0).any() or (self.boxes[:, 2:] <= self.boxes[:, :2]).any() or
                ((self.scores < 0) | (self.scores > 1)).any()):
            raise ValueError("Invalid boxes or confidence values")
        if len(set(self.detection_indices.tolist())) != n or (self.detection_indices < 0).any():
            raise ValueError("detection_indices must be unique and nonnegative within a frame")

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls) -> DetectionBatch:
        """Return a correctly shaped empty frame, including boxes (0,4)."""
        return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32),
                   np.empty(0, np.int64), np.empty(0, np.int64))

    def records(self, class_names: dict[int, str]) -> list[dict]:
        """Serialize without rounding coordinates or changing model class IDs."""
        return [{"bbox": box.tolist(), "confidence": float(score), "class_id": int(cls),
                 "class_name": class_names[int(cls)], "detection_index": int(index)}
                for box, score, cls, index in zip(self.boxes, self.scores, self.class_ids, self.detection_indices)]


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame with an explicit source/session clock."""

    frame: np.ndarray
    frame_id: int
    timestamp_ms: int
    time_basis: str
    session_id: str
