"""Explicit, fixture-driven detector for tests; never presented as a real model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np

from traffic_mot.detection.base import BaseDetector, VEHICLE_IDS
from traffic_mot.utils.types import DetectionBatch, validate_frame


class MockDetector(BaseDetector):
    """Replay lists of {bbox, confidence, class_id} in original pixels.

    Exhausted fixtures yield empty batches. No procedural fake detections.
    """

    is_mock = True

    def __init__(self, frames: Sequence[Sequence[dict]] = (), confidence_threshold: float = 0.1):
        if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be finite and in [0,1]")
        self.frames = frames
        self.threshold = confidence_threshold
        self.index = 0

    @classmethod
    def from_json(cls, path: Path, confidence_threshold: float = 0.1) -> MockDetector:
        """Load a JSON array of frame detection arrays supplied explicitly."""
        frames = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(frames, list) or any(not isinstance(f, list) for f in frames):
            raise ValueError("Mock fixture must be a list of frame detection lists")
        return cls(frames, confidence_threshold)

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        validate_frame(frame)
        rows = self.frames[self.index] if self.index < len(self.frames) else []
        self.index += 1
        h, w = frame.shape[:2]
        boxes, scores, classes, indices = [], [], [], []
        for index, row in enumerate(rows):
            box = np.asarray(row["bbox"], dtype=np.float32)
            score, cls = float(row["confidence"]), row["class_id"]
            if cls not in VEHICLE_IDS or not isinstance(cls, int):
                raise ValueError("Mock fixture needs original vehicle class IDs 3..10")
            if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid mock fixture box/confidence")
            box = np.clip(box, [0, 0, 0, 0], [w, h, w, h]).astype(np.float32)
            if score < self.threshold or (box[2:] <= box[:2]).any():
                continue
            boxes.append(box); scores.append(score); classes.append(cls); indices.append(index)
        return DetectionBatch(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                              np.asarray(classes, np.int64), np.asarray(indices, np.int64))

    def metadata(self) -> dict:
        return {**super().metadata(), "fixture_frames": len(self.frames), "confidence_threshold": self.threshold}
