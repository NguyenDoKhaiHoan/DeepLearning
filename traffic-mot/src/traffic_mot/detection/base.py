"""Detector interface independent of tracker and rendering."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from traffic_mot.utils.types import DetectionBatch

VISDRONE_CLASSES = dict(enumerate(("pedestrian", "people", "bicycle", "car", "van", "truck",
                                  "tricycle", "awning-tricycle", "bus", "motor"), start=1))
VEHICLE_IDS = frozenset(range(3, 11))


class BaseDetector(ABC):
    """Accept BGR uint8 frames and return vehicle detections in original pixels."""

    class_names: dict[int, str] = VISDRONE_CLASSES
    is_mock: bool = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionBatch:
        """Detect one frame without modifying the input."""
        raise NotImplementedError

    def detect_batch(self, frames: Sequence[np.ndarray]) -> list[DetectionBatch]:
        """Sequential fallback for adapters without native batch inference."""
        return [self.detect(frame) for frame in frames]

    def metadata(self) -> dict:
        """Describe detector provenance in each run manifest."""
        return {"name": type(self).__name__, "mock": self.is_mock}
