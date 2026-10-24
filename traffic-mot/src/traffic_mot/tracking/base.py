"""Tracker contract and observed output records."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from traffic_mot.utils.types import DetectionBatch


@dataclass(frozen=True)
class TrackObservation:
    """A confirmed observation; lost predictions are never exported as detections."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    detection_index: int
    track_state: str = "tracked"
    observed: bool = True


class BaseTracker(ABC):
    """One sequential state machine per camera/session; timestamps may be unknown."""

    @abstractmethod
    def update(self, detections: DetectionBatch, frame: np.ndarray,
               timestamp: int | None = None) -> list[TrackObservation]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
