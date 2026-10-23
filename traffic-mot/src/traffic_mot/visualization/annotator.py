"""Headless detector-only overlays; drawing never changes exported coordinates."""
from __future__ import annotations

import cv2
import numpy as np

from traffic_mot.utils.types import DetectionBatch


def annotate(frame: np.ndarray, detections: DetectionBatch, class_names: dict[int, str], *,
             show_confidence: bool = True, mock: bool = False, display_threshold: float = 0.3) -> np.ndarray:
    """Draw bounding boxes/class/confidence on a copy of the original frame."""
    result = frame.copy()
    h, w = result.shape[:2]
    for box, cls, score in zip(detections.boxes, detections.class_ids, detections.scores):
        if score < display_threshold: continue
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = min(w - 1, x1), min(w - 1, x2)
        y1, y2 = min(h - 1, y1), min(h - 1, y2)
        label = class_names[int(cls)] + (f" {score:.2f}" if show_confidence else "")
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 230, 120), 2)
        cv2.putText(result, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 120), 1, cv2.LINE_AA)
    if mock:
        cv2.putText(result, "MOCK DETECTOR - TEST FIXTURE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return result
