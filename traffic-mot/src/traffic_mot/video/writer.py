"""MP4 output with explicit CFR timing and frame-size checks."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from traffic_mot.utils.types import validate_frame


class VideoWriter:
    """Write original-size BGR frames; reject silently distorted timelines."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], *, codec: str = "mp4v",
                 enforce_media_cfr: bool = True):
        if not math.isfinite(fps) or fps <= 0: raise ValueError("Output MP4 requires known positive FPS")
        if len(codec) != 4: raise ValueError("codec must have four characters")
        if min(size) <= 0 or any(n % 2 for n in size):
            raise ValueError("MP4 dimensions must be positive and even; automatic cropping is disabled")
        if path.exists(): raise FileExistsError(f"Refusing to overwrite video: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.size, self.fps, self.count = path, size, fps, 0
        self.enforce_media_cfr = enforce_media_cfr
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Cannot create MP4: {path}")

    def write(self, frame: np.ndarray, timestamp_ms: int) -> None:
        validate_frame(frame)
        if frame.shape[:2] != (self.size[1], self.size[0]): raise ValueError("Output frame dimensions changed")
        if self.enforce_media_cfr and abs(timestamp_ms - self.count * 1000 / self.fps) > max(2, 500 / self.fps):
            raise ValueError("VFR/PTS discontinuity cannot be preserved by CFR MP4 writer; choose explicit frame_fps retiming")
        self.writer.write(frame)
        self.count += 1

    def close(self) -> None:
        self.writer.release()
