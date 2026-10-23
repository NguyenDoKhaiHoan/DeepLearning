"""Explicit video/live clocks; inference speed never determines media time."""
from __future__ import annotations

import math
import time


class FrameClock:
    """Normalize source PTS or compute a declared CFR/monotonic timeline."""

    def __init__(self, basis: str, fps: float | None = None):
        if basis not in {"media_pts", "frame_fps", "monotonic_session"}:
            raise ValueError("Unknown clock basis")
        if basis == "frame_fps" and (fps is None or not math.isfinite(fps) or fps <= 0):
            raise ValueError("frame_fps clock requires positive FPS")
        self.basis, self.fps = basis, fps
        self.origin: float | None = None
        self.last: float | None = None

    def timestamp(self, frame_id: int, pts_ms: float | None = None) -> int:
        """Produce elapsed ms, rejecting stalled/backwards PTS before rounding."""
        if self.basis == "frame_fps":
            value = (frame_id - 1) * 1000 / self.fps
        elif self.basis == "monotonic_session":
            now = time.monotonic() * 1000
            if self.origin is None: self.origin = now
            value = now - self.origin
        else:
            if pts_ms is None or not math.isfinite(pts_ms) or pts_ms < 0:
                raise ValueError("Invalid media PTS; explicitly choose --clock frame_fps with known FPS")
            if self.origin is None: self.origin = pts_ms
            value = pts_ms - self.origin
            if self.last is not None and value <= self.last:
                raise ValueError("Stalled/backwards media PTS; choose frame_fps only for known CFR input")
        if self.last is not None and value < self.last:
            raise ValueError("Source timestamp moved backwards")
        self.last = value
        return round(value)
