"""Context-managed OpenCV capture for files, webcam indices and RTSP."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import cv2

from traffic_mot.utils.types import FramePacket, validate_frame
from traffic_mot.video.clock import FrameClock

LOGGER = logging.getLogger(__name__)


def sanitize_source(source: str | int) -> str:
    """Keep a usable asset reference without URL credentials/query tokens."""
    text = str(source)
    if "://" not in text: return text
    parsed = urlsplit(text)
    hostname = parsed.hostname or ""
    if ":" in hostname: hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, hostname + port, parsed.path, "", ""))


class VideoReader:
    """Read frames sequentially; a successful RTSP reconnect starts a new session.

    Use as a context manager to guarantee release on early break/error. Finite
    file decode failure before the declared frame count is an error, not EOF.
    """

    def __init__(self, source: str | int | Path, *, clock: str | None = None,
                 fallback_fps: float | None = None, reconnect_attempts: int = 3,
                 timeout_ms: int = 5000):
        if isinstance(source, Path): source = str(source.resolve())
        if isinstance(source, str) and source.isdecimal(): source = int(source)
        if isinstance(source, int) and source < 0: raise ValueError("Webcam index must be nonnegative")
        self.is_rtsp = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))
        self.is_live = isinstance(source, int) or self.is_rtsp
        if not self.is_live:
            path = Path(str(source)).resolve()
            if not path.is_file(): raise FileNotFoundError(f"Video source not found: {path}")
            source = str(path)
        if fallback_fps is not None and (not math.isfinite(fallback_fps) or fallback_fps <= 0):
            raise ValueError("fallback_fps must be positive and finite")
        if reconnect_attempts < 0 or timeout_ms < 1: raise ValueError("Invalid capture retry/timeout")
        self.source, self.source_reference = source, sanitize_source(source)
        self.clock_basis = clock or ("monotonic_session" if self.is_live else "media_pts")
        if self.is_live and self.clock_basis != "monotonic_session":
            raise ValueError("Live sources require monotonic_session clock in Phase 1")
        self.fallback_fps, self.reconnect_attempts, self.timeout_ms = fallback_fps, reconnect_attempts, timeout_ms
        self.capture = None
        self.fps: float | None = None
        self.frame_count: int | None = None
        self.frame_id = 0
        self.total_frames = 0
        self.reconnections = 0
        self.session_id = str(uuid4())
        self.closed = False

    def open(self) -> VideoReader:
        """Open once, validate timing metadata, and release on setup errors."""
        if self.capture is not None: return self
        if self.closed: raise RuntimeError("VideoReader is closed; create a new reader")
        if self.is_rtsp:
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms])
        else:
            self.capture = cv2.VideoCapture(self.source)
        try:
            if not self.capture.isOpened(): raise RuntimeError(f"Cannot open video: {self.source_reference}")
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and fps > 0 else self.fallback_fps
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_count = int(count) if not self.is_live and math.isfinite(count) and count > 0 else None
            self.clock = FrameClock(self.clock_basis, self.fps)
        except BaseException:
            self.close()
            raise
        return self

    def _reconnect(self) -> bool:
        for _ in range(self.reconnect_attempts):
            if self.capture is not None: self.capture.release()
            self.capture = None
            self.closed = False
            try:
                self.open()
                ok, frame = self.capture.read()
                if ok:
                    self.session_id = str(uuid4())
                    self.frame_id = 0
                    self.reconnections += 1
                    self.pending_frame = frame
                    LOGGER.warning("RTSP reconnected; started a new session for %s", self.source_reference)
                    return True
            except RuntimeError:
                continue
        return False

    def __iter__(self) -> VideoReader:
        return self.open()

    def __next__(self) -> FramePacket:
        if self.closed: raise StopIteration
        self.open()
        try:
            ok, frame = self.capture.read()
            if not ok and self.is_rtsp and self._reconnect():
                ok, frame = True, self.pending_frame
            if not ok:
                if self.is_live: raise RuntimeError(f"Live capture failed: {self.source_reference}")
                if self.total_frames == 0: raise RuntimeError("Video has no decodable frames")
                if self.frame_count is not None and self.frame_id < self.frame_count:
                    raise RuntimeError(f"Premature decode failure at frame {self.frame_id + 1}/{self.frame_count}")
                self.close()
                raise StopIteration
            validate_frame(frame)
            self.frame_id += 1
            self.total_frames += 1
            timestamp = self.clock.timestamp(self.frame_id, self.capture.get(cv2.CAP_PROP_POS_MSEC))
            return FramePacket(frame, self.frame_id, timestamp, self.clock_basis, self.session_id)
        except StopIteration:
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Idempotent capture release, including on exceptions."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.closed = True

    def __enter__(self) -> VideoReader:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
