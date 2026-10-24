"""Pinned stock ByteTrack/BoT-SORT adapters, with no YOLO model or ReID."""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import yaml

from traffic_mot.tracking.base import BaseTracker, TrackObservation
from traffic_mot.utils.config import require
from traffic_mot.utils.types import DetectionBatch, validate_frame

_ID_LOCK = threading.RLock()


def check_backend_dependencies() -> None:
    """Check before importing Ultralytics, preventing its automatic lap installer."""
    for package, expected in (("ultralytics", "8.4.171"), ("lap", "0.5.12")):
        try: installed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Missing {package}; install traffic-mot/requirements.txt in this kernel environment") from exc
        require(installed == expected, f"Expected {package}=={expected}, found {installed}; install the pinned dependencies")


class UltralyticsTracker(BaseTracker):
    """Use stock matching/Kalman/GMC; isolate backend's global ID counter.

    No changes to high/low association costs or class gating. Counter swapping
    under a process lock permits independent interleaved camera/session instances.
    """

    def __init__(self, config: Path):
        self.config = yaml.safe_load(config.read_text(encoding="utf-8"))
        require(self.config["backend"] == "ultralytics" and self.config["backend_version"] == "8.4.171",
                "Unsupported tracker backend/version")
        self.args = SimpleNamespace(**self.config["tracker"])
        require(self.args.tracker_type in {"bytetrack", "botsort"}, "Unsupported tracker type")
        require(not getattr(self.args, "with_reid", False), "ReID extractor not integrated; baseline requires with_reid=false")
        require(not self.config["adapter"]["class_gating"] and
                not self.config["adapter"]["emit_predicted_as_observed"], "Stock baseline needs class gating off and observed-only output")
        require(0 <= self.args.track_low_thresh < self.args.track_high_thresh <= self.args.new_track_thresh <= 1,
                "Invalid tracker confidence thresholds")
        require(self.args.track_buffer >= 1 and 0 < self.args.match_thresh <= 1, "Invalid tracker buffer/matching cost")
        check_backend_dependencies()
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.basetrack import BaseTrack
        self.boxes_class, self.base_track = Boxes, BaseTrack
        self.backend_class = BYTETracker if self.args.tracker_type == "bytetrack" else BOTSORT
        self.id_counter = 0
        self.last_timestamp = None
        with self._id_scope(): self.backend = self._create_backend()

    def _create_backend(self):
        """Extension point; the stock adapter always builds the unmodified backend."""
        return self.backend_class(self.args)

    @contextmanager
    def _id_scope(self):
        with _ID_LOCK:
            previous = self.base_track._count
            self.base_track._count = self.id_counter
            try: yield
            finally:
                self.id_counter = self.base_track._count
                self.base_track._count = previous

    def update(self, detections: DetectionBatch, frame: np.ndarray,
               timestamp: int | None = None) -> list[TrackObservation]:
        validate_frame(frame)
        if timestamp is not None:
            require(timestamp >= 0 and (self.last_timestamp is None or timestamp >= self.last_timestamp),
                    "Tracker timestamps must be nonnegative and monotonic")
            self.last_timestamp = timestamp
        packed = np.column_stack((detections.boxes, detections.scores, detections.class_ids)).astype(np.float32)
        boxes = self.boxes_class(packed, frame.shape[:2]).numpy()
        with self._id_scope(): rows = self.backend.update(boxes, frame, timestamp_ms=timestamp)
        if rows.size == 0: return []
        require(rows.ndim == 2 and rows.shape[1] == 8, "Pinned tracker output API changed: expected Nx8")
        h, w = frame.shape[:2]
        output = []
        for row in rows:
            x1, y1, x2, y2, track_id, score, cls, index = row.tolist()
            index = int(index)
            require(0 <= index < len(detections) and int(cls) == detections.class_ids[index], "Tracker detection/class index mismatch")
            box = np.clip([x1, y1, x2, y2], [0, 0, 0, 0], [w, h, w, h])
            if box[2] <= box[0] or box[3] <= box[1]: continue
            output.append(TrackObservation(int(track_id), tuple(float(v) for v in box), score, int(cls),
                                           int(detections.detection_indices[index])))
        require(len({track.track_id for track in output}) == len(output), "Duplicate public IDs in a frame")
        return output

    def states(self) -> dict[str, list[int]]:
        """Expose lifecycle for diagnostics, never predicted boxes as observations."""
        return {"tracked": [t.track_id for t in self.backend.tracked_stracks if t.is_activated],
                "tentative": [t.track_id for t in self.backend.tracked_stracks if not t.is_activated],
                "lost": [t.track_id for t in self.backend.lost_stracks],
                "removed": [t.track_id for t in self.backend.removed_stracks]}

    def reset(self) -> None:
        with self._id_scope(): self.backend.reset()
        self.id_counter, self.last_timestamp = 0, None

    def metadata(self) -> dict:
        return {"backend": "ultralytics", "version": version("ultralytics"), "config": self.config,
                "with_reid": False, "recovery_enabled": False, "class_gating": False}
