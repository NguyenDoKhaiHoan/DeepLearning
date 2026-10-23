"""Detector-only video pipeline with auditable JSONL/MP4 output and cleanup."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import logging
from pathlib import Path
import platform
import time
from uuid import uuid4

from traffic_mot.detection.base import BaseDetector
from traffic_mot.detection.bridge import file_sha256
from traffic_mot.video.reader import VideoReader
from traffic_mot.video.writer import VideoWriter
from traffic_mot.visualization.annotator import annotate

LOGGER = logging.getLogger(__name__)


class VideoPipeline:
    """Run ordered detection and retain a truthful manifest even after failure."""

    def __init__(self, detector: BaseDetector, *, output_dir: Path, camera_id: str = "cam_01",
                 save_video: bool = False, save_json: bool = False, max_frames: int | None = None,
                 seed: int = 42, clock: str | None = None, fallback_fps: float | None = None,
                 reconnect_attempts: int = 3, show_confidence: bool = True, display_threshold: float = 0.3):
        if max_frames is not None and max_frames < 1: raise ValueError("max_frames must be positive")
        if not camera_id: raise ValueError("camera_id must be nonempty")
        if not 0 <= display_threshold <= 1: raise ValueError("display_threshold must be in [0,1]")
        self.detector, self.output_dir, self.camera_id = detector, output_dir.resolve(), camera_id
        self.save_video, self.save_json, self.max_frames, self.seed = save_video, save_json, max_frames, seed
        self.clock, self.fallback_fps = clock, fallback_fps
        self.reconnect_attempts, self.show_confidence = reconnect_attempts, show_confidence
        self.display_threshold = display_threshold

    def process(self, source: str | int | Path) -> dict:
        """Process once into a new run directory; preserve JSONL on partial runs."""
        reader = VideoReader(source, clock=self.clock, fallback_fps=self.fallback_fps,
                             reconnect_attempts=self.reconnect_attempts)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": "1.0.0", "phase": 1, "run_id": str(uuid4()), "status": "incomplete",
                    "camera_id": self.camera_id, "video_source": reader.source_reference,
                    "started_at": datetime.now(timezone.utc).isoformat(), "seed": self.seed,
                    "frames_processed": 0, "detection_count": 0, "mock": self.detector.is_mock,
                    "output_dir": str(self.output_dir), "save_video": self.save_video, "save_json": self.save_json,
                    "max_frames": self.max_frames, "clock": self.clock, "fallback_fps": self.fallback_fps,
                    "display_threshold": self.display_threshold,
                    "platform": platform.platform(), "python": platform.python_version(),
                    "environment": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
                    "timing_scope": "smoke run wall time; not a steady-state performance benchmark"}
        if not reader.is_live: manifest["source_sha256"] = file_sha256(Path(reader.source))
        json_stream = None
        writers: dict[str, VideoWriter] = {}
        start = time.perf_counter()
        detector_ms = 0.0
        try:
            if self.save_json:
                json_stream = (self.output_dir / "detections.jsonl").open("x", encoding="utf-8")
            with reader:
                manifest.update(source_fps=reader.fps, source_frame_count=reader.frame_count)
                while self.max_frames is None or manifest["frames_processed"] < self.max_frames:
                    try: packet = next(reader)
                    except StopIteration: break
                    # Synchronization is only relevant for a CUDA detector.
                    device = getattr(self.detector, "device", None)
                    if device is not None and device.type == "cuda":
                        import torch
                        torch.cuda.synchronize(device)
                    before = time.perf_counter()
                    detections = self.detector.detect(packet.frame)
                    if device is not None and device.type == "cuda": torch.cuda.synchronize(device)
                    detector_ms += (time.perf_counter() - before) * 1000
                    if self.save_video:
                        if packet.session_id not in writers:
                            if reader.fps is None: raise ValueError("Saving MP4 requires source FPS or --fallback-fps")
                            filename = "annotated.mp4" if not writers else f"annotated_{packet.session_id}.mp4"
                            writers[packet.session_id] = VideoWriter(self.output_dir / filename, reader.fps,
                                (packet.frame.shape[1], packet.frame.shape[0]), enforce_media_cfr=not reader.is_live)
                        rendered = annotate(packet.frame, detections, self.detector.class_names,
                                            show_confidence=self.show_confidence, mock=self.detector.is_mock,
                                            display_threshold=self.display_threshold)
                        writers[packet.session_id].write(rendered, packet.timestamp_ms)
                    if json_stream:
                        record = {"schema_version": "1.0.0", "record_type": "detection_frame", "mock": self.detector.is_mock,
                                  "camera_id": self.camera_id, "session_id": packet.session_id,
                                  "frame_id": packet.frame_id, "timestamp_ms": packet.timestamp_ms,
                                  "time_basis": packet.time_basis, "video_source": reader.source_reference,
                                  "width": packet.frame.shape[1], "height": packet.frame.shape[0],
                                  "detections": detections.records(self.detector.class_names)}
                        json_stream.write(json.dumps(record, allow_nan=False) + "\n")
                        json_stream.flush()
                    manifest["frames_processed"] += 1
                    manifest["detection_count"] += len(detections)
                    if manifest["frames_processed"] == 1:
                        manifest.update(first_timestamp_ms=packet.timestamp_ms,
                                        frame_size=[packet.frame.shape[1], packet.frame.shape[0]])
                    manifest.update(last_timestamp_ms=packet.timestamp_ms, time_basis=packet.time_basis)
                    if manifest["frames_processed"] % 30 == 0:
                        LOGGER.info("Processed %d frames, %d detections", manifest["frames_processed"], manifest["detection_count"])
                manifest["termination"] = ("frame_limit" if self.max_frames is not None and
                    manifest["frames_processed"] >= self.max_frames else "eof")
                manifest["reconnections"] = reader.reconnections
            manifest["status"] = "complete"
        except BaseException as exc:
            # Error strings from capture include only the sanitized source reference.
            manifest.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            reader.close()
            if json_stream: json_stream.close()
            for writer in writers.values(): writer.close()
            manifest.update(detector=self.detector.metadata(), elapsed_seconds=time.perf_counter() - start,
                            detector_total_ms=detector_ms,
                            ended_at=datetime.now(timezone.utc).isoformat(),
                            video_files=[writer.path.name for writer in writers.values()])
            (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        return manifest
