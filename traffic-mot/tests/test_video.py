"""Video clock, resource lifecycle, reconnect, headless MP4 and CLI tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from traffic_mot.detection.mock import MockDetector
from traffic_mot.video.clock import FrameClock
from traffic_mot.video.pipeline import VideoPipeline
from traffic_mot.video.reader import VideoReader, sanitize_source
from traffic_mot.video.writer import VideoWriter
from traffic_mot.utils.verify import verify_detection_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mp4(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (200, 100))
    assert writer.isOpened()
    for i in range(4): writer.write(np.full((100, 200, 3), 30 + i * 20, np.uint8))
    writer.release()
    return path


def test_file_reader_timeline_and_release(mp4):
    with VideoReader(mp4) as reader:
        frames = list(reader)
    assert [f.frame_id for f in frames] == [1, 2, 3, 4]
    assert [f.timestamp_ms for f in frames] == [0, 200, 400, 600]
    assert all(f.time_basis == "media_pts" for f in frames)
    assert len({f.session_id for f in frames}) == 1
    assert reader.closed and reader.capture is None
    assert reader.fps == 5


def test_reader_release_on_early_exit(mp4):
    with VideoReader(mp4) as reader:
        assert next(reader).frame_id == 1
    assert reader.closed and reader.capture is None


def test_missing_and_empty_video(tmp_path):
    with pytest.raises(FileNotFoundError): VideoReader(tmp_path / "missing.mp4")
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    reader = VideoReader(path)
    with pytest.raises(RuntimeError, match="Cannot open"): reader.open()
    assert reader.closed and reader.capture is None


def test_clock_explicit_fps_and_pts_validation():
    clock = FrameClock("frame_fps", 25)
    assert [clock.timestamp(i) for i in [1, 2, 3]] == [0, 40, 80]
    clock = FrameClock("media_pts")
    assert clock.timestamp(1, 100) == 0
    with pytest.raises(ValueError, match="Stalled"): clock.timestamp(2, 100)
    with pytest.raises(ValueError, match="positive FPS"): FrameClock("frame_fps", 0)


class FakeCapture:
    def __init__(self, frames, *, fps=25, count=0, opened=True):
        self.frames, self.fps, self.count, self.opened = list(frames), fps, count, opened
        self.released = False
    def isOpened(self): return self.opened
    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_POS_MSEC: 0}.get(prop, 0)
    def read(self): return (True, self.frames.pop(0)) if self.frames else (False, None)
    def release(self): self.released = True


def test_webcam_and_rtsp_reconnect(monkeypatch):
    import traffic_mot.video.reader as module
    frame = np.zeros((100, 200, 3), np.uint8)
    captures = [FakeCapture([frame]), FakeCapture([], opened=False), FakeCapture([frame])]
    opened = []
    def factory(*args):
        opened.append(args)
        return captures[len(opened) - 1]
    monkeypatch.setattr(module.cv2, "VideoCapture", factory)
    with VideoReader("rtsp://user:password@host/live?token=secret", reconnect_attempts=2) as reader:
        first, second = next(reader), next(reader)
        assert first.session_id != second.session_id
        assert first.frame_id == second.frame_id == 1
        assert second.time_basis == "monotonic_session"
        assert reader.reconnections == 1
    assert all(c.released for c in captures)
    assert sanitize_source("rtsp://user:password@host/live?token=secret") == "rtsp://host/live"
    webcam = FakeCapture([frame])
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda source: webcam)
    with VideoReader("0") as reader:
        assert reader.source == 0 and next(reader).time_basis == "monotonic_session"
    assert webcam.released


def test_premature_file_decode_failure(mp4, monkeypatch):
    import traffic_mot.video.reader as module
    capture = FakeCapture([np.zeros((100, 200, 3), np.uint8)], count=10)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda source: capture)
    with VideoReader(mp4, clock="frame_fps") as reader:
        next(reader)
        with pytest.raises(RuntimeError, match="Premature"): next(reader)
    assert capture.released


def test_pipeline_mock_roundtrip(mp4, tmp_path):
    fixture = [[{"bbox": [10.5, 30, 100, 80], "confidence": .15, "class_id": 4}]] * 4
    result = VideoPipeline(MockDetector(fixture), output_dir=tmp_path / "out", save_video=True,
                           save_json=True).process(mp4)
    assert result["status"] == "complete" and result["frames_processed"] == 4
    assert result["mock"] and result["detection_count"] == 4 and result["termination"] == "eof"
    rows = [json.loads(line) for line in (tmp_path / "out/detections.jsonl").read_text().splitlines()]
    assert rows[0]["detections"][0]["bbox"][0] == 10.5
    assert rows[-1]["timestamp_ms"] == 600 and all(row["mock"] for row in rows)
    with VideoReader(tmp_path / "out/annotated.mp4") as reader: decoded = list(reader)
    assert len(decoded) == 4 and decoded[0].frame.shape == (100, 200, 3)
    assert verify_detection_run(tmp_path / "out")["decoded_video_frames"] == 4


def test_pipeline_failure_preserves_partial_output_and_releases(mp4, tmp_path, monkeypatch):
    import traffic_mot.video.pipeline as module
    readers = []
    real_reader = VideoReader
    def factory(*a, **k):
        reader = real_reader(*a, **k); readers.append(reader); return reader
    monkeypatch.setattr(module, "VideoReader", factory)
    class FailingDetector(MockDetector):
        def detect(self, frame):
            if self.index == 1: raise RuntimeError("inference test failure")
            return super().detect(frame)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="inference test failure"):
        VideoPipeline(FailingDetector(), output_dir=out, save_json=True, save_video=True).process(mp4)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["frames_processed"] == 1
    assert len((out / "detections.jsonl").read_text().splitlines()) == 1
    assert readers[0].closed
    capture = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert capture.read()[0]; capture.release()


def test_writer_refuses_overwrite_and_vfr(tmp_path):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, 25, (200, 100))
    try:
        writer.write(np.zeros((100, 200, 3), np.uint8), 0)
        with pytest.raises(ValueError, match="VFR"): writer.write(np.zeros((100, 200, 3), np.uint8), 100)
    finally: writer.close()
    with pytest.raises(FileExistsError): VideoWriter(path, 25, (200, 100))
    with pytest.raises(ValueError, match="even"): VideoWriter(tmp_path / "odd.mp4", 25, (201, 100))


def test_cli_mock_and_max_frame_limit_from_foreign_cwd(mp4, tmp_path):
    fixture = tmp_path / "fixture.json"; fixture.write_text("[[],[],[],[]]")
    out = tmp_path / "cli"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_video.py"), "--source", str(mp4),
        "--detector", "mock", "--mock-fixture", str(fixture), "--max-frames", "2", "--save-video", "--save-json",
        "--output-dir", str(out)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["frames_processed"] == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["termination"] == "frame_limit" and manifest["source_frame_count"] == 4


def test_cli_requires_explicit_mock_fixture(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_video.py"), "--source", "unused.mp4",
                             "--detector", "mock"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "--mock-fixture is required" in result.stderr
