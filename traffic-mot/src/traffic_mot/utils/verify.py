"""Validate saved detection metadata/video without claiming model accuracy."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import cv2

from traffic_mot.detection.base import VISDRONE_CLASSES, VEHICLE_IDS
from traffic_mot.detection.bridge import file_sha256
from traffic_mot.utils.config import require


def verify_detection_run(run_dir: Path, *, save_preview: bool = True) -> dict:
    """Read all JSONL/video frames, validate geometry/classes/scores/counts.

    Produces verification.json for smoke acceptance; no AP/MOT metrics or
    steady-state FPS are derived from these artifacts.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Cannot accept an incomplete run")
    require(manifest["save_json"], "JSONL is required to verify detection contracts")
    require(manifest["mock"] or manifest["detector"]["inference_run"], "No real inference was recorded")
    classes, sessions, scores = Counter(), {}, []
    row_count = 0
    threshold = manifest["detector"]["confidence_threshold"]
    with (run_dir / "detections.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            previous = sessions.get(row["session_id"], {"frame_id": 0, "timestamp_ms": 0})
            require(row["frame_id"] == previous["frame_id"] + 1, "Frame sequence has gaps or duplicates")
            require(row["timestamp_ms"] >= previous["timestamp_ms"], "Timestamp moved backwards")
            require(row["mock"] == manifest["mock"], "Mock flag mismatch")
            require(row["camera_id"] == manifest["camera_id"] and row["video_source"] == manifest["video_source"], "Source mismatch")
            indices = set()
            for detection in row["detections"]:
                cls, score, box = detection["class_id"], detection["confidence"], detection["bbox"]
                require(cls in VEHICLE_IDS and detection["class_name"] == VISDRONE_CLASSES[cls], "Class mapping mismatch")
                require(math.isfinite(score) and threshold - 1e-7 <= score <= 1, "Invalid confidence")
                require(len(box) == 4 and all(math.isfinite(x) for x in box) and
                        0 <= box[0] < box[2] <= row["width"] and 0 <= box[1] < box[3] <= row["height"], "Invalid original-coordinate box")
                require(detection["detection_index"] not in indices, "Duplicate detection index")
                indices.add(detection["detection_index"])
                classes[detection["class_name"]] += 1
                scores.append(score)
            sessions[row["session_id"]] = row
            row_count += 1
    require(row_count == manifest["frames_processed"] and sum(classes.values()) == manifest["detection_count"], "Manifest count mismatch")
    decoded, videos = 0, []
    if manifest["save_video"]:
        require(len(manifest["video_files"]) == len(sessions), "Video/session count mismatch")
    for name, session in zip(manifest["video_files"], sessions.values()):
        path = run_dir / name
        capture = cv2.VideoCapture(str(path))
        count = 0
        try:
            require(capture.isOpened(), f"Cannot reopen video {name}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = capture.read()
                if not ok: break
                if save_preview and decoded == 0 and count == 0:
                    require(cv2.imwrite(str(run_dir / "preview.jpg"), frame), "Cannot save preview")
                require(frame.shape[:2] == (session["height"], session["width"]), "Output dimensions changed")
                count += 1
        finally:
            capture.release()
        require(count == session["frame_id"], "Decoded session frame count mismatch")
        videos.append({"file": name, "decoded_frames": count, "encoded_fps": fps, "sha256": file_sha256(path)})
        decoded += count
    if manifest["save_video"]: require(decoded == row_count, "Decoded video frame count mismatch")
    result = {"status": "passed", "mock": manifest["mock"], "frames": row_count,
              "detections": sum(classes.values()), "class_counts": dict(classes),
              "confidence_range": [min(scores), max(scores)] if scores else None,
              "decoded_video_frames": decoded, "videos": videos,
              "source_sha256": manifest.get("source_sha256"), "accuracy_evaluated": False,
              "performance_benchmark": False}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
