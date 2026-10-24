"""Reproducible E0/E1 runs from a shared immutable detector cache."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import logging
from pathlib import Path
import platform
import subprocess
import time
from uuid import uuid4

import cv2
import numpy as np
import yaml

from traffic_mot.detection.base import VISDRONE_CLASSES
from traffic_mot.detection.bridge import file_sha256, preflight
from traffic_mot.evaluation.trackeval_runner import TRACKEVAL_COMMIT, evaluate_sequences, load_metrics, parse_gt, write_metrics
from traffic_mot.tracking.backend_bridge import check_backend_dependencies
from traffic_mot.tracking.bytetrack import ByteTrackTracker
from traffic_mot.tracking.botsort import BoTSORTTracker
from traffic_mot.utils.config import require
from traffic_mot.utils.kaggle import check_mot_dataset
from traffic_mot.utils.seed import seed_everything
from traffic_mot.utils.types import DetectionBatch
from traffic_mot.video.mot_sequence import MOTSequence, read_gt
from traffic_mot.video.writer import VideoWriter

LOGGER = logging.getLogger(__name__)


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _pixel_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _sync(device) -> None:
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def _batch(row: dict) -> DetectionBatch:
    detections = row["detections"]
    require(all(d["class_id"] in range(3, 11) and d["class_name"] == VISDRONE_CLASSES[d["class_id"]]
                for d in detections), "Cache class mapping invalid")
    return DetectionBatch(np.asarray([d["bbox"] for d in detections], np.float32).reshape(-1, 4),
                          np.asarray([d["confidence"] for d in detections], np.float32),
                          np.asarray([d["class_id"] for d in detections], np.int64),
                          np.asarray([d["detection_index"] for d in detections], np.int64))


def _build_cache(root: Path, signature: dict, selected: list[dict], base: Path,
                 config: Path, project_root: Path) -> dict:
    from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
    detector = RTDETRv2P2Detector(config, project_root=project_root)
    root.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "incomplete", "schema_version": "1.0.0", "signature": signature,
                "mock": False, "sequences": {}, "detector_seconds": 0.0, "decode_seconds": 0.0}
    try:
        for seq in selected:
            name, count = seq["name"], signature["frames"][seq["name"]]
            path = root / f"{name}.jsonl"
            with MOTSequence(seq, base) as reader, path.open("x", encoding="utf-8") as stream:
                for index in range(count):
                    start = time.perf_counter()
                    frame = reader.read(index)
                    metadata["decode_seconds"] += time.perf_counter() - start
                    _sync(detector.device)
                    start = time.perf_counter()
                    batch = detector.detect(frame)
                    _sync(detector.device)
                    metadata["detector_seconds"] += time.perf_counter() - start
                    stream.write(json.dumps({"frame_id": index + 1, "width": frame.shape[1], "height": frame.shape[0],
                        "pixel_sha256": _pixel_hash(frame), "detections": batch.records(detector.class_names)}, allow_nan=False) + "\n")
                    if (index + 1) % 30 == 0 or index == 0:
                        LOGGER.info("Real detector %s frame %d/%d", name, index + 1, count)
            metadata["sequences"][name] = {"file": path.name, "sha256": file_sha256(path), "frames": count}
        metadata.update(status="complete", detector=detector.metadata())
    finally:
        _save(root / "manifest.json", metadata)
    return metadata


def _validate_cache(root: Path, signature: dict) -> dict:
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(metadata["status"] == "complete" and metadata["mock"] is False, "Detection cache incomplete or mock")
    require(metadata["signature"] == signature, "Detection cache signature mismatch (model/config/data/frame range)")
    require(set(metadata["sequences"]) == set(signature["frames"]), "Cache sequence list mismatch")
    for name, info in metadata["sequences"].items():
        require(Path(info["file"]).name == info["file"], "Cache filenames must be relative basenames")
        require(file_sha256(root / info["file"]) == info["sha256"] and info["frames"] == signature["frames"][name],
                f"Detection cache changed: {name}")
    return metadata


def _draw(frame: np.ndarray, tracks: list) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (round(v) for v in track.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(output, f"#{track.track_id} {VISDRONE_CLASSES[track.class_id]} {track.confidence:.2f}",
                    (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
    return output


def _run_tracking_experiments(project_root: Path, detector_config: Path, manifest_path: Path, output: Path, *,
                  split: str = "validation", sequence_names: list[str] | None = None,
                  max_frames: int | None = None, cache_root: Path | None = None,
                  tracker_configs: dict[str, Path] | None = None,
                  evaluation_config: Path | None = None,
                  save_video: bool = False, visualization_fps: float | None = None,
                  experiment_profiles: dict[str, dict] | None = None,
                  include_diagnostics: bool = False) -> dict:
    """Shared execution engine; public tuning/final wrappers enforce split gates."""
    require(split in {"validation", "test"}, "Unknown evaluation split")
    require(max_frames is None or max_frames > 0, "max_frames must be positive")
    require(not save_video or (visualization_fps is not None and np.isfinite(visualization_fps) and visualization_fps > 0),
            "Native images have no known FPS: --save-video requires explicit --visualization-fps (display only)")
    require(not output.exists(), f"Refusing to overwrite run directory: {output}")
    check_backend_dependencies()
    _, evaluator_info = load_metrics()  # Fail before expensive detector inference.
    eval_path = evaluation_config or project_root / "configs/evaluation.yaml"
    eval_profile = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
    require(eval_profile["backend"] == "trackeval" and eval_profile["backend_commit"] == TRACKEVAL_COMMIT,
            "Evaluation profile must select the pinned TrackEval backend")
    require(eval_profile["diagnostics"]["iou_threshold"] == .5, "This version implements IoU0.5 CLEAR/Identity and ignore protection")
    class_ids = eval_profile["dataset"]["evaluated_class_ids"]
    require(class_ids and len(set(class_ids)) == len(class_ids) and set(class_ids) <= set(range(3, 11)),
            "Evaluation classes must be unique original vehicle IDs")
    profile = yaml.safe_load(detector_config.read_text(encoding="utf-8"))
    _, provenance = preflight(profile, project_root)
    require(profile["batch_size"] == 1, "Baseline cache uses batch_size=1 for reproducible frame inference")
    configs = tracker_configs or {"E0": project_root / "configs/bytetrack.yaml", "E1": project_root / "configs/botsort.yaml"}
    require(set(configs) == {"E0", "E1"}, "Both E0 and E1 configurations are required")
    definitions = experiment_profiles or {name: {"tracker_config": config, "recovery": None} for name, config in configs.items()}
    require(bool(definitions) and all(Path(name).name == name and name.isidentifier() for name in definitions), "Unsafe experiment names")
    for definition in definitions.values():
        config = Path(definition["tracker_config"])
        conf = yaml.safe_load(config.read_text(encoding="utf-8"))
        require(profile["confidence_threshold"] <= conf["tracker"]["track_low_thresh"], "Detector cutoff removes low-confidence association band")
        if definition.get("recovery") is not None:
            from traffic_mot.occlusion.recovery import validate_policy
            validate_policy(definition["recovery"], conf["tracker"]["track_buffer"])
    report = check_mot_dataset(manifest_path, split=split)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest[split]
    if sequence_names:
        require(len(set(sequence_names)) == len(sequence_names), "Duplicate --sequence argument")
        require(set(sequence_names) <= {s["name"] for s in selected}, "Requested sequence is outside the requested split")
        selected = [s for s in selected if s["name"] in sequence_names]
    require(bool(selected), "No selected MOT sequences")
    require(all(s.get("source_kind") in {"images_zip", "images_dir"} and Path(s["name"]).name == s["name"]
                for s in selected), "Use a native manifest from prepare_mot_dataset.py with safe sequence names")
    frames = {s["name"]: min(max_frames or s["frame_count"], s["frame_count"]) for s in selected}
    # Paths are included conservatively: don't silently reuse caches from another mount/config.
    signature = {"detector_config_sha256": file_sha256(detector_config),
                 "checkpoint_sha256": provenance["checkpoint_sha256"], "model_config_sha256": provenance["model_config_sha256"],
                 "detector_adapter_sha256": file_sha256(project_root / "src/traffic_mot/detection/rtdetrv2_p2.py"),
                 "manifest_sha256": file_sha256(manifest_path), "frames": frames, "seed": 42}
    seed_everything(42, use_torch=True)
    output.mkdir(parents=True)
    smoke = len(selected) != len(manifest[split]) or any(frames[s["name"]] != s["frame_count"] for s in selected)
    run = {"schema_version": "1.0.0", "status": "incomplete", "run_id": str(uuid4()),
           "started_at": datetime.now(timezone.utc).isoformat(), "split": split, "smoke_test": smoke,
           "frame_counts": frames, "mock": False, "with_reid": False,
           "recovery_enabled": any(d.get("recovery") is not None for d in definitions.values()),
           "source_manifest": str(manifest_path), "signature": signature, "evaluator": evaluator_info,
           "preflight": report, "platform": platform.platform(), "visualization_fps": visualization_fps,
           "environment": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()),
           "benchmark": False, "timing_scope": "cold cache creation plus shared replay; not per-baseline end-to-end FPS"}
    for args, key in ((["rev-parse", "HEAD"], "git_commit"), (["status", "--porcelain"], "git_status")):
        run[key] = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True).stdout.strip()
    cache = cache_root or output / "detection_cache"
    start = time.perf_counter()
    try:
        if cache_root is None:
            _build_cache(cache, signature, selected, manifest_path.parent, detector_config, project_root)
        cache_metadata = _validate_cache(cache, signature)
        run.update(detection_cache=str(cache.resolve()), detection_cache_manifest_sha256=file_sha256(cache / "manifest.json"),
                   detector=cache_metadata["detector"], cache_reused=cache_root is not None)
        _save(output / "detector_profile.json", profile)
        _save(output / "evaluation_profile.json", eval_profile)
        run["evaluation_config_sha256"] = file_sha256(eval_path)
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        predictions: dict[str, dict] = {name: {} for name in definitions}
        timing = {name: [] for name in definitions}
        tracker_diagnostics = {name: {} for name in definitions}
        for seq in selected:
            name = seq["name"]
            trackers = {}
            for experiment, definition in definitions.items():
                config = Path(definition["tracker_config"])
                if definition.get("recovery") is None:
                    from traffic_mot.tracking.backend_bridge import UltralyticsTracker
                    trackers[experiment] = UltralyticsTracker(config)
                else:
                    from traffic_mot.occlusion.recovery import RecoveryTracker
                    trackers[experiment] = RecoveryTracker(config, definition["recovery"])
            handles, mot_handles, writers, decision_handles = {}, {}, {}, {}
            try:
                for experiment, tracker in trackers.items():
                    folder = output / experiment
                    (folder / "mot").mkdir(parents=True, exist_ok=True)
                    _save(folder / "tracker_config.json", tracker.metadata())
                    handles[experiment] = (folder / f"{name}.jsonl").open("x", encoding="utf-8")
                    mot_handles[experiment] = (folder / "mot" / f"{name}.txt").open("x", encoding="utf-8")
                    predictions[experiment][name] = []
                    if definitions[experiment].get("recovery") is not None:
                        decision_handles[experiment] = (folder / f"{name}_recovery.jsonl").open("x", encoding="utf-8")
                cache_file = cache / cache_metadata["sequences"][name]["file"]
                with MOTSequence(seq, manifest_path.parent) as reader, cache_file.open(encoding="utf-8") as stream:
                    for index in range(frames[name]):
                        line = stream.readline()
                        require(bool(line), "Detection cache ended early")
                        record = json.loads(line)
                        frame = reader.read(index)
                        require(record["frame_id"] == index + 1 and record["pixel_sha256"] == _pixel_hash(frame),
                                "Detection cache frame/source mismatch")
                        batch = _batch(record)
                        for experiment, tracker in trackers.items():
                            before = time.perf_counter()
                            observed = tracker.update(batch, frame, timestamp=None)
                            timing[experiment].append((time.perf_counter() - before) * 1000)
                            if experiment in decision_handles:
                                decision_handles[experiment].write(json.dumps({"frame_id": index + 1,
                                    "decisions": tracker.backend.last_decisions}, allow_nan=False) + "\n")
                            tracks = [{**asdict(t), "class_name": VISDRONE_CLASSES[t.class_id]} for t in observed]
                            predictions[experiment][name].append(tracks)
                            payload = {"schema_version": "1.0.0", "sequence": name, "camera_id": seq.get("source_video_id", name),
                                       "session_id": f"{run['run_id']}:{name}", "frame_id": index + 1,
                                       "timestamp_ms": None, "time_basis": "frame_index", "tracks": tracks}
                            handles[experiment].write(json.dumps(payload, allow_nan=False) + "\n")
                            for t in observed:
                                x1, y1, x2, y2 = t.bbox
                                mot_handles[experiment].write(f"{index+1},{t.track_id},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{t.confidence:.8f},-1,-1,-1\n")
                            if index == 0 or save_video:
                                rendered = _draw(frame, observed)
                                if index == 0:
                                    require(cv2.imwrite(str(output / experiment / f"{name}_preview.jpg"), rendered), "Cannot write preview")
                                if save_video:
                                    # Explicit display-only padding; inference/evaluation coordinates stay original.
                                    h, w = frame.shape[:2]
                                    rendered = cv2.copyMakeBorder(rendered, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
                                    if experiment not in writers:
                                        writers[experiment] = VideoWriter(output / experiment / f"{name}.mp4", visualization_fps,
                                                                         (w + w % 2, h + h % 2))
                                    writers[experiment].write(rendered, round(index * 1000 / visualization_fps))
                        if (index + 1) % 100 == 0: LOGGER.info("Track replay %s %d/%d", name, index + 1, frames[name])
                    require(not stream.readline(), "Detection cache has extra frames")
            finally:
                for experiment, tracker in trackers.items():
                    if hasattr(tracker, "diagnostics"): tracker_diagnostics[experiment][name] = tracker.diagnostics()
                for handle in [*handles.values(), *mot_handles.values(), *decision_handles.values()]: handle.close()
                for writer in writers.values(): writer.close()
        gt = {seq["name"]: parse_gt(read_gt(seq, manifest_path.parent), frames[seq["name"]]) for seq in selected}
        summary = {}
        for experiment, data in predictions.items():
            result = evaluate_sequences({name: (gt[name], pred) for name, pred in data.items()}, class_ids)
            result.update(split=split, smoke_test=smoke, source_manifest_sha256=signature["manifest_sha256"],
                          sequence_frame_counts=frames, mock=False,
                          recovery_enabled=definitions[experiment].get("recovery") is not None, with_reid=False)
            write_metrics(output / experiment / "metrics.json", result)
            if include_diagnostics:
                from traffic_mot.evaluation.occlusion import evaluate_diagnostics
                diagnostics = evaluate_diagnostics({name: (gt[name], pred) for name, pred in data.items()}, class_ids,
                    recovery_window=eval_profile["diagnostics"]["recovery_window_frames"],
                    stable_frames=eval_profile["diagnostics"]["recovery_stable_frames"])
                _save(output / experiment / "diagnostics.json", diagnostics)
            _save(output / experiment / "recovery_statistics.json", tracker_diagnostics[experiment])
            times = np.asarray(timing[experiment])
            summary[experiment] = {**result["combined"], "observed_track_records": sum(len(row) for v in data.values() for row in v),
                "tracking_update_ms_mean": float(times.mean()), "tracking_update_ms_p95": float(np.percentile(times, 95)),
                "prediction_files_sha256": {name: file_sha256(output / experiment / f"{name}.jsonl") for name in data}}
            summary[experiment]["oar_seconds"] = sum(d["oar_seconds"] for d in tracker_diagnostics[experiment].values())
        run.update(status="complete", summary=summary)
        _save(output / "summary.json", {"smoke_test": smoke, "split": split, "metric_scale": "ratios 0..1; counts raw",
                                       "benchmark": False, "results": summary})
        import csv
        with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = ["experiment", "smoke_test", "HOTA", "IDF1", "MOTA", "IDSW", "MT", "ML", "CLR_TP", "CLR_FP", "CLR_FN"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, values in summary.items():
                writer.writerow({"experiment": name, "smoke_test": smoke, **{k: values[k] for k in fields[2:]}})
    except BaseException as exc:
        run.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        run.update(elapsed_seconds=time.perf_counter() - start, ended_at=datetime.now(timezone.utc).isoformat())
        _save(output / "manifest.json", run)
    return run


def run_baselines(project_root: Path, detector_config: Path, manifest_path: Path, output: Path, *,
                  split: str = "validation", **kwargs) -> dict:
    """Public baseline/ablation entrypoint: tuning never evaluates held-out test."""
    require(split == "validation", "Baseline runner uses validation only; test is reserved for frozen final evaluation")
    return _run_tracking_experiments(project_root, detector_config, manifest_path, output, split=split, **kwargs)
