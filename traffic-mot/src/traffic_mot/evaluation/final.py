"""Frozen-validation selection verification and held-out test execution."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from traffic_mot.detection.bridge import file_sha256, preflight
from traffic_mot.evaluation.trackeval_runner import TRACKEVAL_COMMIT
from traffic_mot.occlusion.recovery import OAR_VERSION, RecoveryTracker, validate_policy
from traffic_mot.tracking.backend_bridge import UltralyticsTracker
from traffic_mot.tracking.baselines import _run_tracking_experiments, _save, _sync
from traffic_mot.utils.config import require
from traffic_mot.utils.seed import seed_everything
from traffic_mot.video.mot_sequence import MOTSequence


def resolve_frozen_config(output_root: Path, ablation_run: Path | None = None) -> Path:
    """Resolve the preceding run deterministically; never promote a smoke candidate.

    Pass ABLATION_RUN from notebook03 when available. With no explicit reference,
    a unique frozen selection is required rather than silently choosing a run.
    """
    if ablation_run is not None:
        ablation_run = Path(ablation_run)
        path = ablation_run if ablation_run.suffix == ".yaml" else ablation_run / "selected_config.yaml"
        require(path.is_file(), f"No frozen selection in {ablation_run}. Finish notebook03 with FULL_VALIDATION=True; a smoke baseline reference must also be replaced by a full baseline.")
        return path.resolve()
    candidates = list(output_root.glob("ablation/selected_config.yaml")) + list(output_root.glob("ablation_*/selected_config.yaml"))
    require(len(candidates) == 1, f"Expected one frozen selection under {output_root}, found {len(candidates)}. Set ABLATION_RUN to the completed full-validation directory; smoke candidate_config.yaml is not a final configuration.")
    return candidates[0].resolve()


def verify_frozen_selection(project_root: Path, frozen_path: Path, detector_path: Path,
                            manifest_path: Path) -> tuple[dict, dict]:
    """Verify actual full-validation artifacts BEFORE inspecting test GT/model load."""
    require(frozen_path.is_file(), f"Frozen config not found: {frozen_path}")
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    require(isinstance(frozen, dict) and frozen.get("schema_version") == "1.0.0", "Unsupported frozen configuration")
    require(frozen.get("frozen") is True and frozen.get("smoke_test") is False
            and frozen.get("selection_split") == "validation" and frozen.get("test_evaluated") is False,
            "Final test requires a frozen full-validation selection; candidate/smoke or test-selected configs are rejected")
    folder = frozen_path.parent
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    require(selection["frozen"] is True and selection["config"] == frozen_path.name
            and selection["sha256"] == file_sha256(frozen_path), "Frozen selection checksum/metadata mismatch")
    run_path = folder / "manifest.json"
    require(file_sha256(run_path) == frozen["source_run_manifest_sha256"], "Validation source-run manifest changed")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    require(run["status"] == "complete" and run["split"] == "validation" and run["smoke_test"] is False
            and run["mock"] is False and run.get("study") == "occlusion_ablation", "Selection must come from a completed real full-validation ablation")
    require(file_sha256(manifest_path) == frozen["source_manifest_sha256"] == run["signature"]["manifest_sha256"],
            "Dataset/split manifest differs from validation freeze")
    # Only metadata/counts are inspected here. Test GT isn't opened until after
    # this entire gate succeeds and the execution engine checks the selected split.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_counts = {s["name"]: s["frame_count"] for s in manifest["validation"]}
    require(run["frame_counts"] == validation_counts, "Frozen source run did not evaluate every declared validation frame")
    experiment = frozen["selected_experiment"]
    require(experiment in {"E0", "E1", "E2", "E3"} and selection["selected_experiment"] == experiment,
            "Invalid selected experiment")
    candidates = ["E0", "E1", "E2", "E3"]
    expected = max(candidates, key=lambda name: (run["summary"][name]["HOTA"], run["summary"][name]["IDF1"],
                                                -run["summary"][name]["IDSW"], -candidates.index(name)))
    require(experiment == expected, "Frozen experiment does not match the declared validation selection rule")
    for key in ["HOTA", "IDF1", "MOTA", "IDSW"]:
        require(frozen["validation_score"][key] == run["summary"][experiment][key], "Frozen validation score changed")
    tracker = frozen["tracker_profile"]
    snapshot = json.loads((folder / experiment / "tracker_config.json").read_text(encoding="utf-8"))
    require(tracker == snapshot["config"], "Frozen tracker profile differs from validated experiment")
    backend = "bytetrack" if experiment in {"E0", "E2"} else "botsort"
    require(tracker["backend_version"] == "8.4.171" and tracker["tracker"]["tracker_type"] == backend
            and not tracker["tracker"].get("with_reid", False) and frozen["with_reid"] is False,
            "Frozen tracker backend/ReID configuration mismatch")
    recovery = frozen["recovery_policy"]
    require((recovery is not None) is (experiment in {"E2", "E3"}), "Frozen OAR flag differs from selected experiment")
    if recovery is not None:
        validate_policy(recovery, tracker["tracker"]["track_buffer"])
        require(snapshot.get("oar_policy") == recovery and frozen["oar_version"] == OAR_VERSION, "OAR policy/version differs from validation")
    evaluation = json.loads((folder / "evaluation_profile.json").read_text(encoding="utf-8"))
    require(frozen["evaluation_profile"] == evaluation and evaluation["backend_commit"] == TRACKEVAL_COMMIT,
            "Frozen evaluation profile differs from validation")
    detector = yaml.safe_load(detector_path.read_text(encoding="utf-8"))
    # JSON snapshots stringify integer class-map keys, whereas YAML may retain
    # them as integers. Compare the same JSON representation without changing IDs.
    canonical_detector = json.loads(json.dumps(detector))
    require(canonical_detector == json.loads(json.dumps(frozen["detector_profile"])),
            "Detector profile differs from validation; use the same resolved CONFIG_PATH")
    require(canonical_detector == json.loads((folder / "detector_profile.json").read_text(encoding="utf-8")),
            "Frozen detector profile differs from validation snapshot")
    require(file_sha256(project_root / "src/traffic_mot/detection/rtdetrv2_p2.py") == run["signature"]["detector_adapter_sha256"],
            "Detector adapter changed since validation freeze")
    # Original selection artifacts contain implementation checksums. The shared
    # runner gains final-test orchestration, but tracker decision code must match.
    critical = ["tracking/backend_bridge.py"]
    if recovery is not None: critical += ["occlusion/evidence.py", "occlusion/recovery.py"]
    for name in critical:
        require(run["implementation_sha256"][name] == file_sha256(project_root / "src/traffic_mot" / name),
                f"Tracker implementation changed since validation: {name}; revalidate before final test")
    _, provenance = preflight(detector, project_root)
    require(provenance["checkpoint_sha256"] == run["signature"]["checkpoint_sha256"]
            and provenance["model_config_sha256"] == run["signature"]["model_config_sha256"], "Model assets differ from validation")
    return frozen, {"status": "passed", "frozen_config_sha256": file_sha256(frozen_path),
                    "validation_source_manifest_sha256": file_sha256(run_path),
                    "source_manifest_sha256": file_sha256(manifest_path), "selected_experiment": experiment,
                    "critical_implementation_checked": critical, "test_gt_read": False}


def benchmark_selected(project_root: Path, detector_path: Path, tracker_path: Path, recovery: dict | None,
                       manifest_path: Path, frame_limits: dict[str, int], *, measured_frames: int, warmup_frames: int,
                       output: Path, split: str = "test") -> dict:
    """Measure real decode+detector+tracking with warmup; no cache or GT input.

    Detector time includes preprocessing, forward, postprocessing and device sync.
    This measures the analytics core, excluding render/encode/export, not full
    video-output throughput or camera source FPS.
    """
    require(measured_frames > 0 and warmup_frames >= 0, "Invalid benchmark frame counts")
    require(split in {"validation", "test"}, "Unknown benchmark split")
    from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
    seed_everything(42, use_torch=True)
    detector = RTDETRv2P2Detector(detector_path, project_root=project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(detector.is_mock is False, "Benchmark requires a real detector")
    sequences = [s for s in manifest[split] if s["name"] in frame_limits]
    require(set(frame_limits) == {s["name"] for s in sequences}
            and all(0 < frame_limits[s["name"]] <= s["frame_count"] for s in sequences), "Benchmark frame limits outside split")
    rows, warmed, decode_ms, detect_ms, track_ms, total_ms = [], 0, [], [], [], []
    tracker = None
    for sequence in sequences:
        tracker = RecoveryTracker(tracker_path, recovery) if recovery is not None else UltralyticsTracker(tracker_path)
        with MOTSequence(sequence, manifest_path.parent) as reader:
            for index in range(frame_limits[sequence["name"]]):
                _sync(detector.device)
                start = time.perf_counter()
                frame = reader.read(index)
                decoded = time.perf_counter()
                batch = detector.detect(frame)
                _sync(detector.device)
                detected = time.perf_counter()
                tracker.update(batch, frame, timestamp=None)
                end = time.perf_counter()
                if warmed < warmup_frames:
                    warmed += 1
                    continue
                parts = [(decoded - start) * 1000, (detected - decoded) * 1000,
                         (end - detected) * 1000, (end - start) * 1000]
                for values, value in zip([decode_ms, detect_ms, track_ms, total_ms], parts): values.append(value)
                rows.append({"sequence": sequence["name"], "frame_id": index + 1,
                             **dict(zip(["decode_ms", "detector_ms", "tracking_including_oar_ms", "core_total_ms"], parts))})
                if len(rows) >= measured_frames: break
        if len(rows) >= measured_frames: break
    require(bool(rows), "No measured benchmark frames after warmup; reduce warmup or use a longer test sequence")
    def summary(values):
        data = np.asarray(values)
        return {"mean_ms": float(data.mean()), "p50_ms": float(np.percentile(data, 50)),
                "p95_ms": float(np.percentile(data, 95)), "p99_ms": float(np.percentile(data, 99))}
    report = {"status": "complete", "mock": False, "cache_used": False, "split": split,
              "scope": "decode + detector(preprocess/forward/postprocess) + tracking/OAR; render/encode/export excluded",
              "warmup_frames": warmed, "measured_frames": len(rows), "requested_measured_frames": measured_frames,
              "device": str(detector.device), "precision": detector.profile["precision"], "batch_size": 1,
              "timings": {name: summary(values) for name, values in
                          [("decode", decode_ms), ("detector", detect_ms), ("tracking_including_oar", track_ms), ("core_total", total_ms)]},
              "measured_core_fps": len(rows) / (sum(total_ms) / 1000), "camera_fps": None,
              "detector_metadata": detector.metadata(), "sequence_boundary_reset": True,
              "warmup_policy": "once before measurement; sequence state resets counted when encountered (cold sequence state included)"}
    with (output / "benchmark_timings.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _save(output / "benchmark.json", report)
    return report


def run_final_evaluation(project_root: Path, frozen_path: Path, detector_path: Path, manifest_path: Path,
                         output: Path, *, split: str = "test", sequence_names: list[str] | None = None,
                         max_frames: int | None = None, save_video: bool = False,
                         visualization_fps: float | None = None, benchmark_frames: int = 100) -> dict:
    """Run one frozen configuration on test, without selection/threshold search."""
    require(split == "test", "Final evaluation supports held-out test only")
    require(not output.exists(), f"Refusing to overwrite final output: {output}")
    require(benchmark_frames >= 0, "benchmark_frames must be nonnegative")
    frozen, verification = verify_frozen_selection(project_root, frozen_path, detector_path, manifest_path)
    # Materialize frozen profiles in a small sibling directory. The shared engine
    # still owns output creation/failure manifests, preserving its cleanup logic.
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="final-profiles-", dir=output.parent) as temp:
        config_dir = Path(temp)
        tracker_path, eval_path = config_dir / "tracker.yaml", config_dir / "evaluation.yaml"
        tracker_path.write_text(yaml.safe_dump(frozen["tracker_profile"], sort_keys=False), encoding="utf-8")
        eval_path.write_text(yaml.safe_dump(frozen["evaluation_profile"], sort_keys=False), encoding="utf-8")
        experiment = frozen["selected_experiment"]
        run = _run_tracking_experiments(project_root, detector_path, manifest_path, output, split="test",
            sequence_names=sequence_names, max_frames=max_frames, evaluation_config=eval_path,
            experiment_profiles={experiment: {"tracker_config": tracker_path, "recovery": frozen["recovery_policy"]}},
            include_diagnostics=True, save_video=save_video, visualization_fps=visualization_fps)
        try:
            _save(output / "freeze_verification.json", verification)
            (output / "selected_config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
            benchmark = None
            if benchmark_frames:
                benchmark = benchmark_selected(project_root, detector_path, tracker_path, frozen["recovery_policy"],
                    manifest_path, run["frame_counts"], measured_frames=benchmark_frames,
                    warmup_frames=frozen["evaluation_profile"]["benchmark"]["warmup_frames"], output=output)
            final = {"status": "complete", "split": "test", "smoke_test": run["smoke_test"],
                     "selected_experiment": experiment, "frozen_config_sha256": file_sha256(frozen_path),
                     "metrics": run["summary"][experiment], "frame_counts": run["frame_counts"],
                     "test_parameters_optimized": False, "validation_score": frozen["validation_score"],
                     "benchmark": benchmark, "freeze_verification": verification}
            _save(output / "final_summary.json", final)
            run.update(study="frozen_final_test", frozen_config_sha256=file_sha256(frozen_path),
                       selected_experiment=experiment, test_parameters_optimized=False,
                       benchmark_executed=benchmark is not None)
            _save(output / "manifest.json", run)
        except BaseException as exc:
            run.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
            _save(output / "manifest.json", run)
            raise
    return final
