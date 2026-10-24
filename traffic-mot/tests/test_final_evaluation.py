"""Freeze provenance, full-validation gating and final CLI/benchmark plumbing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from traffic_mot.detection.bridge import file_sha256
from traffic_mot.evaluation.final import resolve_frozen_config, run_final_evaluation, verify_frozen_selection
from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.types import DetectionBatch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def selection(tmp_path, monkeypatch):
    """Test-only artificial artifacts; never reported as validation/test results."""
    import traffic_mot.evaluation.final as module
    folder = tmp_path / "ablation_abc"; folder.mkdir()
    dataset = {"dataset": "VisDrone-MOT", "validation": [{"name": "val", "frame_count": 2}],
               "test": [{"name": "test", "frame_count": 3}]}
    manifest = tmp_path / "mot.json"; manifest.write_text(json.dumps(dataset))
    detector_profile = yaml.safe_load((ROOT / "configs/detector.yaml").read_text())
    detector = tmp_path / "detector.yaml"; detector.write_text(yaml.safe_dump(detector_profile))
    tracker = yaml.safe_load((ROOT / "configs/bytetrack.yaml").read_text())
    evaluation = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text())
    for filename, value in [("detector_profile.json", detector_profile), ("evaluation_profile.json", evaluation)]:
        (folder / filename).write_text(json.dumps(value))
    (folder / "E0").mkdir()
    (folder / "E0/tracker_config.json").write_text(json.dumps({"config": tracker, "recovery_enabled": False}))
    score = {"HOTA": .7, "IDF1": .8, "MOTA": .6, "IDSW": 1}
    source = {"status": "complete", "split": "validation", "smoke_test": False, "mock": False,
              "study": "occlusion_ablation", "frame_counts": {"val": 2},
              "signature": {"manifest_sha256": file_sha256(manifest), "checkpoint_sha256": "weights",
                            "model_config_sha256": "architecture", "detector_adapter_sha256": file_sha256(ROOT / "src/traffic_mot/detection/rtdetrv2_p2.py")},
              "implementation_sha256": {"tracking/backend_bridge.py": file_sha256(ROOT / "src/traffic_mot/tracking/backend_bridge.py")},
              "summary": {name: dict(score) for name in ["E0", "E1", "E2", "E3"]}}
    (folder / "manifest.json").write_text(json.dumps(source))
    frozen = {"schema_version": "1.0.0", "frozen": True, "smoke_test": False, "selection_split": "validation",
              "test_evaluated": False, "source_run_manifest_sha256": file_sha256(folder / "manifest.json"),
              "source_manifest_sha256": file_sha256(manifest), "selected_experiment": "E0", "validation_score": score,
              "tracker_profile": tracker, "evaluation_profile": evaluation, "detector_profile": detector_profile,
              "recovery_policy": None, "with_reid": False, "oar_version": "motion_overlap_v1"}
    path = folder / "selected_config.yaml"; path.write_text(yaml.safe_dump(frozen))
    (folder / "selection.json").write_text(json.dumps({"frozen": True, "config": path.name,
        "sha256": file_sha256(path), "selected_experiment": "E0"}))
    monkeypatch.setattr(module, "preflight", lambda *a: ({}, {"checkpoint_sha256": "weights", "model_config_sha256": "architecture"}))
    return path, detector, manifest


def refresh_selection(path, *, source=None, frozen=None):
    if source is not None:
        (path.parent / "manifest.json").write_text(json.dumps(source))
    if frozen is None: frozen = yaml.safe_load(path.read_text())
    frozen["source_run_manifest_sha256"] = file_sha256(path.parent / "manifest.json")
    path.write_text(yaml.safe_dump(frozen))
    metadata = json.loads((path.parent / "selection.json").read_text())
    metadata["sha256"] = file_sha256(path)
    (path.parent / "selection.json").write_text(json.dumps(metadata))


def test_unique_and_explicit_resolution(selection):
    path, _, _ = selection
    assert resolve_frozen_config(path.parent.parent) == path.resolve()
    assert resolve_frozen_config(path.parent.parent, path.parent) == path.resolve()
    other = path.parent.parent / "ablation_other"; other.mkdir(); (other / path.name).write_text("another")
    with pytest.raises(ConfigurationError, match="found 2"): resolve_frozen_config(path.parent.parent)
    assert resolve_frozen_config(path.parent.parent, path.parent) == path.resolve()


def test_smoke_candidate_is_not_promoted(tmp_path):
    folder = tmp_path / "ablation_prefix"; folder.mkdir()
    (folder / "candidate_config.yaml").write_text("frozen: false")
    with pytest.raises(ConfigurationError, match="FULL_VALIDATION=True"):
        resolve_frozen_config(tmp_path, folder)


def test_valid_freeze_preflight_does_not_read_test_gt(selection):
    path, detector, manifest = selection
    frozen, report = verify_frozen_selection(ROOT, path, detector, manifest)
    assert report["status"] == "passed" and report["test_gt_read"] is False
    assert frozen["selected_experiment"] == "E0"


def test_json_snapshot_class_map_keys_compare_to_yaml_ids(selection):
    path, detector, manifest = selection
    frozen = yaml.safe_load(path.read_text())
    frozen["detector_profile"] = json.loads(json.dumps(frozen["detector_profile"]))
    refresh_selection(path, frozen=frozen)
    assert verify_frozen_selection(ROOT, path, detector, manifest)[1]["status"] == "passed"


@pytest.mark.parametrize("experiment,backend", [("E2", "bytetrack"), ("E3", "botsort")])
def test_frozen_oar_backend_is_preserved(selection, experiment, backend):
    path, detector, manifest = selection
    source = json.loads((path.parent / "manifest.json").read_text())
    source["summary"][experiment]["HOTA"] = .9
    for name in ["occlusion/evidence.py", "occlusion/recovery.py"]:
        source["implementation_sha256"][name] = file_sha256(ROOT / "src/traffic_mot" / name)
    policy = yaml.safe_load((ROOT / "configs/occlusion.yaml").read_text()); policy["enabled"] = True
    tracker = yaml.safe_load((ROOT / f"configs/{backend}.yaml").read_text())
    (path.parent / experiment).mkdir()
    (path.parent / experiment / "tracker_config.json").write_text(json.dumps({"config": tracker, "oar_policy": policy}))
    frozen = yaml.safe_load(path.read_text())
    frozen.update(selected_experiment=experiment, validation_score=source["summary"][experiment],
                  tracker_profile=tracker, recovery_policy=policy)
    refresh_selection(path, source=source, frozen=frozen)
    metadata = json.loads((path.parent / "selection.json").read_text()); metadata["selected_experiment"] = experiment
    (path.parent / "selection.json").write_text(json.dumps(metadata))
    validated, _ = verify_frozen_selection(ROOT, path, detector, manifest)
    assert validated["recovery_policy"] == policy
    assert validated["tracker_profile"]["tracker"]["tracker_type"] == backend


@pytest.mark.parametrize("mutation,message", [("smoke", "full-validation"), ("checksum", "checksum"),
    ("source", "source-run"), ("counts", "every declared"), ("detector", "Detector profile"),
    ("implementation", "implementation changed"), ("winner", "selection rule")])
def test_freeze_rejects_tampering_or_incomplete_validation(selection, mutation, message):
    path, detector, manifest = selection
    frozen = yaml.safe_load(path.read_text())
    source = json.loads((path.parent / "manifest.json").read_text())
    if mutation == "smoke":
        frozen["frozen"] = False; refresh_selection(path, frozen=frozen)
    elif mutation == "checksum": path.write_text(path.read_text() + "\n# changed")
    elif mutation == "source": (path.parent / "manifest.json").write_text("changed")
    elif mutation == "counts": source["frame_counts"]["val"] = 1; refresh_selection(path, source=source)
    elif mutation == "detector":
        profile = yaml.safe_load(detector.read_text()); profile["confidence_threshold"] = .5
        detector.write_text(yaml.safe_dump(profile))
    elif mutation == "implementation":
        source["implementation_sha256"]["tracking/backend_bridge.py"] = "changed"; refresh_selection(path, source=source)
    elif mutation == "winner":
        source["summary"]["E1"]["HOTA"] = .99; refresh_selection(path, source=source)
    with pytest.raises(ConfigurationError, match=message): verify_frozen_selection(ROOT, path, detector, manifest)


def test_final_uses_only_selected_frozen_configuration(selection, tmp_path, monkeypatch):
    import traffic_mot.evaluation.final as module
    path, detector, manifest = selection
    output = tmp_path / "final"
    called = []
    def engine(*args, **kwargs):
        called.append(kwargs)
        assert kwargs["split"] == "test" and list(kwargs["experiment_profiles"]) == ["E0"]
        assert kwargs["experiment_profiles"]["E0"]["recovery"] is None
        output.mkdir()
        return {"status": "complete", "smoke_test": False, "summary": {"E0": {"HOTA": .5}}, "frame_counts": {"test": 3}}
    monkeypatch.setattr(module, "_run_tracking_experiments", engine)
    report = run_final_evaluation(ROOT, path, detector, manifest, output, benchmark_frames=0)
    assert len(called) == 1 and report["test_parameters_optimized"] is False
    assert report["selected_experiment"] == "E0" and report["benchmark"] is None
    assert json.loads((output / "manifest.json").read_text())["study"] == "frozen_final_test"


def test_failed_final_benchmark_marks_run_incomplete(selection, tmp_path, monkeypatch):
    import traffic_mot.evaluation.final as module
    path, detector, manifest = selection
    output = tmp_path / "final_failed"
    def engine(*args, **kwargs):
        output.mkdir()
        return {"status": "complete", "smoke_test": False, "summary": {"E0": {}}, "frame_counts": {"test": 3}}
    monkeypatch.setattr(module, "_run_tracking_experiments", engine)
    monkeypatch.setattr(module, "benchmark_selected", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("benchmark failed")))
    with pytest.raises(RuntimeError, match="benchmark failed"):
        run_final_evaluation(ROOT, path, detector, manifest, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
    assert not (output / "final_summary.json").exists()


def test_final_cli_has_exact_requested_arguments():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/final_evaluation.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for arg in ["--config", "--detector-config", "--manifest", "--split", "--output-root", "--check-only", "--benchmark-frames"]:
        assert arg in result.stdout


def test_benchmark_is_real_execution_not_cache_throughput(tmp_path, monkeypatch):
    import traffic_mot.evaluation.final as module
    import traffic_mot.detection.rtdetrv2_p2 as detector_module
    state = {"detect": 0, "track": 0, "decode": 0}
    class Detector:
        is_mock = False
        device = SimpleNamespace(type="cpu")
        profile = {"precision": "fp32"}
        def __init__(self, *a, **k): pass
        def detect(self, frame): state["detect"] += 1; return DetectionBatch.empty()
        def metadata(self): return {"fixture": True}
    class Tracker:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): state["track"] += 1
    class Reader:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, index): state["decode"] += 1; return np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(detector_module, "RTDETRv2P2Detector", Detector)
    monkeypatch.setattr(module, "UltralyticsTracker", Tracker)
    monkeypatch.setattr(module, "MOTSequence", Reader)
    monkeypatch.setattr(module, "seed_everything", lambda *a, **k: None)
    ticks = iter(np.arange(0, 10, .001)); monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"test": [{"name": "s", "frame_count": 10}]}))
    result = module.benchmark_selected(ROOT, tmp_path / "detector.yaml", tmp_path / "tracker.yaml", None,
        manifest, {"s": 4}, measured_frames=2, warmup_frames=1, output=tmp_path)
    assert state == {"detect": 3, "track": 3, "decode": 3}
    assert result["warmup_frames"] == 1 and result["measured_frames"] == 2 and result["cache_used"] is False
    assert result["timings"]["core_total"]["mean_ms"] == pytest.approx(3)
    assert result["measured_core_fps"] == pytest.approx(1000 / 3)
