"""Validate Phase 0 contracts without importing PyTorch or tracker backends."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAMES = (
    "detector", "bytetrack", "botsort", "occlusion", "events", "pipeline",
    "evaluation", "experiments",
)


class ConfigurationError(ValueError):
    """A configuration contradicts a cross-component contract."""


def require(condition: bool, message: str) -> None:
    """Raise an actionable validation error when a contract is violated."""
    if not condition:
        raise ConfigurationError(message)


def load_configs(project_root: Path) -> dict[str, dict[str, Any]]:
    """Read project YAML safely; all eight templates must be mappings."""
    configs = {}
    for name in CONFIG_NAMES:
        path = project_root / "configs" / f"{name}.yaml"
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        require(isinstance(value, dict), f"{path}: expected a YAML mapping")
        require(value.get("schema_version") == "1.0", f"{path}: unsupported schema_version")
        configs[name] = value
    return configs


def validate_configs(configs: dict[str, dict[str, Any]]) -> None:
    """Check mapping, score propagation, ReID claims and ablation isolation.

    This is a Phase 0 contract check, not the future runtime config parser.
    """
    detector = configs["detector"]
    mapping = detector["class_mapping"]
    names = mapping["model_id_to_name"]
    require(mapping["classifier_slots"] == 11, "P2 bundle expects 11 classifier slots")
    require(set(names) == set(range(1, 11)), "Mapping must preserve VisDrone model IDs 1..10")
    expected_vehicles = ["bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    require(mapping["vehicle_model_ids"] == list(range(3, 11)), "Do not compact vehicle class IDs")
    require([names[i] for i in range(3, 11)] == expected_vehicles, "Vehicle class names mismatch")
    require(mapping["drop_model_ids"] == [0] and mapping["preserve_model_ids"], "Drop slot 0; preserve IDs")
    require(detector["weights"] == "ema" and detector["strict_load"], "Audited profile requires strict EMA load")
    require(detector["precision"] in {"fp32", "amp"}, "precision must be fp32 or amp")
    require(detector["precision"] != "amp" or detector["device"].startswith("cuda"), "AMP requires CUDA")
    require(detector["preprocessing"]["input_size"] == [640, 640], "Expected 640x640 input")
    threshold = detector["confidence_threshold"]
    require(isinstance(threshold, (int, float)) and math.isfinite(threshold) and 0 <= threshold <= 1,
            "Invalid detector confidence threshold")
    for name in ("bytetrack", "botsort"):
        backend = configs[name]
        tracker = backend["tracker"]
        require(backend["backend"] == "ultralytics" and backend["backend_version"] == "8.4.171",
                f"{name}: unsupported backend profile")
        require(tracker["tracker_type"] == name, f"{name}: incorrect tracker_type")
        require(0 <= threshold <= tracker["track_low_thresh"] < tracker["track_high_thresh"] <= 1,
                f"{name}: detector cutoff must retain the low-confidence association band")
        require(tracker["track_high_thresh"] <= tracker["new_track_thresh"] <= 1,
                f"{name}: invalid new_track_thresh")
        require(tracker["track_buffer"] > 0 and 0 < tracker["match_thresh"] <= 1,
                f"{name}: invalid retention or matching cost")
    botsort = configs["botsort"]
    if botsort["tracker"]["with_reid"]:
        require(botsort["tracker"]["model"] not in {None, "auto"}, "RT-DETR ReID needs an explicit extractor")
        require(botsort["reid"]["status"] == "validated", "ReID model has not been validated")
        require(bool(botsort["reid"]["checkpoint_sha256"]), "ReID needs checkpoint provenance")
    occlusion = configs["occlusion"]
    require(not occlusion["enabled"], "Default profile must keep recovery disabled for baselines")
    require(not occlusion["reassociation"]["use_appearance"] or botsort["tracker"]["with_reid"],
            "Appearance recovery requires an integrated ReID model")
    require(occlusion["retention"]["max_frames"] >= occlusion["retention"]["base_frames"] > 0,
            "Recovery retention must be bounded and >= base_frames")
    require(not occlusion["output"]["emit_unobserved_as_detection"], "Predictions are not observations")
    experiments = configs["experiments"]["experiments"]
    for name, tracker, enabled in (("E0", "bytetrack", False), ("E1", "botsort", False),
                                   ("E2", "bytetrack", True), ("E3", "botsort", True)):
        require(experiments[name]["tracker_config"] == f"{tracker}.yaml" and
                experiments[name]["recovery_enabled"] is enabled and not experiments[name]["with_reid"],
                f"{name}: ablation must isolate recovery with ReID disabled")
    rules = configs["events"]["rules"]
    rule_ids = [rule["id"] for rule in rules]
    require(len(rule_ids) == len(set(rule_ids)), "Event rule IDs must be unique")
    require(configs["pipeline"]["input"]["frame_id_origin"] == 1, "MOT frame IDs start at 1")
    evaluated = configs["evaluation"]["dataset"]["evaluated_class_ids"]
    require(set(evaluated) <= set(mapping["vehicle_model_ids"]), "Evaluation scope must use mapped vehicle IDs")


def validate_assets(project_root: Path, detector: dict[str, Any]) -> dict[str, Any]:
    """Read local provenance; verify SHA256, labels, config and source commit.

    Never load a model, download weights or mutate the upstream repository.
    """
    assets = detector["assets"]
    paths = {name: (project_root / assets[name]).resolve() for name in
             ("repo", "model_config", "checkpoint", "label_map", "classes", "bundle_manifest")}
    for name, path in paths.items():
        require(path.is_dir() if name == "repo" else path.is_file(), f"Missing asset {name}: {path}")
    label_map = json.loads(paths["label_map"].read_text(encoding="utf-8"))
    require(label_map == {str(i): (i if i else None) for i in range(11)}, "Artifact label_map differs from profile")
    names = paths["classes"].read_text(encoding="utf-8").splitlines()
    expected = detector["class_mapping"]["model_id_to_name"]
    require(names == [expected[i] for i in range(1, 11)], "Artifact classes differ from profile")
    model_config = yaml.safe_load(paths["model_config"].read_text(encoding="utf-8"))
    require(model_config["num_classes"] == 11 and not model_config["remap_mscoco_category"],
            "Model config classifier mapping mismatch")
    require(model_config["HybridEncoder"]["feat_strides"] == [4, 8, 16, 32], "Model config missing P2")
    require(model_config["RTDETRTransformerv2"]["num_levels"] == 4, "Expected four decoder levels")
    manifest = json.loads(paths["bundle_manifest"].read_text(encoding="utf-8"))
    entry = next((item for item in manifest["entries"] if item["local_file"] == "best.pth"), None)
    require(entry is not None, "Bundle manifest has no best.pth entry")
    digest = hashlib.sha256()
    with paths["checkpoint"].open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    require(digest.hexdigest() == entry["sha256"], "Checkpoint SHA256 differs from bundle manifest")
    result = subprocess.run(["git", "-C", str(paths["repo"]), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True)
    commit = result.stdout.strip()
    require(commit == assets["repo_commit"] == manifest["repo_commit"], "RT-DETR commit mismatch")
    return {"checkpoint_sha256": digest.hexdigest(), "repo_commit": commit,
            "label_mapping": "artifact_verified", "inference_run": False}
