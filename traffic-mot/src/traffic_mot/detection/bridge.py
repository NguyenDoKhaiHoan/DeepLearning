"""Fail-fast provenance/config checks before importing or loading RT-DETR."""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from traffic_mot.detection.base import VISDRONE_CLASSES
from traffic_mot.utils.config import ConfigurationError, require, validate_assets


def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory, including large checkpoints/videos."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(profile: dict[str, Any], project_root: Path) -> tuple[dict, dict]:
    """Verify the audited R18-P2 profile, source tree, artifacts and transforms.

    The supported model config is self-contained; include-based training configs
    must first be resolved into an inference config, to avoid hidden overrides.
    """
    require(profile.get("backend") == "rtdetrv2_p2", "Expected rtdetrv2_p2 detector profile")
    require(profile.get("weights") in {"ema", "model"} and profile.get("strict_load") is True,
            "Select explicit ema/model weights with strict_load=true")
    mapping = profile["class_mapping"]
    require(mapping["model_id_to_name"] == VISDRONE_CLASSES and mapping["vehicle_model_ids"] == list(range(3, 11))
            and mapping["drop_model_ids"] == [0] and mapping["preserve_model_ids"] is True,
            "Class mapping must preserve the audited VisDrone IDs and names")
    threshold = profile["confidence_threshold"]
    require(isinstance(threshold, (int, float)) and 0 <= threshold <= 1, "confidence_threshold must be in [0,1]")
    require(profile["precision"] in {"fp32", "amp"}, "precision must be fp32 or amp")
    require(isinstance(profile["batch_size"], int) and profile["batch_size"] >= 1, "batch_size must be positive")
    provenance = validate_assets(project_root, profile)
    assets = profile["assets"]
    repo = (project_root / assets["repo"]).resolve()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout
    require(not dirty.strip(), "RT-DETR source is modified; use the clean pinned checkout")
    model_path = (project_root / assets["model_config"]).resolve()
    cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    require("__include__" not in cfg, "Use a resolved self-contained inference config")
    require(cfg["PResNet"]["depth"] == 18 and cfg["PResNet"]["return_idx"] == [0, 1, 2, 3], "Expected PResNet-18 P2..P5")
    enc, dec = cfg["HybridEncoder"], cfg["RTDETRTransformerv2"]
    require(enc["in_channels"] == [64, 128, 256, 512] and enc["hidden_dim"] == 256 and
            dec["feat_strides"] == [4, 8, 16, 32] and dec["hidden_dim"] == 256 and
            dec["num_layers"] == 3 and dec["num_queries"] == 300, "P2 architecture mismatch")
    require(cfg["eval_spatial_size"] == [640, 640] and cfg["use_focal_loss"] is True and
            cfg["RTDETRPostProcessor"]["num_top_queries"] == 300, "Postprocessor/input profile mismatch")
    transforms = cfg["val_dataloader"]["dataset"]["transforms"]
    require(transforms == {"type": "Compose", "ops": [
        {"type": "Resize", "size": [640, 640]},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]},
        "Validation transforms must be Resize640 + float32/255; normalization/letterbox unsupported")
    prep = profile["preprocessing"]
    require(prep["input_size"] == [640, 640] and prep["source_color"] == "BGR" and prep["color"] == "RGB"
            and prep["resize"] == "stretch" and prep["mean"] is None and prep["std"] is None
            and abs(prep["scale"] - 1 / 255) < 1e-12 and prep["original_size_order"] == ["width", "height"]
            and prep["transforms"] == "from_model_validation_config", "Preprocessing profile contradicts model config")
    provenance.update(model_config_sha256=file_sha256(model_path), source_clean=True,
                      source_repo=str(repo), preflight_passed=True)
    return copy.deepcopy(cfg), provenance


def load_upstream(repo: Path) -> tuple[Any, Any]:
    """Import only the verified checkout; reject a previously loaded foreign src."""
    repo = repo.resolve()
    existing = sys.modules.get("src")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        require(origin is not None and Path(origin).resolve() == repo / "src" / "__init__.py",
                "Another package named src is already loaded; start a fresh process")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    core = importlib.import_module("src.core")
    require(Path(core.__file__).resolve().is_relative_to(repo / "src"), "Foreign RT-DETR src.core import")
    transforms = importlib.import_module("src.data.transforms")
    return core.YAMLConfig, transforms.Compose
