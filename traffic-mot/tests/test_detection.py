"""Detector contract, explicit fixtures, postprocessing and preflight failures."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import yaml

from traffic_mot.detection.base import BaseDetector
from traffic_mot.detection.bridge import load_upstream, preflight
from traffic_mot.detection.mock import MockDetector
from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.types import DetectionBatch

ROOT = Path(__file__).resolve().parents[1]


def test_base_is_abstract():
    with pytest.raises(TypeError): BaseDetector()


def test_mock_replays_low_confidence_original_ids_and_clips():
    frame = np.zeros((100, 200, 3), np.uint8)
    detector = MockDetector([[{"bbox": [-3, 20, 240, 90], "confidence": .15, "class_id": 4},
                             {"bbox": [1, 1, 5, 5], "confidence": .05, "class_id": 10}]])
    batch = detector.detect(frame)
    assert batch.boxes.tolist() == [[0, 20, 200, 90]]
    assert batch.class_ids.tolist() == [4]
    assert batch.scores[0] == pytest.approx(.15)
    assert batch.detection_indices.tolist() == [0]
    assert detector.detect(frame).boxes.shape == (0, 4)
    assert detector.metadata()["mock"] is True
    assert not frame.any()


@pytest.mark.parametrize("frame", [None, np.zeros((10, 10), np.uint8), np.zeros((0, 4, 3), np.uint8),
                                   np.zeros((10, 10, 3), np.float32)])
def test_invalid_frames(frame):
    with pytest.raises(ValueError, match="BGR uint8"): MockDetector().detect(frame)


def test_mock_batch_and_invalid_fixture():
    frame = np.zeros((20, 30, 3), np.uint8)
    assert [len(d) for d in MockDetector().detect_batch([frame, frame])] == [0, 0]
    with pytest.raises(ValueError, match="3..10"):
        MockDetector([[{"bbox": [0, 0, 5, 5], "confidence": .9, "class_id": 0}]]).detect(frame)


def test_batch_confidence_validation():
    with pytest.raises(ValueError, match="confidence"):
        DetectionBatch(np.array([[0, 0, 2, 2]], np.float32), np.array([1.1], np.float32),
                       np.array([4]), np.array([0]))


def postprocessor_adapter():
    # Isolate filter_output unit tests; no model/checkpoint is created here.
    adapter = object.__new__(RTDETRv2P2Detector)
    adapter.threshold = .1
    return adapter


def test_filter_preserves_low_scores_and_rejects_invalid_geometry():
    batch = postprocessor_adapter().filter_output({
        "boxes": torch.tensor([[-2, 10, 230, 90], [20, 20, 10, 30], [1, 1, 10, 10],
                               [1, 1, 10, 10], [float('nan'), 1, 5, 5], [1, 1, 5, 5]]),
        "scores": torch.tensor([.15, .8, .99, .09, .7, .99]),
        "labels": torch.tensor([4, 4, 0, 10, 5, 1])}, (100, 200, 3))
    assert batch.boxes.tolist() == [[0, 10, 200, 90]]
    assert batch.scores[0] == pytest.approx(.15)
    assert batch.class_ids.tolist() == [4]


def test_filter_unknown_label_fails_even_below_cutoff():
    with pytest.raises(ConfigurationError, match="Unmapped"):
        postprocessor_adapter().filter_output({"boxes": torch.tensor([[0., 0, 2, 2]]),
            "scores": torch.tensor([.01]), "labels": torch.tensor([11])}, (10, 10, 3))


def test_filter_empty():
    batch = postprocessor_adapter().filter_output({"boxes": torch.empty((0, 4)),
                "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.int64)}, (10, 20, 3))
    assert batch.boxes.shape == (0, 4)


def test_missing_checkpoint_fails_before_load(tmp_path, monkeypatch):
    profile = yaml.safe_load((ROOT / "configs/detector.yaml").read_text(encoding="utf-8"))
    profile["assets"]["checkpoint"] = str(tmp_path / "missing.pth")
    profile["assets"]["repo"] = str(tmp_path)
    existing_config = tmp_path / "model.yml"
    existing_config.write_text("{}")
    profile["assets"]["model_config"] = str(existing_config)
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append(True))
    with pytest.raises(ConfigurationError, match="Missing asset checkpoint"):
        preflight(profile, ROOT)
    assert not called


@pytest.mark.parametrize("mutation", ["mapping", "transforms", "architecture", "preprocessing", "dirty_source"])
def test_preflight_rejects_inconsistent_bundle(tmp_path, monkeypatch, mutation):
    # Existing config is the source of the contract, not a synthetic checkpoint.
    profile = yaml.safe_load((ROOT / "configs/detector.yaml").read_text(encoding="utf-8"))
    model_cfg = {
        "PResNet": {"depth": 18, "return_idx": [0, 1, 2, 3]},
        "HybridEncoder": {"in_channels": [64, 128, 256, 512], "hidden_dim": 256},
        "RTDETRTransformerv2": {"feat_strides": [4, 8, 16, 32], "hidden_dim": 256,
                               "num_layers": 3, "num_queries": 300},
        "eval_spatial_size": [640, 640], "use_focal_loss": True,
        "RTDETRPostProcessor": {"num_top_queries": 300},
        "val_dataloader": {"dataset": {"transforms": {"type": "Compose", "ops": [
            {"type": "Resize", "size": [640, 640]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]}}}}
    import traffic_mot.detection.bridge as bridge
    monkeypatch.setattr(bridge, "validate_assets", lambda *a: {})
    monkeypatch.setattr(bridge.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" M src/core.py" if mutation == "dirty_source" else ""))
    if mutation == "mapping": profile["class_mapping"]["model_id_to_name"][4] = "bus"
    elif mutation == "transforms": model_cfg["val_dataloader"]["dataset"]["transforms"]["ops"].append({"type": "Normalize"})
    elif mutation == "architecture": model_cfg["RTDETRTransformerv2"]["num_layers"] = 6
    elif mutation == "preprocessing": profile["preprocessing"]["resize"] = "letterbox"
    config_file = tmp_path / "model.yml"
    config_file.write_text(yaml.safe_dump(model_cfg), encoding="utf-8")
    profile["assets"]["model_config"] = str(config_file)
    with pytest.raises(ConfigurationError): preflight(profile, ROOT)


def test_foreign_src_import_fails(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "src", SimpleNamespace(__file__=str(tmp_path / "src/__init__.py")))
    with pytest.raises(ConfigurationError, match="already loaded"): load_upstream(ROOT)


@pytest.mark.parametrize("device,precision,message", [("cpu", "amp", "AMP requires CUDA"),
                                                     ("cuda:999", "fp32", "CUDA unavailable")])
def test_unsupported_execution_fails_before_weight_load(monkeypatch, device, precision, message):
    import traffic_mot.detection.rtdetrv2_p2 as module
    monkeypatch.setattr(module, "preflight", lambda *args: ({}, {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append(True))
    with pytest.raises(ConfigurationError, match=message):
        RTDETRv2P2Detector(ROOT / "configs/detector.yaml", device=device, precision=precision)
    assert not called
