"""Opt-in integration checks: real strict EMA load plus controlled contract probes."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import torch
import yaml

from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
from traffic_mot.utils.config import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def detector():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    model = RTDETRv2P2Detector(ROOT / "configs/detector.yaml", device="cpu")
    yield model
    torch.set_num_threads(previous)


def test_real_load_and_rgb_scaling(detector):
    assert detector.checkpoint_epoch == 119 and detector.provenance["strict_load"]
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:] = [10, 80, 240]  # BGR, not RGB
    tensor, size = detector.preprocess(frame)
    assert tensor.shape == (3, 640, 640) and size.tolist() == [200, 100]
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([240, 80, 10]) / 255)
    assert not detector.model.training


def test_known_logits_scale_and_low_confidence_batch(detector, monkeypatch):
    class KnownOutputs(torch.nn.Module):
        def forward(self, images):
            b = len(images)
            logits = torch.full((b, 300, 11), -50.)
            logits[:, 0, 0] = 20.  # reserved slot
            logits[:, 1, 4] = float(torch.logit(torch.tensor(.8)))
            logits[:, 2, 5] = float(torch.logit(torch.tensor(.15)))
            logits[:, 3, 1] = 15.  # pedestrian excluded
            return {"pred_logits": logits, "pred_boxes": torch.tensor([.5, .5, .2, .4]).repeat(b, 300, 1)}
    with monkeypatch.context() as patch:
        patch.setattr(detector, "model", KnownOutputs())
        patch.setattr(detector, "batch_size", 2)
        batches = detector.detect_batch([np.zeros((100, 200, 3), np.uint8), np.zeros((200, 100, 3), np.uint8)])
    assert batches[0].class_ids.tolist() == [4, 5]
    assert batches[0].scores.tolist() == pytest.approx([.8, .15])
    assert np.allclose(batches[0].boxes, [[80, 30, 120, 70]] * 2, atol=1e-4)
    assert np.allclose(batches[1].boxes, [[40, 60, 60, 140]] * 2, atol=1e-4)


def test_real_forward_on_existing_traffic_frame(detector):
    source = ROOT.parent / "outputs/web_demo/0315259f-8088-44c9-8c3c-129efe920d21/input.mp4"
    capture = cv2.VideoCapture(str(source))
    ok, frame = capture.read(); capture.release()
    assert ok, "A real traffic input video is required for --run-model"
    batch = detector.detect(frame)
    assert len(batch) > 0
    assert set(batch.class_ids) <= set(range(3, 11))
    assert np.isfinite(batch.boxes).all() and np.isfinite(batch.scores).all()
    assert (batch.scores >= .1).all() and (batch.scores <= 1).all()
    assert (batch.boxes[:, [0, 2]] <= frame.shape[1]).all()
    assert (batch.boxes[:, [1, 3]] <= frame.shape[0]).all()
    assert detector.metadata()["mock"] is False and detector.metadata()["inference_run"] is True


def test_tampered_manifest_rejects_before_weight_load(tmp_path, monkeypatch):
    profile = yaml.safe_load((ROOT / "configs/detector.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / profile["assets"]["bundle_manifest"]).read_text(encoding="utf-8"))
    next(entry for entry in manifest["entries"] if entry["local_file"] == "best.pth")["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    profile["assets"]["bundle_manifest"] = str(manifest_path)
    profile_path = tmp_path / "detector.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append(True))
    with pytest.raises(ConfigurationError, match="SHA256"):
        RTDETRv2P2Detector(profile_path, project_root=ROOT)
    assert not called
