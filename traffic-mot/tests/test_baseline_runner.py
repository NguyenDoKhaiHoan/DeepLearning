"""Shared-cache integrity, CLI failures and native sequence source invariants."""
import io
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from traffic_mot.detection.bridge import file_sha256
from traffic_mot.tracking.baselines import _batch, _validate_cache, run_baselines
from traffic_mot.utils.config import ConfigurationError
from traffic_mot.video.mot_sequence import MOTSequence

ROOT = Path(__file__).resolve().parents[1]


def test_cache_rejects_incomplete_changed_or_mock(tmp_path):
    path = tmp_path / "s.jsonl"; path.write_text('{"frame_id": 1}\n')
    signature = {"frames": {"s": 1}}
    metadata = {"signature": signature, "status": "complete", "mock": False,
                "sequences": {"s": {"file": "s.jsonl", "sha256": file_sha256(path), "frames": 1}}}
    meta = tmp_path / "manifest.json"; meta.write_text(json.dumps(metadata))
    assert _validate_cache(tmp_path, signature)["status"] == "complete"
    with pytest.raises(ConfigurationError, match="signature"):
        _validate_cache(tmp_path, {"frames": {"s": 2}})
    metadata["mock"] = True; meta.write_text(json.dumps(metadata))
    with pytest.raises(ConfigurationError, match="mock"): _validate_cache(tmp_path, signature)
    metadata["mock"] = False; meta.write_text(json.dumps(metadata)); path.write_text("changed")
    with pytest.raises(ConfigurationError, match="changed"): _validate_cache(tmp_path, signature)


def test_cache_empty_batch_preserves_shape():
    batch = _batch({"detections": []})
    assert batch.boxes.shape == (0, 4) and batch.class_ids.dtype == np.int64


def test_native_zip_sequence_order_and_original_odd_size(tmp_path):
    pixels = io.BytesIO(); Image.new("RGB", (31, 23), (10, 80, 240)).save(pixels, format="JPEG")
    path = tmp_path / "seq.zip"
    with ZipFile(path, "w") as archive:
        for i in [2, 1]: archive.writestr(f"data/sequences/s/{i:07d}.jpg", pixels.getvalue())
    sequence = {"source_kind": "images_zip", "archive": str(path), "frames_prefix": "data/sequences/s/", "frame_count": 2}
    reader = MOTSequence(sequence, tmp_path)
    with reader:
        assert reader.frames[0].endswith("0000001.jpg")
        assert reader.read(0).shape == (23, 31, 3)
    assert reader.archive.fp is None


def test_test_split_is_blocked_before_any_inference(tmp_path):
    with pytest.raises(ConfigurationError, match="test is reserved"):
        run_baselines(ROOT, tmp_path / "missing.yaml", tmp_path / "missing.json", tmp_path / "out", split="test")
    assert not (tmp_path / "out").exists()


def test_video_requires_explicit_display_fps(tmp_path):
    with pytest.raises(ConfigurationError, match="visualization-fps"):
        run_baselines(ROOT, tmp_path / "missing.yaml", tmp_path / "missing.json", tmp_path / "out", save_video=True)


def test_cli_exists_and_exposes_the_notebook_arguments():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_mot_baselines.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for argument in ["--detector-config", "--manifest", "--split", "--output-root", "--max-frames", "--sequence"]:
        assert argument in result.stdout
