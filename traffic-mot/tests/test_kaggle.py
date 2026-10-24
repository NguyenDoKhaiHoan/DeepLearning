"""Notebook structure and honest preflight, including split leakage rejection."""
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.kaggle import check_gpu, check_mot_dataset, run_stage

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_is_required_without_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="GPU/CUDA unavailable"): check_gpu()
    assert check_gpu(required=False)["cuda_available"] is False


def test_unimplemented_stage_stops_instead_of_generating_scores():
    with pytest.raises(ConfigurationError, match="Stage not implemented"):
        run_stage(ROOT, "run_ablation.py", [])


@pytest.fixture
def split_manifest(tmp_path):
    # Synthetic fixtures validate the checker only, never reported as experiments.
    for index, name in enumerate(("validation", "test")):
        writer = cv2.VideoWriter(str(tmp_path / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
        assert writer.isOpened()
        for _ in range(2): writer.write(np.full((32, 32, 3), 40 + index * 100, np.uint8))
        writer.release()
        (tmp_path / f"{name}.txt").write_text("1,1,0,0,10,10,1,4,0,0\n2,1,1,0,10,10,1,4,0,1\n")
    data = {"dataset": "VisDrone-MOT", "validation": [{"name": "val_01", "video": "validation.mp4", "gt": "validation.txt"}],
            "test": [{"name": "test_01", "video": "test.mp4", "gt": "test.txt"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_mot_preflight_and_no_test_gt_inspection(split_manifest):
    (split_manifest.parent / "test.txt").write_text("held out; do not parse during validation")
    result = check_mot_dataset(split_manifest, split="validation")
    assert result["sequences"][0]["gt_observations"] == 2
    assert result["full_evaluator_validation"] is False


def test_reject_split_leakage(split_manifest):
    data = json.loads(split_manifest.read_text())
    data["test"][0]["video"] = "validation.mp4"
    split_manifest.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="different source sequences"): check_mot_dataset(split_manifest)


def test_reject_detection_gt(split_manifest):
    (split_manifest.parent / "validation.txt").write_text("1,2,10,20,1,4,0,0\n")
    with pytest.raises(ConfigurationError, match="not VisDrone DET"): check_mot_dataset(split_manifest)


def test_notebooks_are_thin_unexecuted_and_syntax_valid():
    names = ["01_detection_inference", "02_mot_baselines", "03_occlusion_ablation", "04_final_benchmark"]
    for name in names:
        notebook = json.loads((ROOT.parent / "notebooks/kaggle" / f"{name}.ipynb").read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        all_code = ""
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                text = "".join(cell["source"])
                ast.parse(text)
                all_code += text
                assert cell["execution_count"] is None and cell["outputs"] == []
        assert "traffic_mot.utils.kaggle" in all_code and "require_gpu=True" in all_code
        assert "class " not in all_code and "def " not in all_code
        if name in {"02_mot_baselines", "03_occlusion_ablation"}:
            assert 'split="validation"' in all_code
