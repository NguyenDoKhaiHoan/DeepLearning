"""Study definitions, validation isolation and smoke-vs-frozen selection guards."""
from pathlib import Path
import json
import subprocess
import sys

import pytest
import yaml

pytest.importorskip("trackeval")
pytest.importorskip("lap")

from traffic_mot.occlusion.ablation import experiment_definitions, run_ablation
from traffic_mot.utils.config import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_components_isolate_one_factor():
    conf = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    definitions, policy, path = experiment_definitions(conf, ROOT / "configs")
    assert len(definitions) == 8 and path.is_file()
    assert definitions["E0"]["recovery"] is None and definitions["E1"]["recovery"] is None
    assert definitions["E0"]["tracker_config"] == definitions["E2"]["tracker_config"]
    assert definitions["E1"]["tracker_config"] == definitions["E3"]["tracker_config"]
    for backend in ["BT", "BS"]:
        assert not definitions[f"{backend}_retention_only"]["recovery"]["reassociation"]["enabled"]
        assert not definitions[f"{backend}_reassociation_only"]["recovery"]["retention"]["adaptive"]
    assert policy["enabled"]


def test_ablation_blocks_test_before_reading_assets(tmp_path):
    with pytest.raises(ConfigurationError, match="never optimize on test"):
        run_ablation(ROOT, tmp_path / "no.yml", tmp_path / "no.yml", tmp_path / "no.json", tmp_path / "out", split="test")
    assert not (tmp_path / "out").exists()


def test_cli_exposes_exact_notebook_arguments():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_ablation.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for name in ["--config", "--detector-config", "--manifest", "--split", "--output-root", "--baseline-run"]:
        assert name in result.stdout


@pytest.mark.parametrize("smoke", [True, False])
def test_selection_is_not_frozen_on_prefix_smoke(tmp_path, monkeypatch, smoke):
    # Fake runner records test-only metrics to exercise selection/serialization,
    # never reported as experiments. Real runner has separate smoke verification.
    import traffic_mot.occlusion.ablation as module
    output = tmp_path / "run"
    metrics = {"HOTA": .5, "IDF1": .6, "MOTA": .4, "IDSW": 3, "MT": 1, "ML": 2, "CLR_TP": 8, "CLR_FP": 2, "CLR_FN": 4}
    def fixture_runner(*args, **kwargs):
        output.mkdir()
        summaries = {}
        for name in kwargs["experiment_profiles"]:
            (output / name).mkdir()
            (output / name / "diagnostics.json").write_text(json.dumps({"episodes_gt_sha256": "fixed_gt"}))
            summaries[name] = dict(metrics)
        manifest = {"status": "complete", "smoke_test": smoke, "summary": summaries,
                    "signature": {"manifest_sha256": "real_hash_field_fixture"}}
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "detector_profile.json").write_text("{}")
        return manifest
    monkeypatch.setattr(module, "run_baselines", fixture_runner)
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k: {})
    result = run_ablation(ROOT, ROOT / "configs/experiments.yaml", tmp_path / "detector.yml", tmp_path / "mot.json", output)
    assert result["status"] == "complete"
    selection = json.loads((output / "selection.json").read_text())
    assert selection["frozen"] is (not smoke)
    assert selection["selected_experiment"] == "E0"  # Prefer baseline on equal metrics.
    assert (output / "selected_config.yaml").exists() is (not smoke)
    assert (output / "candidate_config.yaml").exists() is smoke


def test_different_gt_episode_denominator_fails_study(tmp_path, monkeypatch):
    import traffic_mot.occlusion.ablation as module
    output = tmp_path / "broken"
    def fixture_runner(*args, **kwargs):
        output.mkdir()
        for name in kwargs["experiment_profiles"]:
            (output / name).mkdir()
            (output / name / "diagnostics.json").write_text(json.dumps({"episodes_gt_sha256": name}))
        return {"status": "complete", "smoke_test": True, "signature": {}, "summary": {}}
    monkeypatch.setattr(module, "run_baselines", fixture_runner)
    with pytest.raises(ConfigurationError, match="denominator differs"):
        run_ablation(ROOT, ROOT / "configs/experiments.yaml", tmp_path / "detector.yml", tmp_path / "mot.json", output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
    assert not (output / "selected_config.yaml").exists()


def test_real_toolkit_paired_bootstrap_uses_sequences(tmp_path):
    from traffic_mot.evaluation.trackeval_runner import evaluate_sequences
    from traffic_mot.occlusion.ablation import paired_bootstrap
    g = {"track_id": 0, "class_id": 4, "bbox": [0, 0, 10, 10], "ignored": False}
    p = {"track_id": 1, "class_id": 4, "bbox": [0, 0, 10, 10]}
    baseline = evaluate_sequences({"one": ([[g]], [[]]), "two": ([[g]] * 3, [[]] * 3)}, [4])
    proposed = evaluate_sequences({"one": ([[g]], [[p]]), "two": ([[g]] * 3, [[p]] * 3)}, [4])
    for name, values in [("E0", baseline), ("E1", baseline), ("E2", proposed), ("E3", proposed)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(values))
    result = paired_bootstrap(tmp_path, samples=100, seed=42)
    for comparison in result["comparisons"].values():
        assert comparison["sequence_count"] == 2
        assert comparison["ci95"]["HOTA"]["mean"] == 1
        assert comparison["ci95"]["HOTA"]["low"] == comparison["ci95"]["HOTA"]["high"] == 1
        assert comparison["ci95"]["HOTA"]["degenerate_one_sequence"] is False
