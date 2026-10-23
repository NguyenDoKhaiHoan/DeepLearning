"""Protect detector/tracker boundary contracts before runtime implementation."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from traffic_mot.utils.config import ConfigurationError, load_configs, validate_configs

ROOT = Path(__file__).resolve().parents[1]


def test_template_contracts() -> None:
    validate_configs(load_configs(ROOT))


@pytest.mark.parametrize("mutation, message", [
    ("detector_cutoff", "low-confidence"),
    ("compact_classes", "compact"),
    ("reid_without_model", "extractor"),
    ("baseline_recovery", "Default profile"),
    ("wrong_experiment", "E0"),
    ("cpu_amp", "CUDA"),
])
def test_reject_invalid_integration(mutation: str, message: str) -> None:
    configs = copy.deepcopy(load_configs(ROOT))
    if mutation == "detector_cutoff":
        configs["detector"]["confidence_threshold"] = 0.3
    elif mutation == "compact_classes":
        configs["detector"]["class_mapping"]["vehicle_model_ids"] = list(range(8))
    elif mutation == "reid_without_model":
        configs["botsort"]["tracker"]["with_reid"] = True
    elif mutation == "baseline_recovery":
        configs["occlusion"]["enabled"] = True
    elif mutation == "wrong_experiment":
        configs["experiments"]["experiments"]["E0"]["recovery_enabled"] = True
    elif mutation == "cpu_amp":
        configs["detector"]["precision"] = "amp"
    with pytest.raises(ConfigurationError, match=message):
        validate_configs(configs)


def test_cli_works_from_another_directory(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_phase0.py")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["configurations"] == 8


def test_cli_reports_missing_configuration(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_phase0.py"),
                             "--project-root", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "validation failed" in result.stderr


@pytest.mark.parametrize("kind", ["track", "event"])
def test_integration_examples(kind: str) -> None:
    schema = json.loads((ROOT / "schemas" / f"{kind}.schema.json").read_text(encoding="utf-8"))
    example = json.loads((ROOT / "schemas" / f"{kind}.example.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    validator.validate(example)
    invalid = dict(example, frame_id=0)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(invalid)
    invalid = dict(example, confidence=1.5)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(invalid)
