"""Generate thin Kaggle notebooks; algorithms stay in traffic_mot source."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build() -> None:
    output = ROOT / "notebooks/kaggle"
    output.mkdir(parents=True, exist_ok=True)
    setup = code('''from pathlib import Path
import sys

# Point to the checked-out repository and real Kaggle Input bundle.
REPO_ROOT = Path("/kaggle/working/DeepLearning")
PROJECT_ROOT = REPO_ROOT / "traffic-mot"
BUNDLE_DIR = Path("/kaggle/input/REPLACE_ME/E1_p2_final")
UPSTREAM_REPO = REPO_ROOT / "RT-DETR/rtdetrv2_pytorch"
VIDEO = Path("/kaggle/input/REPLACE_ME/traffic.mp4")
OUTPUT_ROOT = Path("/kaggle/working/traffic_mot_results")
MOT_DATA_ROOT = Path("/kaggle/input/visdrone-mot")  # Use the actual Add Input mount path.
MOT_MANIFEST = OUTPUT_ROOT / "mot_split_manifest.json"  # Generated below for notebooks 02-04.
assert (PROJECT_ROOT / "src/traffic_mot").is_dir(), "Mount/clone this repository first"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from traffic_mot.utils.kaggle import (prepare_detection_config, check_video_dataset,
                                     check_mot_dataset, run_detection, run_stage)
''')
    prepare_mot = code('''# VisDrone MOT uses image sequences; do not convert them into a fake-FPS MP4.
# Create from ZIPs or extracted directories; val is tuning, test-dev is held out.
from traffic_mot.utils.mot_data import prepare_mot_manifest
if not MOT_MANIFEST.exists():
    prepare_mot_manifest(MOT_MANIFEST, root=MOT_DATA_ROOT)
print("MOT_MANIFEST:", MOT_MANIFEST)
# Preparation lists frame/GT members without parsing held-out test annotations.
# Input preparation only; final test additionally requires the frozen validation artifact.
''')
    preflight = code('''config_path, preflight_report = prepare_detection_config(
    PROJECT_ROOT, BUNDLE_DIR, UPSTREAM_REPO, OUTPUT_ROOT / "configs",
    device="cuda:0", require_gpu=True)
print(preflight_report)
''')
    install = markdown('''## Environment and assets

Enable a Kaggle GPU accelerator. Clone/mount this repository, then install its
`traffic-mot/requirements.txt` and `requirements-dev.txt` into an environment with
matching PyTorch/torchvision CUDA wheels. Inspect `pip check` before proceeding.
Clone the official RT-DETR repository at commit
`29320b6fd828f8e0987a71426cf2d961b09dfed7` into `RT-DETR/` if it is absent;
keep this checkout clean. Do not substitute a YOLO detector or auto-download weights.

Mount the real bundle (`best.pth`, `demo_config.yml`, `label_map.json`,
`classes.txt`, `bundle_manifest.json`). Edit paths below; REPLACE_ME must not remain.
Preflight rejects missing/wrong weights, mapping, preprocessing or source/GPU.
No notebook has been executed on Kaggle in this implementation session.

Repository: https://github.com/NguyenDoKhaiHoan/DeepLearning
Exact clone/install commands: `notebooks/kaggle/README.md` in that repository.
The Git repository contains source only; attach the checkpoint bundle and video
as Kaggle Input separately. Restart the kernel if dependency installation changed
an already imported PyTorch/NumPy version.
''')
    specs = [
        ("01_detection_inference.ipynb", "01 — Detection integration / Kaggle inference",
         "Implemented shared detection code; local real CPU smoke evidence exists. Kaggle GPU gate is pending.",
         [code('''video_report = check_video_dataset(VIDEO)
print(video_report)  # This video has no MOT GT: no HOTA/IDF1/MOTA may be reported.
'''), code('''from uuid import uuid4
result = run_detection(PROJECT_ROOT, config_path, VIDEO,
    OUTPUT_ROOT / ("detection_" + uuid4().hex), max_frames=48)
print({k: result[k] for k in ["status", "mock", "frames_processed", "detection_count", "output_dir"]})
assert result["status"] == "complete" and result["mock"] is False
from traffic_mot.utils.verify import verify_detection_run
print(verify_detection_run(Path(result["output_dir"])))
# Smoke only. Remove max_frames for a full video after inspecting the outputs.
''')]),
        ("02_mot_baselines.ipynb", "02 — ByteTrack / BoT-SORT and MOT baseline evaluation",
         "Implemented E0/E1 stock baselines and official TrackEval metrics. Real local CPU prefix smoke verified; full Kaggle validation pending.",
         [markdown('''## Baseline dependencies

Install `traffic-mot/requirements.txt` and `traffic-mot/requirements-eval.txt`
using this kernel's Python before running. TrackEval must be the Git commit pinned
in requirements-eval.txt. E0=ByteTrack, E1=BoT-SORT with GMC and ReID off.
This notebook runs validation only. Start with a sequence/prefix smoke, inspect
outputs, then select FULL_VALIDATION=True. No OAR/ablation/final-test result is produced.
'''), code('''from traffic_mot.tracking.backend_bridge import check_backend_dependencies
from traffic_mot.evaluation.trackeval_runner import load_metrics
check_backend_dependencies()
_, metric_backend = load_metrics()
print(metric_backend)
dataset_report = check_mot_dataset(MOT_MANIFEST, split="validation")
print(dataset_report)
'''), code('''from uuid import uuid4
FULL_VALIDATION = False  # Set True only after the real prefix smoke succeeds.
BASELINE_RUN = OUTPUT_ROOT / ("baselines_" + uuid4().hex)
arguments = [
    "--detector-config", str(config_path), "--manifest", str(MOT_MANIFEST),
    "--split", "validation", "--output-root", str(BASELINE_RUN)]
if not FULL_VALIDATION:
    arguments += ["--sequence", dataset_report["sequences"][0]["name"], "--max-frames", "48"]
# Optional display-only video (native JPG FPS is unknown):
# arguments += ["--save-video", "--visualization-fps", "10"]
run_stage(PROJECT_ROOT, "run_mot_baselines.py", arguments)
import json
summary = json.loads((BASELINE_RUN / "summary.json").read_text())
print(summary)
print("Outputs:", BASELINE_RUN)
# HOTA/IDF1/MOTA are ratios 0..1; IDSW is a count. Prefix smoke is NOT a full split result.
# E0/E1 share a verified detection cache. No FPS benchmark is inferred from replay timings.
''')]),
        ("03_occlusion_ablation.ipynb", "03 — Occlusion-aware recovery / validation ablations",
         "Implemented OAR E2/E3 plus retention/reassociation component runs. Real CPU prefix smoke verified; full validation and improvement evidence pending.",
         [markdown('''## Validation-only study

Install the same pinned runtime and requirements-eval.txt as notebook02.
This runs stock E0/E1 plus OAR E2/E3 and both trackers' retention-only /
reassociation-only variants using one verified detection cache. No ReID.
Use BASELINE_REFERENCE to reuse a completed notebook02 run and verify stock metrics.
Without a reference the script recomputes E0/E1 itself. A smoke run creates only
candidate_config.yaml; selected_config.yaml is created only after full validation.
OAR may reduce metrics; inspect all results rather than assuming improvement.
'''), code('''from traffic_mot.tracking.backend_bridge import check_backend_dependencies
from traffic_mot.evaluation.trackeval_runner import load_metrics
check_backend_dependencies()
_, metric_backend = load_metrics()
print(metric_backend)
dataset_report = check_mot_dataset(MOT_MANIFEST, split="validation")
print(dataset_report)
'''), code('''from uuid import uuid4
import json
FULL_VALIDATION = False
BASELINE_REFERENCE = None  # e.g. Path("/kaggle/working/traffic_mot_results/baselines_<uuid>")
ABLATION_RUN = OUTPUT_ROOT / ("ablation_" + uuid4().hex)
arguments = [
    "--config", str(PROJECT_ROOT / "configs/experiments.yaml"),
    "--detector-config", str(config_path), "--manifest", str(MOT_MANIFEST),
    "--split", "validation", "--output-root", str(ABLATION_RUN)]
if BASELINE_REFERENCE is not None:
    arguments += ["--baseline-run", str(BASELINE_REFERENCE)]  # Exact range/cache inferred from reference.
elif not FULL_VALIDATION:
    arguments += ["--sequence", dataset_report["sequences"][0]["name"], "--max-frames", "48"]
run_stage(PROJECT_ROOT, "run_ablation.py", arguments)
print(json.loads((ABLATION_RUN / "ablation_summary.json").read_text()))
print(json.loads((ABLATION_RUN / "selection.json").read_text()))
print("Outputs:", ABLATION_RUN)
# No test tuning. A prefix smoke never freezes selected_config.yaml for notebook04.
''')]),
        ("04_final_benchmark.ipynb", "04 — Frozen held-out evaluation / benchmark / demo",
         "Implemented frozen-config test evaluation and real core benchmark. Requires a completed full-validation selection; Kaggle test execution evidence pending.",
         [markdown('''## Select the actual preceding validation run

Notebook03 writes into ablation_<uuid>, not necessarily a folder named ablation.
Set ABLATION_RUN below to that directory when running in a fresh kernel, or the
helper uses the existing notebook03 variable/unique frozen selection.
Smoke candidate_config.yaml cannot be renamed into a final configuration: source
manifest, full validation counts, selection checksum and tracker/model profiles
are verified before test GT or inference. No thresholds are tuned on test.
'''), code('''from traffic_mot.evaluation.final import resolve_frozen_config
# Fresh kernel example (replace with your completed FULL validation run):
# ABLATION_RUN = OUTPUT_ROOT / "ablation_<uuid>"
FROZEN_CONFIG = resolve_frozen_config(OUTPUT_ROOT, globals().get("ABLATION_RUN"))
CONFIG_PATH = Path(globals().get("CONFIG_PATH", config_path))
assert CONFIG_PATH.is_file(), f"Missing detector config: {CONFIG_PATH}"
print("Frozen validation config:", FROZEN_CONFIG)
# No test GT is read until the final runner validates the freeze artifact.
'''), code('''from uuid import uuid4
import json
FINAL_RUN = OUTPUT_ROOT / ("final_" + uuid4().hex)
arguments = [
    "--config", str(FROZEN_CONFIG), "--detector-config", str(CONFIG_PATH),
    "--manifest", str(MOT_MANIFEST), "--split", "test",
    "--output-root", str(FINAL_RUN), "--benchmark-frames", "100"]
# Optional demo playback (display-only FPS, not camera/processing FPS):
# arguments += ["--save-video", "--visualization-fps", "10"]
run_stage(PROJECT_ROOT, "final_evaluation.py", arguments)
result = json.loads((FINAL_RUN / "final_summary.json").read_text())
print(result)
print("DONE. Outputs:", FINAL_RUN)
# Benchmark measures actual decode+detector+tracking/OAR after warmup.
# Rendering/encoding/export are excluded; cached replay is never reported as core FPS.
''')]),
    ]
    for name, title, status, cells in specs:
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "implementation_status": status},
            "cells": [markdown(f"# {title}\n\n{status}\n"), install, setup,
                      *([prepare_mot] if not name.startswith("01_") else []), preflight, *cells,
                      markdown("## Acceptance evidence\n\nRetain manifest, resolved config, source/checkpoint/data hashes, exact environment, raw predictions and decoded video verification. Only real runs on the declared split may populate metric tables. Missing assets/data/GPU are blockers, not zero scores.\n")]}
        for index, cell in enumerate(notebook["cells"]): cell["id"] = f"cell-{index:02d}"
        (output / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
