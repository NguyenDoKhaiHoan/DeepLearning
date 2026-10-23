"""Shared Kaggle preflight/orchestration; notebooks contain no CV algorithms."""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import yaml

from traffic_mot.detection.bridge import file_sha256, preflight
from traffic_mot.utils.config import require


def check_gpu(*, required: bool = True) -> dict:
    """Report actual PyTorch CUDA availability; never silently select CPU on Kaggle."""
    import torch
    available = torch.cuda.is_available()
    require(not required or available, "GPU/CUDA unavailable. Enable a Kaggle GPU accelerator before running.")
    return {"cuda_available": available, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else []}


def prepare_detection_config(project_root: Path, bundle_dir: Path, upstream_repo: Path,
                             output_dir: Path, *, device: str = "cuda:0", require_gpu: bool = True) -> tuple[Path, dict]:
    """Resolve Kaggle mount paths, verify the real bundle, and write a run profile."""
    project_root, bundle_dir, upstream_repo = project_root.resolve(), bundle_dir.resolve(), upstream_repo.resolve()
    gpu = check_gpu(required=require_gpu)
    require(not require_gpu or device.startswith("cuda"), "GPU notebook requires an explicit CUDA device")
    profile = yaml.safe_load((project_root / "configs/detector.yaml").read_text(encoding="utf-8"))
    profile = copy.deepcopy(profile)
    profile["device"] = device
    for key, name in {"model_config": "demo_config.yml", "checkpoint": "best.pth", "label_map": "label_map.json",
                      "classes": "classes.txt", "bundle_manifest": "bundle_manifest.json"}.items():
        profile["assets"][key] = str(bundle_dir / name)
    profile["assets"]["repo"] = str(upstream_repo)
    _, assets = preflight(profile, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"detector_{uuid4().hex}.yaml"
    target.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return target, {"gpu": gpu, "assets": assets, "config_sha256": file_sha256(target)}


def check_video_dataset(path: Path) -> dict:
    """Check a real video before model loading; this provides no detection/MOT GT."""
    from traffic_mot.video.reader import VideoReader
    with VideoReader(path) as reader:
        packet = next(reader)
        return {"path": reader.source_reference, "sha256": file_sha256(path),
                "fps": reader.fps, "frame_count": reader.frame_count,
                "width": packet.frame.shape[1], "height": packet.frame.shape[0], "tracking_gt": False}


def check_mot_dataset(manifest_path: Path, *, split: str = "validation") -> dict:
    """Sanity-check real tracking annotation and disjoint splits, not MOT scoring.

    Manifest declares validation/test sequence lists with name/video/gt paths.
    Paths resolve relative to the manifest. Reject DET rows and identity-less GT.
    Full ignore policy/class conversion remains a Stage 3 evaluation task.
    """
    require(split in {"validation", "test"}, "Use an explicit validation or test split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("dataset") == "VisDrone-MOT", "Declare a real VisDrone-MOT dataset")
    require(bool(manifest.get("validation")) and bool(manifest.get("test")), "Provide disjoint validation and test sequences")
    base = manifest_path.resolve().parent
    names, videos, hashes = set(), set(), set()
    selected = []
    for group in ("validation", "test"):
        for sequence in manifest[group]:
            name = sequence["name"]
            video, gt = (base / sequence["video"]).resolve(), (base / sequence["gt"]).resolve()
            require(video.is_file() and gt.is_file(), f"Missing real video or MOT GT for {name}")
            video_hash = file_sha256(video)
            require(name not in names and video not in videos and video_hash not in hashes,
                    "Validation/test must use different source sequences, not copies of one clip")
            names.add(name); videos.add(video); hashes.add(video_hash)
            if group != split: continue  # Do not inspect held-out annotations during tuning.
            video_info = check_video_dataset(video)
            identities: dict[int, set[int]] = {}
            seen = set()
            with gt.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.reader(stream):
                    if not row: continue
                    require(len(row) == 10, "Expected 10-column MOT annotation, not VisDrone DET")
                    values = [float(value) for value in row]
                    require(all(math.isfinite(value) for value in values), "Nonfinite annotation value")
                    frame, target = int(values[0]), int(values[1])
                    require(values[0] == frame and values[1] == target and frame >= 1, "Invalid GT frame/identity")
                    require(video_info["frame_count"] is None or frame <= video_info["frame_count"], "GT frame exceeds video length")
                    if values[6] == 0 or values[7] in {0, 11}: continue
                    require(target >= 1 and values[4] > 0 and values[5] > 0, "Valid MOT GT needs positive persistent ID and box")
                    require((frame, target) not in seen, "Duplicate frame/identity in MOT GT")
                    seen.add((frame, target)); identities.setdefault(target, set()).add(frame)
            require(any(len(frames) >= 2 for frames in identities.values()), "No persistent identity over time in selected GT")
            selected.append({"name": name, "video": str(video), "gt": str(gt), "video_info": video_info,
                             "gt_sha256": file_sha256(gt), "video_sha256": video_hash, "gt_observations": len(seen)})
    return {"dataset": "VisDrone-MOT", "split": split, "sequences": selected,
            "manifest_sha256": file_sha256(manifest_path), "full_evaluator_validation": False}


def run_detection(project_root: Path, config_path: Path, video: Path, output_dir: Path,
                  *, max_frames: int | None = None) -> dict:
    """Use the same production adapter/pipeline as local run_video.py."""
    from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
    from traffic_mot.utils.seed import seed_everything
    from traffic_mot.video.pipeline import VideoPipeline
    check_video_dataset(video)
    seed_everything(42, use_torch=True)
    detector = RTDETRv2P2Detector(config_path, project_root=project_root)
    return VideoPipeline(detector, output_dir=output_dir, save_video=True, save_json=True,
                         max_frames=max_frames).process(video)


def run_stage(project_root: Path, entrypoint: str, arguments: list[str]) -> None:
    """Execute a future shared CLI only when it actually exists; no fake results."""
    path = project_root / "scripts" / entrypoint
    require(path.is_file(), f"Stage not implemented: {entrypoint}. Complete the earlier acceptance gate first.")
    subprocess.run([sys.executable, str(path), *arguments], check=True, cwd=project_root)
