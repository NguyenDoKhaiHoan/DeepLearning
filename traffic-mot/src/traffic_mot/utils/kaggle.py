"""Shared Kaggle preflight/orchestration; notebooks contain no CV algorithms."""
from __future__ import annotations

import copy
import json
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

    Supports native image sequences (ZIP or directory) and older video entries.
    Paths resolve relative to the manifest. Raw VisDrone GT ID 0 is valid.
    Full ignore policy/class conversion remains a Stage 3 evaluation task.
    """
    require(split in {"validation", "test"}, "Use an explicit validation or test split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("dataset") == "VisDrone-MOT", "Declare a real VisDrone-MOT dataset")
    require(bool(manifest.get("validation")) and bool(manifest.get("test")), "Provide disjoint validation and test sequences")
    base = manifest_path.resolve().parent
    from traffic_mot.utils.mot_data import check_native_manifest, validate_gt
    all_sequences = manifest["validation"] + manifest["test"]
    if any(sequence.get("source_kind") in {"images_zip", "images_dir"} for sequence in all_sequences):
        require(all(sequence.get("source_kind") in {"images_zip", "images_dir"} for sequence in all_sequences),
                "Do not mix native image sequences and legacy video entries in one manifest")
        report = check_native_manifest(manifest, base, split)
        report["manifest_sha256"] = file_sha256(manifest_path)
        return report
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
            stats = validate_gt(gt.read_text(encoding="utf-8-sig"), video_info["frame_count"])
            selected.append({"name": name, "video": str(video), "gt": str(gt), "video_info": video_info,
                             "gt_sha256": file_sha256(gt), "video_sha256": video_hash, **stats})
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
    """Execute a shared CLI from the notebook's own environment, or fail clearly."""
    path = project_root / "scripts" / entrypoint
    require(path.is_file(), f"Stage not implemented: {entrypoint}. Complete the earlier acceptance gate first.")
    subprocess.run([sys.executable, str(path), *arguments], check=True, cwd=project_root)
