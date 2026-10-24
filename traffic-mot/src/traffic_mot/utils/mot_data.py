"""Native VisDrone MOT image-sequence manifests, without video conversion.

Preparation reads archive indexes, not held-out GT labels. Selected-split
preflight reads annotations and one real image per sequence. This is data
validation, not an MOT evaluator or tracking implementation.
"""
from __future__ import annotations

from contextlib import ExitStack
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

from traffic_mot.detection.bridge import file_sha256
from traffic_mot.utils.config import require

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _ordered_frames(names: list[str]) -> list[str]:
    """Require an explicit contiguous 1-based timeline in native image names."""
    require(bool(names), "No image frames found")
    require(all(PurePosixPath(name).stem.isdecimal() for name in names), "Frame filenames must be numeric")
    ordered = sorted(names, key=lambda name: int(PurePosixPath(name).stem))
    require([int(PurePosixPath(name).stem) for name in ordered] == list(range(1, len(ordered) + 1)),
            "Image frame IDs must be unique, contiguous and start at 1")
    return ordered


def _zip_index(archive: ZipFile) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate ZIP members are ambiguous")
    for name in names:
        path = PurePosixPath(name)
        if path.suffix.lower() in IMAGE_SUFFIXES and path.parent.parent.name == "sequences":
            groups.setdefault(str(path.parent) + "/", []).append(name)
    require(bool(groups), "No sequences/<name>/<frame>.jpg in archive")
    return {prefix: _ordered_frames(frames) for prefix, frames in sorted(groups.items())}


def _index_fingerprint(archive: ZipFile, frames: list[str]) -> str:
    """Fingerprint filenames, uncompressed sizes and CRCs (not a pixel hash)."""
    values = [(PurePosixPath(name).name, archive.getinfo(name).file_size, archive.getinfo(name).CRC) for name in frames]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _find_release(root: Path, release: str) -> Path:
    candidates = list(root.rglob(f"{release}.zip")) + [
        path for path in root.rglob(release) if path.is_dir() and (path / "sequences").is_dir()]
    require(len(candidates) == 1, f"Expected exactly one {release} ZIP or directory under {root}; found {len(candidates)}")
    return candidates[0]


def _describe_release(source: Path, expected_release: str) -> list[dict]:
    if source.is_file():
        archive_hash = file_sha256(source)
        with ZipFile(source) as archive:
            result = []
            for prefix, frames in _zip_index(archive).items():
                folder = PurePosixPath(prefix.rstrip("/"))
                require(folder.parent.parent.name == expected_release,
                        f"Expected {expected_release} data, not a different release/split")
                gt = str(folder.parent.parent / "annotations" / f"{folder.name}.txt")
                require(gt in archive.namelist(), f"Missing MOT annotation: {gt}")
                result.append({"name": folder.name, "source_kind": "images_zip", "archive": str(source.resolve()),
                               "archive_sha256": archive_hash, "frames_prefix": prefix, "gt_member": gt,
                               "frame_count": len(frames), "frame_index_sha256": _index_fingerprint(archive, frames),
                               "source_video_id": folder.name.split("_")[0], "fps": None})
            return result
    require((source / "sequences").is_dir() and (source / "annotations").is_dir(),
            f"Expected sequences/ and annotations/ under {source}")
    require(source.name == expected_release, f"Expected directory named {expected_release}")
    result = []
    for folder in sorted((source / "sequences").iterdir()):
        if not folder.is_dir(): continue
        frames = _ordered_frames([path.name for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])
        gt = source / "annotations" / f"{folder.name}.txt"
        require(gt.is_file(), f"Missing MOT annotation: {gt}")
        result.append({"name": folder.name, "source_kind": "images_dir", "frames_dir": str(folder.resolve()),
                       "gt": str(gt.resolve()), "frame_count": len(frames),
                       "source_video_id": folder.name.split("_")[0], "fps": None})
    require(bool(result), "No native MOT image sequences found")
    return result


def _check_disjoint(groups: dict[str, list[dict]]) -> None:
    names, sources = set(), {}
    for split in ("validation", "test"):
        for sequence in groups[split]:
            require(sequence["name"] not in names, "Validation/test sequence names overlap")
            names.add(sequence["name"])
            source_id = sequence.get("source_video_id", sequence["name"])
            require(source_id not in sources or sources[source_id] == split,
                    "Validation/test contain segments of the same source UAV video")
            sources[source_id] = split


def prepare_mot_manifest(output: Path, *, root: Path | None = None,
                         validation_source: Path | None = None, test_source: Path | None = None) -> dict:
    """Map official val→validation/test-dev→held-out test, no GT label parsing.

    Accept ZIPs or extracted release directories. Do not fabricate MP4 timing or
    metrics, extract archives, or overwrite an existing manifest.
    """
    if output.exists(): raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    if root is not None:
        require(validation_source is None and test_source is None, "Choose --root or explicit split sources")
        validation_source = _find_release(root, "VisDrone2019-MOT-val")
        test_source = _find_release(root, "VisDrone2019-MOT-test-dev")
    require(validation_source is not None and test_source is not None, "Supply both validation and test-dev sources")
    manifest = {"schema_version": "1.1.0", "dataset": "VisDrone-MOT", "release": "2019",
                "validation": _describe_release(validation_source, "VisDrone2019-MOT-val"),
                "test": _describe_release(test_source, "VisDrone2019-MOT-test-dev"),
                "split_policy": "official val for tuning; official test-dev held out for final evaluation",
                "gt_id_origin": 0, "frame_id_origin": 1, "test_labels_parsed": False,
                "evaluation_performed": False}
    _check_disjoint(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def validate_gt(text: str, frame_count: int | None) -> dict:
    """Validate VisDrone GT where raw target ID 0 is valid (unlike output IDs)."""
    identities: dict[int, set[int]] = {}
    seen, zero_rows, row_count = set(), 0, 0
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        require(len(row) == 10, "Expected 10-column MOT annotation, not VisDrone DET")
        values = [float(value) for value in row]
        require(all(math.isfinite(value) for value in values), "Nonfinite annotation value")
        frame, target, category = int(values[0]), int(values[1]), int(values[7])
        require(values[0] == frame and values[1] == target and frame >= 1, "Invalid GT frame/identity")
        require(frame_count is None or frame <= frame_count, "GT frame exceeds source length")
        require(values[7] == category and 0 <= category <= 11, "Invalid VisDrone category")
        require(values[6] in {0, 1}, "GT score must be evaluation flag 0 or 1")
        row_count += 1
        if values[6] == 0 or category in {0, 11}: continue
        require(target >= 0 and values[4] > 0 and values[5] > 0, "Valid VisDrone GT needs nonnegative persistent ID and positive box")
        require((frame, target) not in seen, "Duplicate frame/identity in MOT GT")
        seen.add((frame, target))
        identities.setdefault(target, set()).add(frame)
        zero_rows += int(target == 0)
    require(any(len(frames) >= 2 for frames in identities.values()), "No persistent identity over time in selected GT")
    return {"gt_observations": len(seen), "gt_rows": row_count, "gt_identities": len(identities),
            "gt_id_zero_observations": zero_rows}


def check_native_manifest(manifest: dict, base: Path, split: str) -> dict:
    """Check native sequences, reading GT/images only for the selected split.

    Archive integrity is hashed for both splits without interpreting held-out
    labels. Only frame-index fingerprints and source IDs are used for overlap
    checks; this is not a complete perceptual near-duplicate/leakage detector.
    """
    _check_disjoint(manifest)
    selected, fingerprints = [], {}
    with ExitStack() as stack:
        archives: dict[Path, tuple[ZipFile, dict[str, list[str]], str]] = {}
        for group in ("validation", "test"):
            for sequence in manifest[group]:
                kind = sequence["source_kind"]
                if kind == "images_zip":
                    path = (base / sequence["archive"]).resolve()
                    require(path.is_file(), f"Missing MOT archive: {path}")
                    if path not in archives:
                        archive = stack.enter_context(ZipFile(path))
                        archives[path] = archive, _zip_index(archive), file_sha256(path)
                    archive, index, digest = archives[path]
                    require(digest == sequence["archive_sha256"], f"MOT archive checksum mismatch: {path.name}")
                    frames = index.get(sequence["frames_prefix"], [])
                    require(bool(frames), "Sequence image prefix missing in ZIP")
                    require(sequence["gt_member"] in archive.namelist(), "Missing MOT GT member")
                    fingerprint = _index_fingerprint(archive, frames)
                    require(fingerprint == sequence["frame_index_sha256"], "Image frame index changed")
                    gt_reference = f"{path}!/{sequence['gt_member']}"
                elif kind == "images_dir":
                    folder, gt = (base / sequence["frames_dir"]).resolve(), (base / sequence["gt"]).resolve()
                    require(folder.is_dir() and gt.is_file(), "Missing image directory or MOT GT")
                    frames = _ordered_frames([p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
                    # Hash real image contents; do not parse GT from the other split.
                    fingerprint = hashlib.sha256("".join(file_sha256(folder / name) for name in frames).encode()).hexdigest()
                    gt_reference = str(gt)
                else:
                    raise ValueError(f"Unsupported native MOT source_kind: {kind}")
                require(len(frames) == sequence["frame_count"], "Image frame count differs from manifest")
                require(fingerprint not in fingerprints or fingerprints[fingerprint] == group,
                        "Validation/test contain duplicate image sequences")
                fingerprints[fingerprint] = group
                if group != split: continue
                if kind == "images_zip":
                    raw_gt, raw_image = archive.read(sequence["gt_member"]), archive.read(frames[0])
                else:
                    raw_gt, raw_image = gt.read_bytes(), (folder / frames[0]).read_bytes()
                with Image.open(io.BytesIO(raw_image)) as image:
                    image.load()
                    width, height = image.size
                stats = validate_gt(raw_gt.decode("utf-8-sig"), len(frames))
                selected.append({"name": sequence["name"], "source_kind": kind, "frame_count": len(frames),
                                 "width": width, "height": height, "fps": sequence.get("fps"),
                                 "gt": gt_reference, "gt_sha256": hashlib.sha256(raw_gt).hexdigest(), **stats})
    return {"dataset": "VisDrone-MOT", "split": split, "sequences": selected,
            "full_evaluator_validation": False, "evaluation_performed": False,
            "raw_gt_id_zero_allowed": True}
