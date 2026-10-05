#!/usr/bin/env python3
"""
Prepare VisDrone2019-DET for RT-DETR training on Kaggle.

Supported input layouts
-----------------------

Official extracted layout:

  <root>/
    VisDrone2019-DET-train/
      images/*.jpg
      annotations/*.txt
    VisDrone2019-DET-val/
      images/*.jpg
      annotations/*.txt

Kaggle layout with an extra VisDrone2019 directory:

  <root>/
    VisDrone2019/
      VisDrone2019-DET-train/
        images/
        annotations/
      VisDrone2019-DET-val/
        images/
        annotations/

Kaggle layout with duplicated split directories:

  <root>/
    VisDrone2019/
      VisDrone2019-DET-train/
        VisDrone2019-DET-train/
          images/
          annotations/
      VisDrone2019-DET-val/
        VisDrone2019-DET-val/
          images/
          annotations/

The script also performs a recursive fallback search for the split folders.

VisDrone annotation format
--------------------------

Each row is:

  bbox_left,bbox_top,bbox_width,bbox_height,score,
  object_category,truncation,occlusion

Classes 1..10 are retained.

Category 0 (ignored regions) and category 11 (others) are excluded because
they are not part of the standard 10-class VisDrone detection label set used
by most RT-DETR recipes.

Output
------

The output directory contains:

  instances_train.json
  instances_val.json
  bad_images_train.json
  bad_images_val.json
  dataset_stats.json
  classes.txt
  rtdetr_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is required. Run: pip install -q pillow", file=sys.stderr)
    raise


CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

CLASS_ID = {i + 1: name for i, name in enumerate(CLASSES)}

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def is_valid_split_dir(path: Path) -> bool:
    """Return True when a directory contains both images/ and annotations/."""
    return (
        path.is_dir()
        and (path / "images").is_dir()
        and (path / "annotations").is_dir()
    )


def find_split(root: Path, split: str) -> Path:
    """
    Find a VisDrone split directory.

    This supports both the normal VisDrone layout and the nested Kaggle
    structure visible in datasets such as:

      VisDrone2019/
        VisDrone2019-DET-train/
          VisDrone2019-DET-train/
            images/
            annotations/
    """
    root = root.expanduser().resolve()

    split_name = f"VisDrone2019-DET-{split}"

    # Most common layouts first.
    candidates = [
        # Official layout
        root / split_name,

        # Some archives may preserve capitalization differently.
        root / f"VisDrone2019-DET-{split.title()}",

        # Simple train/val directory.
        root / split,

        # Kaggle dataset containing an outer VisDrone2019 directory.
        root / "VisDrone2019" / split_name,

        # Your current Kaggle structure:
        # VisDrone2019/VisDrone2019-DET-train/VisDrone2019-DET-train/
        root / "VisDrone2019" / split_name / split_name,

        # Duplicated split folder without outer VisDrone2019.
        root / split_name / split_name,

        # If --input already points at VisDrone2019.
        root / split_name / split_name,
    ]

    # Remove duplicate paths while preserving order.
    unique_candidates = []
    seen = set()

    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)

    for candidate in unique_candidates:
        if is_valid_split_dir(candidate):
            print(f"[OK] Found {split}: {candidate}")
            return candidate

    # Recursive fallback.
    #
    # This makes the script resilient to extra archive/Kaggle folder levels.
    recursive_matches = []

    if root.exists():
        try:
            for candidate in root.rglob(split_name):
                if is_valid_split_dir(candidate):
                    recursive_matches.append(candidate)

                # Handle duplicated folder:
                # .../VisDrone2019-DET-train/VisDrone2019-DET-train/
                nested = candidate / split_name
                if is_valid_split_dir(nested):
                    recursive_matches.append(nested)
        except PermissionError:
            pass

    # De-duplicate fallback matches.
    recursive_matches = sorted(
        {p.resolve() for p in recursive_matches},
        key=lambda p: (len(p.parts), str(p)),
    )

    if recursive_matches:
        selected = recursive_matches[0]
        print(f"[OK] Found {split} recursively: {selected}")

        if len(recursive_matches) > 1:
            print(
                f"[WARN] Multiple '{split}' directories found. "
                f"Using the first one: {selected}"
            )
            for other in recursive_matches[1:]:
                print(f"       alternative: {other}")

        return selected

    tried = "\n".join(f"  - {p}" for p in unique_candidates)

    raise FileNotFoundError(
        f"\nCannot find VisDrone '{split}' split under:\n"
        f"  {root}\n\n"
        f"The script requires a directory containing BOTH:\n"
        f"  images/\n"
        f"  annotations/\n\n"
        f"Paths checked:\n{tried}\n"
    )


def read_rows(path: Path):
    """Read one VisDrone annotation text file."""
    if not path.exists():
        return []

    rows = []

    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines(),
        1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = [x.strip() for x in line.split(",")]

        if len(parts) < 8:
            print(
                f"WARN {path}:{line_no}: "
                f"expected at least 8 comma-separated fields, got {len(parts)}"
            )
            continue

        try:
            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            cls = int(float(parts[5]))
            trunc = int(float(parts[6]))
            occ = int(float(parts[7]))
        except ValueError:
            print(f"WARN {path}:{line_no}: malformed numeric row")
            continue

        rows.append((x, y, w, h, score, cls, trunc, occ))

    return rows


def clamp_box(x, y, w, h, width, height):
    """Clamp a bounding box to the image boundaries."""
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))

    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))

    return (
        x1,
        y1,
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def process_split(
    split_dir: Path,
    split: str,
    out_dir: Path,
    stats: dict,
):
    """Convert one VisDrone split to COCO detection JSON."""
    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    print()
    print("=" * 80)
    print(f"Processing split : {split}")
    print(f"Split directory  : {split_dir}")
    print(f"Images directory : {images_dir}")
    print(f"Labels directory : {ann_dir}")
    print("=" * 80)

    coco = {
        "info": {
            "description": f"VisDrone2019-DET-{split} prepared for RT-DETR"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": i,
                "name": name,
                "supercategory": "vehicle/person",
            }
            for i, name in CLASS_ID.items()
        ],
    }

    image_files = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    )

    if not image_files:
        raise RuntimeError(
            f"No images found in:\n  {images_dir}"
        )

    print(f"Found {len(image_files)} image files.")

    ann_id = 1
    bad_images = []
    missing_annotations = 0

    # COCO image IDs must correspond only to successfully loaded images.
    next_image_id = 1

    for index, image_path in enumerate(image_files, 1):
        if index == 1 or index % 500 == 0 or index == len(image_files):
            print(
                f"[{split}] image {index}/{len(image_files)} "
                f"({index / len(image_files) * 100:.1f}%)"
            )

        try:
            with Image.open(image_path) as im:
                width, height = im.size
                im.verify()
        except Exception as exc:
            bad_images.append(
                {
                    "file": str(image_path),
                    "error": repr(exc),
                }
            )
            continue

        image_id = next_image_id
        next_image_id += 1

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        annotation_path = ann_dir / f"{image_path.stem}.txt"

        if not annotation_path.exists():
            missing_annotations += 1

        rows = read_rows(annotation_path)

        for x, y, w, h, score, cls, trunc, occ in rows:
            stats["raw_rows"] += 1
            stats["raw_classes"][str(cls)] += 1

            # VisDrone:
            # class 0  = ignored region
            # class 11 = others
            # score 0  = ignored sample
            if (
                score == 0
                or cls not in CLASS_ID
                or w <= 0
                or h <= 0
            ):
                stats["ignored_rows"] += 1
                continue

            x, y, w, h = clamp_box(
                x,
                y,
                w,
                h,
                width,
                height,
            )

            area = w * h

            if area <= 0:
                stats["ignored_rows"] += 1
                continue

            if area < 32 * 32:
                size = "small"
            elif area < 96 * 96:
                size = "medium"
            else:
                size = "large"

            occ_name = {
                0: "none",
                1: "partial",
                2: "heavy",
            }.get(occ, "unknown")

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [
                        round(x, 3),
                        round(y, 3),
                        round(w, 3),
                        round(h, 3),
                    ],
                    "area": round(area, 3),
                    "iscrowd": 0,
                    "segmentation": [],
                    "ignore": 0,
                    "truncation": trunc,
                    "occlusion": occ,
                }
            )

            ann_id += 1

            stats["kept_rows"] += 1
            stats["classes"][CLASS_ID[cls]] += 1
            stats["sizes"][size] += 1
            stats["occlusion"][occ_name] += 1
            stats["per_split"][split]["annotations"] += 1

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    instances_path = out_dir / f"instances_{split}.json"
    bad_images_path = out_dir / f"bad_images_{split}.json"

    instances_path.write_text(
        json.dumps(
            coco,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bad_images_path.write_text(
        json.dumps(
            bad_images,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats["per_split"][split]["images"] = len(coco["images"])
    stats["per_split"][split]["bad_images"] = len(bad_images)
    stats["per_split"][split]["missing_annotations"] = missing_annotations

    print(
        f"[DONE] {split}: "
        f"{len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations, "
        f"{len(bad_images)} bad images, "
        f"{missing_annotations} missing annotation files"
    )

    print(f"COCO JSON: {instances_path}")

    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone2019-DET as COCO JSON for RT-DETR."
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Dataset root. The script automatically detects official and "
            "nested Kaggle VisDrone directory layouts."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory, for example "
            "/kaggle/working/visdrone_coco"
        ),
    )

    args = parser.parse_args()

    root = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    print(f"Input root : {root}")
    print(f"Output dir : {out}")

    if not root.exists():
        raise FileNotFoundError(
            f"Input root does not exist:\n  {root}"
        )

    # Resolve once so we do not repeatedly search the entire dataset.
    train_dir = find_split(root, "train")
    val_dir = find_split(root, "val")

    stats = {
        "classes": Counter(),
        "raw_classes": Counter(),
        "sizes": Counter(),
        "occlusion": Counter(),
        "raw_rows": 0,
        "kept_rows": 0,
        "ignored_rows": 0,
        "per_split": defaultdict(
            lambda: {
                "images": 0,
                "annotations": 0,
                "bad_images": 0,
                "missing_annotations": 0,
            }
        ),
    }

    process_split(
        train_dir,
        "train",
        out,
        stats,
    )

    process_split(
        val_dir,
        "val",
        out,
        stats,
    )

    # Convert defaultdict / Counter objects into plain JSON-compatible dicts.
    stats = json.loads(
        json.dumps(
            stats,
            default=dict,
        )
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "dataset_stats.json").write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (out / "classes.txt").write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8",
    )

    config = {
        "dataset_root": str(out),
        "train_annotations": str(out / "instances_train.json"),
        "val_annotations": str(out / "instances_val.json"),
        "train_images": str((train_dir / "images").resolve()),
        "val_images": str((val_dir / "images").resolve()),
        "num_classes": len(CLASSES),
        "class_names": CLASSES,
    }

    (out / "rtdetr_dataset.json").write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()
    print(f"Prepared COCO files in: {out}")
    print(f"Train images: {train_dir / 'images'}")
    print(f"Val images  : {val_dir / 'images'}")


if __name__ == "__main__":
    main()
