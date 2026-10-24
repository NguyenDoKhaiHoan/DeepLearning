"""Official TrackEval metrics over explicit VisDrone research preprocessing.

This is a multiclass research protocol, not a claim of VisDrone challenge
equivalence. No GT is exposed to inference. HOTA/IDF1/CLEAR are never reimplemented.
"""
from __future__ import annotations

import csv
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_mot.detection.base import VISDRONE_CLASSES
from traffic_mot.utils.config import require

TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
PROTOCOL = "visdrone_vehicle_research_v1"


class _NumpyCompatibility:
    """Module-local legacy dtype names; never change global numpy attributes."""
    def __getattr__(self, name):
        if name == "float": return float
        if name == "int": return int
        return getattr(np, name)


def load_metrics() -> tuple[dict, dict]:
    """Require the pinned upstream install, and bridge NumPy-removed dtype names."""
    try: package = distribution("trackeval")
    except PackageNotFoundError as exc:
        raise RuntimeError("TrackEval missing; install traffic-mot/requirements-eval.txt in the notebook kernel") from exc
    url = json.loads(package.read_text("direct_url.json") or "{}")
    require(url.get("vcs_info", {}).get("commit_id") == TRACKEVAL_COMMIT,
            "TrackEval must be installed from the pinned requirements-eval.txt Git commit")
    for name in ("hota", "identity"):
        import_module(f"trackeval.metrics.{name}").np = _NumpyCompatibility()
    from trackeval.metrics import HOTA, CLEAR, Identity
    return {"HOTA": HOTA(), "CLEAR": CLEAR({"PRINT_CONFIG": False}),
            "Identity": Identity({"PRINT_CONFIG": False})}, {
                "name": "TrackEval", "version": package.version, "commit": TRACKEVAL_COMMIT,
                "numpy_compatibility": "module-local float/int dtype aliases only",
                "limitations": ["Pinned CLEAR can undercount Frag across frames with zero predictions; raw Frag retained but omitted from summary"]}


def parse_gt(text: str, frame_count: int) -> list[list[dict]]:
    """Read original frame/ID/class/ignore data, preserving raw identity zero."""
    frames = [[] for _ in range(frame_count)]
    seen = set()
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        require(len(row) == 10, "Expected VisDrone MOT 10-column GT")
        v = [float(x) for x in row]
        require(all(np.isfinite(v)), "Nonfinite GT")
        f, target, category = int(v[0]), int(v[1]), int(v[7])
        require(v[0] == f and v[1] == target and v[7] == category and f >= 1 and 0 <= category <= 11,
                "Invalid GT frame/identity/class")
        require(v[6] in {0, 1} and v[4] > 0 and v[5] > 0, "Invalid GT box/evaluation flag")
        # Prefix smoke evaluates a declared temporal subset; never pads/cuts predictions silently.
        if f > frame_count: continue
        ignored = v[6] == 0 or category in {0, 11}
        if not ignored:
            require(target >= 0 and (f, target) not in seen, "Duplicate or negative valid GT identity")
            seen.add((f, target))
        frames[f - 1].append({"track_id": target, "class_id": category,
            "bbox": [v[2], v[3], v[2] + v[4], v[3] + v[5]], "ignored": ignored,
            "occlusion": v[9], "truncation": v[8]})
    return frames


class VisDroneMOTAdapter:
    """Prepare official metric data for a single class over the full timeline.

    Protect predictions assigned to valid GT at IoU>=.5 first; only unmatched
    predictions with IoA>=.5 in explicit ignore boxes are suppressed. Other
    classes are not silently treated as ignore. IDs remap once per sequence/class.
    """

    def __init__(self):
        from trackeval.datasets._base_dataset import _BaseDataset
        self.iou = _BaseDataset._calculate_box_ious

    def prepare(self, gt_frames: list[list[dict]], predictions: list[list[dict]], class_id: int) -> tuple[dict, dict]:
        require(len(gt_frames) == len(predictions) and len(gt_frames) > 0, "GT/prediction timeline length mismatch or empty")
        data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [],
                "num_timesteps": len(gt_frames), "num_gt_dets": 0, "num_tracker_dets": 0}
        gt_map, pred_map, suppressed = {}, {}, 0
        for gt, pred in zip(gt_frames, predictions):
            require(len({p["track_id"] for p in pred}) == len(pred), "Duplicate predicted identity in frame")
            for p in pred:
                box = p["bbox"]
                require(p.get("observed", True) and p.get("track_state", "tracked") == "tracked",
                        "MOT scoring accepts confirmed observed predictions only")
                require(isinstance(p["track_id"], int) and p["track_id"] >= 1 and
                        p["class_id"] in range(3, 11) and len(box) == 4 and np.isfinite(box).all() and
                        box[2] > box[0] and box[3] > box[1], "Invalid prediction identity/class/box")
            valid = [g for g in gt if not g["ignored"] and g["class_id"] == class_id]
            ignored = np.asarray([g["bbox"] for g in gt if g["ignored"]], float).reshape(-1, 4)
            observed = [p for p in pred if p["class_id"] == class_id]
            gbox = np.asarray([g["bbox"] for g in valid], float).reshape(-1, 4)
            pbox = np.asarray([p["bbox"] for p in observed], float).reshape(-1, 4)
            similarity = self.iou(gbox, pbox, box_format="x0y0x1y1")
            protected = np.zeros(len(observed), bool)
            if len(valid) and len(observed):
                # Maximize valid-match cardinality before IoU to protect real detections.
                rows, cols = linear_sum_assignment(-((similarity >= .5) * 1000 + similarity))
                protected[cols[similarity[rows, cols] >= .5]] = True
            remove = np.zeros(len(observed), bool)
            if len(ignored) and len(observed):
                ioa = self.iou(pbox, ignored, box_format="x0y0x1y1", do_ioa=True)
                remove = (~protected) & (ioa.max(axis=1) >= .5)
            suppressed += int(remove.sum())
            kept = [p for p, drop in zip(observed, remove) if not drop]
            gids, pids = [], []
            for g in valid:
                raw = g["track_id"]
                require(raw >= 0, "Raw GT ID zero is valid, negative ID is not")
                if raw not in gt_map: gt_map[raw] = len(gt_map)
                gids.append(gt_map[raw])
            for p in kept:
                raw = p["track_id"]
                if raw not in pred_map: pred_map[raw] = len(pred_map)
                pids.append(pred_map[raw])
            require(len(set(gids)) == len(gids), "Duplicate GT identity in frame")
            data["gt_ids"].append(np.asarray(gids, dtype=int))
            data["tracker_ids"].append(np.asarray(pids, dtype=int))
            data["similarity_scores"].append(similarity[:, ~remove])
            data["num_gt_dets"] += len(gids)
            data["num_tracker_dets"] += len(pids)
        data.update(num_gt_ids=len(gt_map), num_tracker_ids=len(pred_map))
        return data, {"gt_observations": data["num_gt_dets"], "gt_identities": len(gt_map),
                      "predicted_observations": data["num_tracker_dets"], "ignored_predictions": suppressed}


def jsonable(value):
    """Serialize raw toolkit arrays/counts without rounding or inventing NaNs."""
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    return value


def summarize(result: dict) -> dict:
    """Ratios remain 0..1; IDSW/MT/ML/FP/FN are counts, MOTA may be negative."""
    return {"HOTA": float(np.mean(result["HOTA"]["HOTA"])),
            "DetA": float(np.mean(result["HOTA"]["DetA"])), "AssA": float(np.mean(result["HOTA"]["AssA"])),
            "IDF1": float(result["Identity"]["IDF1"]),
            **{key: jsonable(result["CLEAR"][key]) for key in ["MOTA", "IDSW", "MT", "ML", "CLR_TP", "CLR_FP", "CLR_FN"]}}


def evaluate_sequences(sequences: dict[str, tuple[list, list]], classes: list[int]) -> dict:
    """Evaluate and aggregate with official toolkit combination functions."""
    metrics, provenance = load_metrics()
    adapter = VisDroneMOTAdapter()
    per_sequence, counts = {}, {}
    for name, (gt, pred) in sequences.items():
        per_sequence[name], counts[name] = {}, {}
        for cls in classes:
            data, statistics = adapter.prepare(gt, pred, cls)
            per_sequence[name][cls] = {name: metric.eval_sequence(data) for name, metric in metrics.items()}
            counts[name][cls] = statistics
    require(bool(per_sequence), "No sequences to evaluate")
    combined_classes = {cls: {name: metric.combine_sequences({seq: result[cls][name] for seq, result in per_sequence.items()})
                             for name, metric in metrics.items()} for cls in classes}
    combined = {name: metric.combine_classes_det_averaged({cls: values[name] for cls, values in combined_classes.items()})
                for name, metric in metrics.items()}
    return jsonable({"protocol": PROTOCOL, "toolkit": provenance, "metric_ratio_scale": "0..1 (MOTA may be negative)",
                     "class_ids": classes, "combined": summarize(combined),
                     "per_class": {VISDRONE_CLASSES[cls]: summarize(values) for cls, values in combined_classes.items()},
                     "per_sequence": {seq: {VISDRONE_CLASSES[cls]: summarize(r) for cls, r in values.items()}
                                      for seq, values in per_sequence.items()},
                     "counts": counts, "raw_combined": combined, "raw_per_class": combined_classes,
                     "raw_per_sequence": per_sequence})


def write_metrics(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
