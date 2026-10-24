"""Post-inference GT diagnostics with fixed occlusion episode denominators."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_mot.evaluation.trackeval_runner import load_metrics
from traffic_mot.utils.config import require


def _matches(gt_frames, predictions, classes):
    """Associate all valid GT first; never filter only heavy/small before matching."""
    from trackeval.datasets._base_dataset import _BaseDataset
    require(len(gt_frames) == len(predictions), "Diagnostic timeline mismatch")
    output = []
    for gt, pred in zip(gt_frames, predictions):
        matched = {}
        for cls in classes:
            gs = [g for g in gt if not g["ignored"] and g["class_id"] == cls]
            ps = [p for p in pred if p["class_id"] == cls]
            if not gs or not ps: continue
            similarity = _BaseDataset._calculate_box_ious(np.asarray([g["bbox"] for g in gs], float),
                np.asarray([p["bbox"] for p in ps], float), box_format="x0y0x1y1")
            rows, cols = linear_sum_assignment(-((similarity >= .5) * 1000 + similarity))
            for row, col in zip(rows, cols):
                if similarity[row, col] >= .5: matched[gs[row]["track_id"]] = ps[col]["track_id"]
        output.append(matched)
    return output


def gt_episodes(gt_frames: list[list[dict]], classes: list[int], stable_frames: int) -> list[dict]:
    """Fixed GT episodes: visible pre-window, labeled occlusion, visible post-window.

    Missing/unannotated gaps and truncation are excluded rather than guessed as
    complete occlusion. All required GT observations must be consecutive.
    """
    by_id = defaultdict(dict)
    for frame_id, gt in enumerate(gt_frames, 1):
        for g in gt:
            if not g["ignored"] and g["class_id"] in classes: by_id[g["track_id"]][frame_id] = g
    episodes = []
    for identity, timeline in sorted(by_id.items()):
        ordered = sorted(timeline)
        for start in ordered:
            if timeline[start].get("occlusion") not in {1, 2}: continue
            if start - 1 in timeline and timeline[start - 1].get("occlusion") in {1, 2}: continue
            end = start
            while end + 1 in timeline and timeline[end + 1].get("occlusion") in {1, 2}: end += 1
            pre, post = list(range(start - stable_frames, start)), list(range(end + 1, end + 1 + stable_frames))
            if not all(f in timeline and timeline[f].get("occlusion") == 0 for f in pre + post): continue
            window = list(range(pre[0], post[-1] + 1))
            if not all(f in timeline and timeline[f].get("truncation", 0) == 0 for f in window): continue
            if len({timeline[f]["class_id"] for f in window}) != 1: continue
            episodes.append({"gt_id": int(identity), "start_frame": start, "end_frame": end,
                             "pre_frames": pre, "post_frames": post,
                             "occlusion": "heavy" if any(timeline[f]["occlusion"] == 2 for f in range(start, end + 1)) else "partial"})
    return episodes


def evaluate_diagnostics(sequences: dict[str, tuple[list, list]], classes: list[int], *,
                         recovery_window: int = 10, stable_frames: int = 3) -> dict:
    """GT-bin recall and continuity success, not HOTA-heavy or guaranteed recovery."""
    require(recovery_window >= stable_frames >= 1, "Invalid recovery diagnostic window")
    load_metrics()
    bins = {group: defaultdict(lambda: {"gt": 0, "tp": 0}) for group in ["occlusion", "size", "size_occlusion"]}
    episodes, eligible_gt = [], []
    for name, (gt, predictions) in sequences.items():
        matches = _matches(gt, predictions, classes)
        for index, frame in enumerate(gt):
            for g in frame:
                if g["ignored"] or g["class_id"] not in classes: continue
                area = (g["bbox"][2] - g["bbox"][0]) * (g["bbox"][3] - g["bbox"][1])
                size = "small" if area < 1024 else "medium" if area < 9216 else "large"
                occ = {0: "none", 1: "partial", 2: "heavy"}.get(g.get("occlusion"), "unknown")
                for group, label in [("size", size), ("occlusion", occ), ("size_occlusion", f"{size}_{occ}")]:
                    bins[group][label]["gt"] += 1
                    bins[group][label]["tp"] += int(g["track_id"] in matches[index])
        for episode in gt_episodes(gt, classes, stable_frames):
            fixed = {"sequence": name, **episode}
            eligible_gt.append(fixed)
            identity = episode["gt_id"]
            pre_ids = [matches[f - 1].get(identity) for f in episode["pre_frames"]]
            pre_id = pre_ids[0] if pre_ids[0] is not None and len(set(pre_ids)) == 1 else None
            end = episode["end_frame"]
            last = min(len(gt), end + recovery_window)
            recovery_frame = None
            if pre_id is not None:
                for frame in range(end + 1, last - stable_frames + 2):
                    if all(matches[f - 1].get(identity) == pre_id for f in range(frame, frame + stable_frames)):
                        if not any(other != identity and pred == pre_id for f in range(episode["start_frame"], frame + stable_frames)
                                   for other, pred in matches[f - 1].items()):
                            recovery_frame = frame
                            break
            episodes.append({**fixed, "pre_tracked": pre_id is not None, "pre_prediction_id": pre_id,
                             "success": recovery_frame is not None, "recovery_frame": recovery_frame,
                             "delay_frames": recovery_frame - end if recovery_frame is not None else None})
    for values in bins.values():
        for row in values.values():
            row["fn"] = row["gt"] - row["tp"]
            row["recall"] = row["tp"] / row["gt"] if row["gt"] else None
    denominator = len(episodes)
    successes = sum(e["success"] for e in episodes)
    return {"protocol": "gt_labeled_occlusion_continuity_v1", "matching_iou": .5,
            "strata_matching": "all valid GT in each frame before binning", "bins": dict(bins),
            "episodes_gt_sha256": hashlib.sha256(json.dumps(eligible_gt, sort_keys=True).encode()).hexdigest(),
            "eligible_gt_episodes": denominator, "successful_episodes": successes,
            "pre_tracked_episodes": sum(e["pre_tracked"] for e in episodes),
            "recovery_success_rate": successes / denominator if denominator else None,
            "recovery_window_frames": recovery_window, "stable_frames": stable_frames,
            "episodes": episodes, "fully_occluded_unannotated_gaps_evaluated": False,
            "interpretation": "Same-ID continuity after GT-labeled occlusion; may include uninterrupted tracks; not causal proof of OAR recovery"}
