"""Bounded retention and conservative unmatched-detection recovery on stock hooks."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_mot.occlusion.evidence import EvidenceHistory, overlap_ioa, suspected_exit
from traffic_mot.tracking.backend_bridge import UltralyticsTracker
from traffic_mot.utils.config import require

OAR_VERSION = "motion_overlap_v1"


def validate_policy(profile: dict, track_buffer: int | None = None) -> dict:
    """Reject unsupported claims/modes; all gates are inference-only hypotheses."""
    p = copy.deepcopy(profile)
    require(p.get("version") == OAR_VERSION, "Unsupported OAR version")
    require(p.get("enabled") is True, "Recovery tracker requires enabled=true")
    e, r, a = p["evidence"], p["retention"], p["reassociation"]
    require(0 < e["overlap_ioa_threshold"] <= 1 and 1 <= e["min_overlap_frames"] <= e["history_frames"], "Invalid occlusion evidence window")
    require(0 <= e["exit_border_fraction"] < .5, "Invalid exit border fraction")
    require(r["max_frames"] >= r["base_frames"] > 0 and r["max_gap_ms"] > 0, "Invalid bounded retention")
    if track_buffer is not None:
        require(r["base_frames"] == track_buffer, "OAR base_frames must match the stock track_buffer for a fair ablation")
    require(a["class_gate"] is True and not a["use_appearance"], "OAR v1 requires class gating; no ReID extractor is integrated")
    require(a["min_hits"] >= 1 and a["max_candidates"] >= 1 and a["mahalanobis_squared_gate"] > 0,
            "Invalid recovery candidate/Kalman gates")
    require(0 < a["min_predicted_iou"] <= 1 and 0 < a["min_detection_confidence"] <= 1
            and 0 <= a["ambiguity_margin"] <= 1, "Invalid recovery matching thresholds")
    require(not p["output"]["emit_unobserved_as_detection"], "Lost predictions must not be emitted as detections")
    require(p.get("max_position_std_box_ratio", 2.0) > 0, "Covariance must be bounded")
    return p


def conservative_assignment(costs: np.ndarray, margin: float) -> list[tuple[int, int]]:
    """One-to-one solver with mutual best and row/column ambiguity rejection."""
    if costs.size == 0: return []
    finite = np.isfinite(costs)
    rows, cols = linear_sum_assignment(np.where(finite, costs, 1e6))
    matches = []
    for row, col in zip(rows, cols):
        if not finite[row, col]: continue
        row_alternatives, col_alternatives = np.delete(costs[row], col), np.delete(costs[:, col], row)
        alternative = min(np.min(row_alternatives, initial=np.inf), np.min(col_alternatives, initial=np.inf))
        if alternative - costs[row, col] >= margin:
            matches.append((int(row), int(col)))
    return matches


class OcclusionAwareMixin:
    """Extend lifecycle/recovery only; stock predict/high/low/GMC run exactly once."""

    def __init__(self, args, policy):
        self.oar = validate_policy(policy, args.track_buffer)
        self.history: dict[int, EvidenceHistory] = {}
        self.decision_counts = Counter()
        self.last_decisions = []
        self.oar_seconds = 0.0
        self.timestamp_ms = None
        self.image_shape = None
        super().__init__(args)

    def _record(self, event: str, track_id: int, **values):
        self.decision_counts[event] += 1
        # Stream a bounded per-frame trace; counts remain complete.
        if len(self.last_decisions) < 128:
            self.last_decisions.append({"event": event, "track_id": int(track_id), "frame_id": self.frame_id, **values})
        else:
            self.decision_counts["trace_rows_truncated"] += 1

    def update(self, results, img=None, feats=None, **kwargs):
        start = time.perf_counter()
        self.last_decisions = []
        self.timestamp_ms = kwargs.pop("timestamp_ms", None)
        self.image_shape = img.shape if img is not None else None
        self.oar_seconds += time.perf_counter() - start
        output = super().update(results, img, feats, **kwargs)
        start = time.perf_counter()
        active_ids = set()
        for track in self.tracked_stracks + self.lost_stracks:
            if track.state == 3: continue  # Removed, using pinned TrackState values.
            active_ids.add(track.track_id)
            if track.frame_id == self.frame_id and track.is_activated:
                history = self.history.setdefault(track.track_id, EvidenceHistory())
                history.hits += 1
                history.last_timestamp = self.timestamp_ms
        self.history = {key: value for key, value in self.history.items() if key in active_ids}
        self.oar_seconds += time.perf_counter() - start
        return output

    def _pre_first_associate(self, pool, unconfirmed, img, results_high):
        # Stock hook performs GMC on track means/covariances. All OAR geometry is
        # evaluated afterwards in the same current-frame coordinate system.
        super()._pre_first_associate(pool, unconfirmed, img, results_high)
        start = time.perf_counter()
        # Extended tracks must expire BEFORE another stock high-score match can
        # revive them beyond the extension bounds. Stock E0/E1 are unaffected.
        if self.oar["retention"]["adaptive"]:
            expired = [t for t in pool if t.state == 2 and self.frame_id - t.end_frame > self.max_frames_lost
                       and not self._eligible(t)]
            expired_ids = {t.track_id for t in expired}
            for track in expired:
                track.mark_removed()
                self.removed_stracks.append(track)
                self._record("extended_track_expired", track.track_id)
            pool[:] = [t for t in pool if t.track_id not in expired_ids]
            self.lost_stracks = [t for t in self.lost_stracks if t.track_id not in expired_ids]
        evidence = self.oar["evidence"]
        # Only recent observed tracks can act as occluders, not long-lived ghosts.
        recent = [t for t in pool if t.is_activated and self.frame_id - t.end_frame <= 1]
        for track in recent:
            history = self.history.setdefault(track.track_id, EvidenceHistory())
            other_boxes = np.asarray([t.xyxy for t in recent if t.track_id != track.track_id], float).reshape(-1, 4)
            ioa = overlap_ioa(track.xyxy, other_boxes)
            history.add_overlap(self.frame_id, ioa >= evidence["overlap_ioa_threshold"], evidence["history_frames"])
            if self.image_shape is not None:
                history.exit_suspected = suspected_exit(track.xyxy, track.mean[4:6], self.image_shape,
                    evidence["exit_border_fraction"], evidence["exit_outward_motion_required"])
            if history.exit_suspected: self._record("suspected_exit", track.track_id)
        self.oar_seconds += time.perf_counter() - start

    def _eligible(self, track) -> bool:
        history = self.history.get(track.track_id)
        e, r, a = self.oar["evidence"], self.oar["retention"], self.oar["reassociation"]
        gap = self.frame_id - track.end_frame
        limit = r["max_frames"] if r["adaptive"] else r["base_frames"]
        if (history is None or not track.is_activated or history.hits < a["min_hits"] or
                history.exit_suspected or gap <= 0 or gap > limit or
                not history.occlusion_evidence(track.end_frame, e["history_frames"], e["min_overlap_frames"])):
            return False
        if self.timestamp_ms is not None and history.last_timestamp is not None:
            if self.timestamp_ms - history.last_timestamp > r["max_gap_ms"]: return False
        if self.image_shape is not None and suspected_exit(track.xyxy, track.mean[4:6], self.image_shape,
                e["exit_border_fraction"], e["exit_outward_motion_required"]): return False
        if not np.isfinite(track.mean).all() or not np.isfinite(track.covariance).all(): return False
        size = np.maximum(track.xyxy[2:] - track.xyxy[:2], 1)
        std = np.sqrt(np.maximum(np.diag(track.covariance)[:2], 0))
        return bool((std / size <= self.oar.get("max_position_std_box_ratio", 2.0)).all())

    def _init_new_tracks(self, unmatched, detections, activated, refind=None):
        start = time.perf_counter()
        a = self.oar["reassociation"]
        used = set()
        if a["enabled"]:
            # This hook runs AFTER stock high/low/unconfirmed matches. No live
            # match is stolen, no second Kalman predict, and no ID is born yet.
            candidates = sorted([t for t in self.lost_stracks if t.state == 2 and self._eligible(t)],
                                key=lambda t: (self.frame_id - t.end_frame, t.track_id))[:a["max_candidates"]]
            indices = sorted([i for i in unmatched if detections[i].score >= a["min_detection_confidence"]],
                             key=lambda i: (-detections[i].score, i))[:a["max_candidates"]]
            costs = np.full((len(candidates), len(indices)), np.inf)
            for row, track in enumerate(candidates):
                for col, index in enumerate(indices):
                    det = detections[index]
                    if int(track.cls) != int(det.cls): continue
                    left, right = track.xyxy, det.xyxy
                    inter = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2])).prod()
                    union = np.maximum(0, left[2:] - left[:2]).prod() + np.maximum(0, right[2:] - right[:2]).prod() - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou < a["min_predicted_iou"]: continue
                    try:
                        measurement = track.convert_coords(det.tlwh)[None]
                        distance = float(track.kalman_filter.gating_distance(track.mean, track.covariance, measurement)[0])
                    except np.linalg.LinAlgError:
                        self._record("invalid_covariance", track.track_id)
                        continue
                    if not np.isfinite(distance) or distance > a["mahalanobis_squared_gate"]: continue
                    costs[row, col] = .7 * (1 - iou) + .3 * distance / a["mahalanobis_squared_gate"]
            matches = conservative_assignment(costs, a["ambiguity_margin"])
            for row, col in matches:
                track, index = candidates[row], indices[col]
                gap = self.frame_id - track.end_frame
                track.re_activate(detections[index], self.frame_id, new_id=False)
                (refind if refind is not None else activated).append(track)
                used.add(index)
                self._record("reassociated", track.track_id, gap_frames=gap, cost=float(costs[row, col]),
                             detection_index=int(detections[index].idx))
            self.decision_counts["eligible_reassociation_tracks"] += len(candidates)
            self.decision_counts["gated_candidate_pairs"] += int(np.isfinite(costs).sum())
            self.decision_counts["ambiguous_or_competing_pairs"] += max(0, int(np.isfinite(costs).sum()) - len(matches))
        self.oar_seconds += time.perf_counter() - start
        super()._init_new_tracks([i for i in unmatched if i not in used], detections, activated, refind)

    def _remove_stale_lost(self, removed):
        start = time.perf_counter()
        for track in self.lost_stracks:
            gap = self.frame_id - track.end_frame
            if gap <= self.max_frames_lost: continue
            if self.oar["retention"]["adaptive"] and track.state == 2 and self._eligible(track):
                self._record("retained_beyond_base", track.track_id, gap_frames=gap)
            else:
                track.mark_removed()
                removed.append(track)
                self._record("removed_after_gap", track.track_id, gap_frames=gap)
        self.oar_seconds += time.perf_counter() - start

    def reset(self):
        super().reset()
        self.history.clear()
        self.decision_counts.clear()
        self.last_decisions = []
        self.oar_seconds = 0.0


class RecoveryTracker(UltralyticsTracker):
    """ByteTrack or BoT-SORT with explicit versioned OAR policy, no GT input."""
    def __init__(self, config: Path, policy: dict):
        self.policy = validate_policy(policy)
        super().__init__(config)

    def _create_backend(self):
        cls = type(f"OcclusionAware{self.backend_class.__name__}", (OcclusionAwareMixin, self.backend_class), {})
        return cls(self.args, self.policy)

    def diagnostics(self) -> dict:
        return {"version": OAR_VERSION, "counts": dict(self.backend.decision_counts),
                "oar_seconds": self.backend.oar_seconds, "history_tracks": len(self.backend.history),
                "milliseconds_cap_enforced": self.backend.timestamp_ms is not None}

    def metadata(self) -> dict:
        return {**super().metadata(), "recovery_enabled": True, "oar_version": OAR_VERSION, "oar_policy": self.policy}
