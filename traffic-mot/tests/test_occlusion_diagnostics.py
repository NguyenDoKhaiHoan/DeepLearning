"""GT-first bins and an identical episode denominator for every experiment."""
import copy
import pytest

pytest.importorskip("trackeval")

from traffic_mot.evaluation.occlusion import evaluate_diagnostics


def g(occ, identity=0, x=0):
    return {"track_id": identity, "class_id": 4, "bbox": [x, 0, x + 20, 20],
            "ignored": False, "occlusion": occ, "truncation": 0}


def p(identity=1, x=0):
    return {"track_id": identity, "class_id": 4, "bbox": [x, 0, x + 20, 20]}


def test_recovery_fixed_gt_denominator_includes_missing_pretrack():
    gt = [[g(occ)] for occ in [0, 0, 0, 2, 2, 0, 0, 0]]
    good = evaluate_diagnostics({"seq": (gt, [[p()]] * 8)}, [4])
    bad = evaluate_diagnostics({"seq": (gt, [[]] * 8)}, [4])
    assert good["eligible_gt_episodes"] == bad["eligible_gt_episodes"] == 1
    assert good["episodes_gt_sha256"] == bad["episodes_gt_sha256"]
    assert good["recovery_success_rate"] == 1 and bad["recovery_success_rate"] == 0
    assert bad["pre_tracked_episodes"] == 0


def test_gt_first_matching_does_not_steal_visible_box_for_heavy_gt():
    gt = [[g(0), g(2, identity=2, x=8)]]
    result = evaluate_diagnostics({"seq": (gt, [[p()]])}, [4])
    assert result["bins"]["occlusion"]["none"]["tp"] == 1
    assert result["bins"]["occlusion"]["heavy"]["tp"] == 0
    assert result["eligible_gt_episodes"] == 0 and result["recovery_success_rate"] is None


def test_identity_switch_is_not_successful_recovery():
    gt = [[g(occ)] for occ in [0, 0, 0, 1, 1, 0, 0, 0]]
    pred = [[p()]] * 3 + [[], []] + [[p(2)]] * 3
    result = evaluate_diagnostics({"seq": (gt, pred)}, [4])
    assert result["eligible_gt_episodes"] == 1 and result["successful_episodes"] == 0


def test_unannotated_or_truncated_gaps_are_excluded():
    gt = [[g(occ)] for occ in [0, 0, 0, 1, 1, 0, 0, 0]]
    gt[3][0]["truncation"] = 1
    assert evaluate_diagnostics({"seq": (gt, [[]] * 8)}, [4])["eligible_gt_episodes"] == 0
    gt[3] = []
    assert evaluate_diagnostics({"seq": (gt, [[]] * 8)}, [4])["eligible_gt_episodes"] == 0
