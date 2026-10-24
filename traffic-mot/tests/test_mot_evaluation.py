"""Known MOT cases and comparison to the actual toolkit metric implementations."""
import numpy as np
import pytest

pytest.importorskip("trackeval", reason="Install requirements-eval.txt for official metric tests")

from traffic_mot.evaluation.trackeval_runner import VisDroneMOTAdapter, evaluate_sequences, load_metrics, parse_gt
from traffic_mot.utils.config import ConfigurationError


def g(identity=0, cls=4, x=0, ignored=False):
    return {"track_id": identity, "class_id": cls, "bbox": [x, 0, x + 20, 20], "ignored": ignored}


def p(identity=1, cls=4, x=0):
    return {"track_id": identity, "class_id": cls, "bbox": [x, 0, x + 20, 20]}


def evaluate(gt, pred, classes=None):
    return evaluate_sequences({"fixture": (gt, pred)}, classes or [4])


def test_perfect_and_zero_gt_id():
    result = evaluate([[g()], [g()]], [[p()], [p()]])
    summary = result["combined"]
    assert summary["HOTA"] == summary["IDF1"] == summary["MOTA"] == 1
    assert summary["IDSW"] == 0 and summary["MT"] == 1 and summary["ML"] == 0


def test_identity_switch_uses_global_assignment():
    result = evaluate([[g()]] * 4, [[p(1)], [p(1)], [p(2)], [p(2)]])["combined"]
    assert result["IDSW"] == 1 and result["MOTA"] == .75
    assert result["IDF1"] == .5 and result["HOTA"] == pytest.approx(np.sqrt(.5))


def test_missing_fragment_and_false_positive():
    result = evaluate([[g()]] * 3, [[p()], [], [p(), p(2, x=60)]])["combined"]
    assert result["CLR_FN"] == 1 and result["CLR_FP"] == 1
    assert result["MOTA"] == pytest.approx(1 / 3)


def test_upstream_fragmentation_gap_limit_is_disclosed():
    result = evaluate([[g()]] * 3, [[p()], [], [p()]])
    assert result["raw_combined"]["CLEAR"]["Frag"] == 0  # Upstream empty-frame shortcut.
    assert "Frag" not in result["combined"]
    assert any("Frag" in message for message in result["toolkit"]["limitations"])


def test_no_gt_no_pred_and_pred_only():
    result = evaluate([[], []], [[], []])["combined"]
    assert result["CLR_FP"] == result["CLR_FN"] == 0
    result = evaluate([[], []], [[p()], [p()]])["combined"]
    assert result["CLR_FP"] == 2 and result["HOTA"] == 0
    result = evaluate([[g()], [g()]], [[], []])["combined"]
    assert result["ML"] == 1 and result["CLR_FN"] == 2


def test_ignore_does_not_remove_valid_match():
    result = evaluate([[g(), g(identity=-1, cls=0, ignored=True)],
                       [g(x=60), g(identity=-1, cls=0, ignored=True)]],
                      [[p()], [p(x=60), p(2)]])
    assert result["combined"]["MOTA"] == 1
    assert result["counts"]["fixture"]["4"]["ignored_predictions"] == 1


def test_wrong_class_is_false_negative_and_false_positive():
    result = evaluate([[g()]], [[p(cls=5)]], [4, 5])["combined"]
    assert result["CLR_FN"] == 1 and result["CLR_FP"] == 1
    assert result["IDF1"] == 0 and result["MOTA"] == -1


def test_duplicate_ids_and_length_mismatch_fail():
    with pytest.raises(ConfigurationError, match="Duplicate predicted"):
        evaluate([[g()]], [[p(), p(x=60)]])
    with pytest.raises(ConfigurationError, match="timeline length"):
        evaluate([[g()], [g()]], [[p()]])


def test_preprocessing_data_matches_direct_official_metrics():
    metrics, _ = load_metrics()
    data, _ = VisDroneMOTAdapter().prepare([[g()], [g()]], [[p()], []], 4)
    actual = evaluate([[g()], [g()]], [[p()], []])["raw_per_sequence"]["fixture"]["4"]
    for name, metric in metrics.items():
        expected = metric.eval_sequence(data)
        for key in expected: assert np.allclose(actual[name][key], expected[key])
    assert "float" not in np.__dict__  # Compatibility wrapper never mutates NumPy.


def test_combination_uses_sequence_counts_not_mean_scores():
    result = evaluate_sequences({"short": ([[g()]], [[p()]]),
                                 "long": ([[g()]] * 9, [[]] * 9)}, [4])["combined"]
    assert result["CLR_TP"] == 1 and result["CLR_FN"] == 9
    assert result["MOTA"] == pytest.approx(.1)


def test_parse_gt_keeps_raw_zero_and_declared_prefix():
    frames = parse_gt("1,0,0,0,20,20,1,4,0,0\n2,0,0,0,20,20,1,4,0,0\n", 1)
    assert len(frames) == 1 and frames[0][0]["track_id"] == 0
