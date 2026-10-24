"""OAR invariants and synthetic recovery probes, never experimental evidence."""
from pathlib import Path
import copy

import numpy as np
import pytest
import yaml

pytest.importorskip("lap")

from traffic_mot.occlusion.evidence import overlap_ioa, suspected_exit
from traffic_mot.occlusion.recovery import RecoveryTracker, conservative_assignment
from traffic_mot.tracking.backend_bridge import UltralyticsTracker
from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.types import DetectionBatch

ROOT = Path(__file__).resolve().parents[1]
FRAME = np.zeros((160, 200, 3), np.uint8)


def detection(a=True, x=40, b=45, cls=4):
    rows = ([([x, 40, x + 20, 60], .9, cls)] if a else []) + [([b, 40, b + 20, 60], .95, 9)]
    return DetectionBatch(np.asarray([r[0] for r in rows], np.float32).reshape(-1, 4),
                          np.asarray([r[1] for r in rows], np.float32), np.asarray([r[2] for r in rows], np.int64),
                          np.arange(len(rows)))


def policy(**overrides):
    p = yaml.safe_load((ROOT / "configs/occlusion.yaml").read_text())
    p["enabled"] = True
    for key, value in overrides.items(): p[key].update(value)
    return p


@pytest.fixture(params=["bytetrack", "botsort"])
def short_config(request, tmp_path):
    p = yaml.safe_load((ROOT / f"configs/{request.param}.yaml").read_text())
    p["tracker"]["track_buffer"] = 2
    if request.param == "botsort": p["tracker"]["gmc_method"] = "none"  # Controlled geometry probe.
    path = tmp_path / f"{request.param}.yaml"; path.write_text(yaml.safe_dump(p))
    return path


def identity(tracks, cls=4):
    return next((t.track_id for t in tracks if t.class_id == cls), None)


def test_retention_recovers_gap_longer_than_stock(short_config):
    p = policy(retention={"base_frames": 2, "max_frames": 6}, reassociation={"enabled": False})
    oar, stock = RecoveryTracker(short_config, p), UltralyticsTracker(short_config)
    for _ in range(3):
        old_oar = identity(oar.update(detection(), FRAME))
        old_stock = identity(stock.update(detection(), FRAME))
    for _ in range(4):
        assert identity(oar.update(detection(a=False), FRAME)) is None
        assert identity(stock.update(detection(a=False), FRAME)) is None
    recovered = oar.update(detection(), FRAME)
    stock.update(detection(), FRAME)
    fresh = stock.update(detection(), FRAME)
    assert identity(recovered) == old_oar
    assert identity(fresh) != old_stock
    assert oar.diagnostics()["counts"]["retained_beyond_base"] > 0
    assert all(t.observed for t in recovered)


def test_no_overlap_does_not_extend_retention(short_config):
    oar = RecoveryTracker(short_config, policy(retention={"base_frames": 2, "max_frames": 6}))
    for _ in range(3): old = identity(oar.update(detection(b=140), FRAME))
    for _ in range(4): oar.update(detection(a=False, b=140), FRAME)
    oar.update(detection(b=140), FRAME)
    assert identity(oar.update(detection(b=140), FRAME)) != old
    assert not oar.diagnostics()["counts"].get("retained_beyond_base")


def test_extended_track_expires_before_reappearing_match(short_config):
    oar = RecoveryTracker(short_config, policy(retention={"base_frames": 2, "max_frames": 4}))
    for _ in range(3): old = identity(oar.update(detection(), FRAME))
    for _ in range(4): oar.update(detection(a=False), FRAME)
    # At gap5 a matching high detection must NOT revive the retained ID.
    assert identity(oar.update(detection(), FRAME)) is None
    assert identity(oar.update(detection(), FRAME)) != old
    assert len(oar.backend.history) <= len(oar.backend.tracked_stracks) + len(oar.backend.lost_stracks)


def test_time_cap_applies_when_timestamps_known(short_config):
    oar = RecoveryTracker(short_config, policy(retention={"base_frames": 2, "max_frames": 20, "max_gap_ms": 100}))
    for timestamp in [0, 20, 40]: old = identity(oar.update(detection(), FRAME, timestamp))
    for timestamp in [60, 100, 200]: oar.update(detection(a=False), FRAME, timestamp)
    assert old in oar.states()["removed"]
    assert oar.diagnostics()["milliseconds_cap_enforced"] is True


def test_unmatched_reassociation_preserves_old_id(monkeypatch):
    oar = RecoveryTracker(ROOT / "configs/bytetrack.yaml", policy(retention={"adaptive": False}))
    stock = UltralyticsTracker(ROOT / "configs/bytetrack.yaml")
    calls = []
    original = oar.backend.multi_predict
    monkeypatch.setattr(oar.backend, "multi_predict", lambda pool: (calls.append(1), original(pool))[1])
    for _ in range(3):
        old = identity(oar.update(detection(), FRAME))
        stock.update(detection(), FRAME)
    for _ in range(3):
        oar.update(detection(a=False), FRAME)
        stock.update(detection(a=False), FRAME)
    result = oar.update(detection(x=53), FRAME)
    stock_result = stock.update(detection(x=53), FRAME)
    assert identity(stock_result) is None  # IoU*score stock matching rejected it.
    assert identity(result) == old
    assert oar.diagnostics()["counts"].get("reassociated") == 1
    assert len(calls) == 7  # Exactly one stock Kalman multi_predict per frame.


def test_appearance_and_inconsistent_buffer_are_rejected(short_config):
    with pytest.raises(ConfigurationError, match="base_frames"):
        RecoveryTracker(short_config, policy())
    with pytest.raises(ConfigurationError, match="ReID"):
        RecoveryTracker(ROOT / "configs/bytetrack.yaml", policy(reassociation={"use_appearance": True}))


def test_ambiguity_and_one_to_one_assignment():
    assert conservative_assignment(np.array([[.2, .25], [.24, .2]]), .15) == []
    assert conservative_assignment(np.array([[.1, .8], [.8, .1]]), .15) == [(0, 0), (1, 1)]
    assert conservative_assignment(np.full((2, 3), np.inf), .15) == []


def test_exit_requires_outward_direction():
    box = np.array([1, 40, 21, 60])
    assert suspected_exit(box, np.array([-1, 0]), FRAME.shape, .03, True)
    assert not suspected_exit(box, np.array([1, 0]), FRAME.shape, .03, True)
    assert overlap_ioa(np.array([0, 0, 10, 10]), np.array([[5, 0, 15, 10]])) == .5


def test_disabled_components_match_stock_and_reset(short_config):
    oar = RecoveryTracker(short_config, policy(retention={"base_frames": 2, "adaptive": False},
                                             reassociation={"enabled": False}))
    stock = UltralyticsTracker(short_config)
    for detections in [detection(), detection(x=41), detection(a=False), detection(x=42), detection()]:
        left, right = oar.update(detections, FRAME), stock.update(detections, FRAME)
        assert left == right
    oar.reset()
    assert not oar.backend.history and not oar.diagnostics()["counts"]
    assert identity(oar.update(detection(), FRAME)) == 1


def test_recovery_rejects_wrong_class_and_large_uncertainty():
    oar = RecoveryTracker(ROOT / "configs/bytetrack.yaml", policy(retention={"adaptive": False}))
    for _ in range(3): old = identity(oar.update(detection(), FRAME))
    for _ in range(3): oar.update(detection(a=False), FRAME)
    lost = next(t for t in oar.backend.lost_stracks if t.track_id == old)
    assert oar.backend._eligible(lost)
    covariance = lost.covariance.copy()
    lost.covariance[:2, :2] = np.eye(2) * 1e8
    assert not oar.backend._eligible(lost)
    lost.covariance = covariance
    wrong = oar.update(detection(x=53, cls=5), FRAME)
    assert all(t.track_id != old for t in wrong)
    assert not oar.diagnostics()["counts"].get("reassociated")
