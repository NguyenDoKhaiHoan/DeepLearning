"""Stock backend parity, low-confidence association and independent ID state."""
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("lap", reason="Install runtime requirements for standard tracking tests")

from traffic_mot.tracking.backend_bridge import UltralyticsTracker
from traffic_mot.tracking.bytetrack import ByteTrackTracker
from traffic_mot.tracking.botsort import BoTSORTTracker
from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.types import DetectionBatch

ROOT = Path(__file__).resolve().parents[1]
FRAME = np.zeros((160, 200, 3), np.uint8)


def batch(x=10, score=.9, cls=4, detection_index=12):
    return DetectionBatch(np.array([[x, 30, x + 20, 60]], np.float32), np.array([score], np.float32),
                          np.array([cls]), np.array([detection_index]))


@pytest.fixture(params=["bytetrack", "botsort"])
def tracker(request):
    return UltralyticsTracker(ROOT / "configs" / f"{request.param}.yaml")


def test_high_low_and_empty_state(tracker):
    first = tracker.update(batch(), FRAME, 0)[0]
    assert first.track_id == 1 and first.detection_index == 12
    low = tracker.update(batch(11, .15), FRAME, 40)
    assert len(low) == 1 and low[0].track_id == first.track_id
    assert low[0].confidence == pytest.approx(.15)
    assert tracker.update(DetectionBatch.empty(), FRAME, 80) == []
    assert first.track_id in tracker.states()["lost"]
    recovered = tracker.update(batch(12), FRAME, 120)
    assert recovered[0].track_id == first.track_id
    # A fresh low-score object must not initialize a track.
    tracker.reset()
    assert tracker.update(batch(score=.15), FRAME) == []
    assert not tracker.states()["tracked"] and not tracker.states()["tentative"]


def test_no_gating_preserves_stock_class_switch(tracker):
    identity = tracker.update(batch(cls=4), FRAME)[0].track_id
    result = tracker.update(batch(cls=5), FRAME)[0]
    assert result.track_id == identity and result.class_id == 5


def test_tentative_birth_and_removal(tracker):
    tracker.update(DetectionBatch.empty(), FRAME)
    assert tracker.update(batch(), FRAME) == []  # Late birth is tentative.
    assert tracker.states()["tentative"] == [1]
    assert tracker.update(batch(11), FRAME)[0].track_id == 1
    for _ in range(35): tracker.update(DetectionBatch.empty(), FRAME)
    assert 1 in tracker.states()["removed"]
    assert not tracker.states()["lost"]
    assert tracker.update(batch(11), FRAME) == []
    assert tracker.update(batch(12), FRAME)[0].track_id != 1


def test_independent_instances_with_interleaved_births():
    a = ByteTrackTracker(ROOT / "configs/bytetrack.yaml")
    assert a.update(batch(), FRAME)[0].track_id == 1
    b = BoTSORTTracker(ROOT / "configs/botsort.yaml")
    assert b.update(batch(), FRAME)[0].track_id == 1
    both = DetectionBatch(np.array([[10, 30, 30, 60], [130, 30, 150, 60]], np.float32),
                          np.array([.9, .9], np.float32), np.array([4, 4]), np.array([5, 9]))
    a.update(both, FRAME); b.update(both, FRAME)
    assert {t.track_id for t in a.update(both, FRAME)} == {1, 2}
    assert {t.track_id for t in b.update(both, FRAME)} == {1, 2}


@pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
def test_adapter_matches_direct_backend(kind):
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.bot_sort import BOTSORT
    config = ROOT / f"configs/{kind}.yaml"
    adapter = UltralyticsTracker(config)
    cls = BYTETracker if kind == "bytetrack" else BOTSORT
    raw = cls(SimpleNamespace(**yaml.safe_load(config.read_text())["tracker"]))
    for detection in [batch(), batch(11, .15), DetectionBatch.empty(), batch(12), batch(14)]:
        packed = np.column_stack((detection.boxes, detection.scores, detection.class_ids)).astype(np.float32)
        expected = raw.update(Boxes(packed, FRAME.shape[:2]).numpy(), FRAME)
        actual = adapter.update(detection, FRAME)
        assert len(expected) == len(actual)
        if actual:
            assert np.allclose([t.bbox for t in actual], expected[:, :4])
            assert [t.track_id for t in actual] == expected[:, 4].astype(int).tolist()
            assert [t.class_id for t in actual] == expected[:, 6].astype(int).tolist()


def test_reid_is_rejected_without_extractor(tmp_path):
    config = yaml.safe_load((ROOT / "configs/botsort.yaml").read_text())
    config["tracker"]["with_reid"] = True
    path = tmp_path / "reid.yaml"; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigurationError, match="ReID"): UltralyticsTracker(path)


def test_timestamp_cannot_go_backwards(tracker):
    tracker.update(batch(), FRAME, 100)
    with pytest.raises(ConfigurationError, match="monotonic"): tracker.update(batch(), FRAME, 99)
