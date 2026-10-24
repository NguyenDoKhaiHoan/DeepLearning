"""Native frame/GT mapping, valid zero IDs and held-out label isolation."""
from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import pytest

from traffic_mot.utils.config import ConfigurationError
from traffic_mot.utils.kaggle import check_mot_dataset
from traffic_mot.utils.mot_data import prepare_mot_manifest, validate_gt


def write_release(path: Path, release: str, sequence: str, color: int, *, gt: str | None = None,
                  frame_ids: tuple[int, ...] = (1, 2)) -> None:
    pixels = io.BytesIO()
    Image.new("RGB", (32, 24), (color, 10, 20)).save(pixels, format="JPEG")
    with ZipFile(path, "w") as archive:
        for frame in frame_ids:
            archive.writestr(f"{release}/sequences/{sequence}/{frame:07d}.jpg", pixels.getvalue())
        annotation = "1,0,1,1,10,10,1,4,0,0\n2,0,2,1,10,10,1,4,0,1\n" if gt is None else gt
        archive.writestr(f"{release}/annotations/{sequence}.txt", annotation)


@pytest.fixture
def sources(tmp_path):
    validation, test = tmp_path / "VisDrone2019-MOT-val.zip", tmp_path / "VisDrone2019-MOT-test-dev.zip"
    write_release(validation, "VisDrone2019-MOT-val", "uav0001_00000_v", 60)
    # No test GT parsing during preparation/validation; the contents are deliberately not rows.
    write_release(test, "VisDrone2019-MOT-test-dev", "uav0002_00000_v", 160, gt="held-out labels")
    return validation, test


def test_prepare_real_format_preserves_zero_id_and_does_not_parse_test(sources, tmp_path):
    manifest_path = tmp_path / "out/mot_split_manifest.json"
    manifest = prepare_mot_manifest(manifest_path, root=tmp_path)
    assert manifest["validation"][0]["source_kind"] == "images_zip"
    assert manifest["test_labels_parsed"] is False
    report = check_mot_dataset(manifest_path)
    assert report["sequences"][0]["gt_id_zero_observations"] == 2
    assert report["sequences"][0]["gt_identities"] == 1
    assert report["sequences"][0]["frame_count"] == 2
    assert report["sequences"][0]["width"] == 32
    assert report["sequences"][0]["fps"] is None
    assert report["evaluation_performed"] is False
    with pytest.raises(ConfigurationError, match="10-column"):
        check_mot_dataset(manifest_path, split="test")


def test_extracted_image_release(sources, tmp_path):
    val, test = sources
    with ZipFile(val) as archive: archive.extractall(tmp_path / "extracted")
    with ZipFile(test) as archive: archive.extractall(tmp_path / "extracted")
    manifest_path = tmp_path / "images.json"
    prepare_mot_manifest(manifest_path, root=tmp_path / "extracted")
    report = check_mot_dataset(manifest_path)
    assert report["sequences"][0]["source_kind"] == "images_dir"
    assert report["sequences"][0]["gt_id_zero_observations"] == 2


def test_no_overwrite_and_missing_sources(sources, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("keep me")
    with pytest.raises(FileExistsError): prepare_mot_manifest(path, root=tmp_path)
    assert path.read_text() == "keep me"
    with pytest.raises(ConfigurationError, match="Supply both"):
        prepare_mot_manifest(tmp_path / "new.json")


def test_source_video_segments_cannot_cross_splits(sources, tmp_path):
    val, test = sources
    write_release(test, "VisDrone2019-MOT-test-dev", "uav0001_00200_v", 120)
    with pytest.raises(ConfigurationError, match="same source UAV"):
        prepare_mot_manifest(tmp_path / "leakage.json", validation_source=val, test_source=test)


def test_manifest_checksum_blocks_changed_archive(sources, tmp_path):
    val, test = sources
    path = tmp_path / "manifest.json"
    prepare_mot_manifest(path, validation_source=val, test_source=test)
    with ZipFile(val, "a") as archive: archive.writestr("changed.txt", "tampered")
    with pytest.raises(ConfigurationError, match="checksum mismatch"): check_mot_dataset(path)


def test_timeline_gaps_are_rejected(sources, tmp_path):
    val, test = sources
    write_release(val, "VisDrone2019-MOT-val", "uav0001_00000_v", 60, frame_ids=(1, 3))
    with pytest.raises(ConfigurationError, match="contiguous"):
        prepare_mot_manifest(tmp_path / "gaps.json", validation_source=val, test_source=test)


@pytest.mark.parametrize("text, message", [
    ("1,0,1,1,10,10,1,4\n", "not VisDrone DET"),
    ("1,-1,1,1,10,10,1,4,0,0\n", "nonnegative"),
    ("3,0,1,1,10,10,1,4,0,0\n", "exceeds"),
    ("1,0,1,1,10,10,1,4,0,0\n1,0,2,2,10,10,1,4,0,0\n", "Duplicate"),
])
def test_gt_validation_rejects_invalid_inputs(text, message):
    with pytest.raises(ConfigurationError, match=message): validate_gt(text, 2)
