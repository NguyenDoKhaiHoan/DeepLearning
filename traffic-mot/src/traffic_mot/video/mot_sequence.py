"""Read native MOT JPG sequences in order, keeping GT out of inference."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np

from traffic_mot.utils.config import require
from traffic_mot.utils.mot_data import IMAGE_SUFFIXES, _ordered_frames


class MOTSequence:
    """Context-managed ZIP/directory images; no fabricated source timestamps/FPS."""

    def __init__(self, sequence: dict, manifest_base: Path):
        self.sequence, self.base, self.archive = sequence, manifest_base, None

    def __enter__(self):
        if self.sequence["source_kind"] == "images_zip":
            self.archive = ZipFile((self.base / self.sequence["archive"]).resolve())
            prefix = self.sequence["frames_prefix"]
            names = [n for n in self.archive.namelist() if n.startswith(prefix) and Path(n).suffix.lower() in IMAGE_SUFFIXES]
        elif self.sequence["source_kind"] == "images_dir":
            self.folder = (self.base / self.sequence["frames_dir"]).resolve()
            names = [p.name for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            raise ValueError("MOT runner requires native images_zip/images_dir manifest; prepare_mot_dataset.py creates this")
        try:
            self.frames = _ordered_frames(names)
            require(len(self.frames) == self.sequence["frame_count"], "Sequence frame count differs from manifest")
        except BaseException:
            self.__exit__()
            raise
        return self

    def read(self, index: int) -> np.ndarray:
        """Decode frame by zero-based index, leaving original image dimensions."""
        name = self.frames[index]
        raw = self.archive.read(name) if self.archive else (self.folder / name).read_bytes()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        require(frame is not None, f"Cannot decode source frame {name}")
        return frame

    def __exit__(self, *exc):
        if self.archive is not None: self.archive.close()


def read_gt(sequence: dict, base: Path) -> str:
    """Offline evaluation only; never passed into detector/tracker interfaces."""
    if sequence["source_kind"] == "images_zip":
        with ZipFile((base / sequence["archive"]).resolve()) as archive:
            return archive.read(sequence["gt_member"]).decode("utf-8-sig")
    return (base / sequence["gt"]).read_text(encoding="utf-8-sig")
