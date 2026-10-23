"""RT-DETRv2-R18-P2 adapter using the verified official model and postprocessor."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
import threading
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from traffic_mot.detection.base import BaseDetector, VEHICLE_IDS
from traffic_mot.detection.bridge import load_upstream, preflight
from traffic_mot.utils.config import require
from traffic_mot.utils.types import DetectionBatch, validate_frame

_MODEL_INIT_LOCK = threading.Lock()  # Upstream registry/config has process-global state.
LOGGER = logging.getLogger(__name__)


class RTDETRv2P2Detector(BaseDetector):
    """Load an explicit verified checkpoint; never fall back to mock or new weights."""

    def __init__(self, config: str | Path, *, project_root: Path | None = None,
                 device: str | None = None, confidence_threshold: float | None = None,
                 precision: str | None = None):
        config = Path(config).resolve()
        self.project_root = (project_root or config.parent.parent).resolve()
        profile = yaml.safe_load(config.read_text(encoding="utf-8"))
        if device is not None: profile["device"] = device
        if confidence_threshold is not None: profile["confidence_threshold"] = confidence_threshold
        if precision is not None: profile["precision"] = precision
        self.profile = profile
        model_cfg, self.provenance = preflight(profile, self.project_root)
        self.device = torch.device(profile["device"])
        require(self.device.type in {"cpu", "cuda"}, "Only CPU/CUDA devices are supported")
        require(self.device.type != "cuda" or (torch.cuda.is_available() and
                (self.device.index is None or self.device.index < torch.cuda.device_count())),
                "CUDA unavailable or device index invalid; select --device cpu")
        self.amp = profile["precision"] == "amp"
        require(not self.amp or self.device.type == "cuda", "AMP requires CUDA")
        self.threshold = float(profile["confidence_threshold"])
        self.batch_size = profile["batch_size"]
        self.inference_frames = 0
        assets = profile["assets"]
        repo = (self.project_root / assets["repo"]).resolve()
        checkpoint_path = (self.project_root / assets["checkpoint"]).resolve()
        with _MODEL_INIT_LOCK:
            config_cls, compose_cls = load_upstream(repo)
            cfg = config_cls(str((self.project_root / assets["model_config"]).resolve()))
            # Isolate this model from the upstream loader's mutable default dict.
            cfg.yaml_cfg = copy.deepcopy(model_cfg)
            cfg.yaml_cfg["PResNet"]["pretrained"] = False
            transform_cfg = copy.deepcopy(model_cfg["val_dataloader"]["dataset"]["transforms"])
            transform_cfg.pop("type")
            self.transforms = compose_cls(**transform_cfg)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if profile["weights"] == "ema":
                require(isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"],
                        "Requested EMA weights are missing; no fallback is allowed")
                state = checkpoint["ema"]["module"]
            else:
                require("model" in checkpoint, "Requested model weights are missing")
                state = checkpoint["model"]
            self.checkpoint_epoch = checkpoint.get("last_epoch")
            self.model = cfg.model
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
            self.postprocessor = cfg.postprocessor.to(self.device).eval()
        self.provenance.update(checkpoint_epoch=self.checkpoint_epoch, weights=profile["weights"], strict_load=True)
        LOGGER.info("Loaded real RT-DETRv2-P2 epoch %s (%s, %s)", self.checkpoint_epoch, profile["weights"], self.device)

    def preprocess(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """BGR→RGB, training validation Compose, CHW; size tensor is [W,H]."""
        validate_frame(frame)
        h, w = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = self.transforms(image)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and
                tuple(tensor.shape) == (3, 640, 640), "Validation Compose returned invalid CHW tensor")
        return tensor.to(self.device), torch.tensor([w, h], dtype=torch.float32, device=self.device)

    def filter_output(self, output: dict, frame_shape: tuple[int, ...]) -> DetectionBatch:
        """Validate labels/confidence, clip xyxy once, retain low-score vehicles."""
        boxes = output["boxes"].detach().float().cpu().numpy().copy()
        scores = output["scores"].detach().float().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        n = len(scores)
        require(boxes.shape == (n, 4) and labels.shape == (n,), "Postprocessor output shapes mismatch")
        require(np.isfinite(labels).all() and np.isin(labels, list(range(11))).all(),
                "Unmapped output label; check checkpoint/class mapping")
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= 0) & (scores <= 1)
        if not valid.all(): LOGGER.warning("Dropping %d nonfinite/invalid detector rows", int((~valid).sum()))
        h, w = frame_shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (valid & (scores >= self.threshold) & np.isin(labels, list(VEHICLE_IDS)) &
                (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        return DetectionBatch(boxes[keep], scores[keep], labels[keep].astype(np.int64), np.flatnonzero(keep))

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: Sequence[np.ndarray]) -> list[DetectionBatch]:
        """Native batched inference, with a per-image original-size postprocess."""
        results = []
        for offset in range(0, len(frames), self.batch_size):
            chunk = frames[offset:offset + self.batch_size]
            prepared = [self.preprocess(frame) for frame in chunk]
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(torch.stack([item[0] for item in prepared]))
            # FP32 postprocessing also when the model forward used AMP.
            outputs = {key: value.float() for key, value in outputs.items() if isinstance(value, torch.Tensor)}
            with torch.inference_mode():
                processed = self.postprocessor(outputs, torch.stack([item[1] for item in prepared]))
            require(len(processed) == len(chunk), "Postprocessor batch length mismatch")
            results.extend(self.filter_output(out, frame.shape) for out, frame in zip(processed, chunk))
            self.inference_frames += len(chunk)
        return results

    def metadata(self) -> dict:
        return {**super().metadata(), **self.provenance, "inference_run": self.inference_frames > 0,
                "inference_frames": self.inference_frames,
                "device": str(self.device), "precision": self.profile["precision"],
                "confidence_threshold": self.threshold, "resolved_profile": self.profile}
