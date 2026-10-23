"""Phase 1: headless video detection using verified RT-DETRv2-P2 or explicit mocks."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    """Parse only implemented options; a missing real model is an error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="MP4 path, webcam index, or RTSP URL")
    parser.add_argument("--detector", choices=["rtdetrv2_p2", "mock"], default="rtdetrv2_p2")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/detector.yaml")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Base for configured asset paths")
    parser.add_argument("--mock-fixture", type=Path, help="Required JSON list of frame detections with --detector mock")
    parser.add_argument("--device", help="cpu or cuda:N; defaults to detector YAML")
    parser.add_argument("--precision", choices=["fp32", "amp"], help="AMP requires CUDA")
    parser.add_argument("--confidence", type=float, help="Detector confidence cutoff in [0,1]")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / f"run_{uuid4().hex}")
    parser.add_argument("--save-video", action="store_true", help="Write annotated CFR MP4, without GUI/audio")
    parser.add_argument("--save-json", action="store_true", help="Write frame-level detections.jsonl")
    parser.add_argument("--max-frames", type=int, help="Explicit smoke-run limit, otherwise read to EOF")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--clock", choices=["media_pts", "frame_fps", "monotonic_session"])
    parser.add_argument("--fallback-fps", type=float, help="FPS if capture metadata is missing")
    parser.add_argument("--rtsp-reconnect-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--hide-confidence", action="store_true", help="Hide scores on video only")
    parser.add_argument("--display-threshold", type=float, default=0.3, help="Overlay cutoff only; JSON retains detector cutoff")
    args = parser.parse_args()
    if args.confidence is not None and not 0 <= args.confidence <= 1: parser.error("--confidence must be in [0,1]")
    if args.max_frames is not None and args.max_frames < 1: parser.error("--max-frames must be positive")
    if args.torch_threads < 1: parser.error("--torch-threads must be positive")
    if not 0 <= args.display_threshold <= 1: parser.error("--display-threshold must be in [0,1]")
    if args.detector == "mock" and args.mock_fixture is None: parser.error("--mock-fixture is required for mock detector")
    if args.detector != "mock" and args.mock_fixture is not None: parser.error("--mock-fixture requires --detector mock")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from traffic_mot.utils.seed import seed_everything
        from traffic_mot.video.pipeline import VideoPipeline
        seed_everything(args.seed, use_torch=args.detector != "mock")
        if args.detector == "mock":
            from traffic_mot.detection.mock import MockDetector
            detector = MockDetector.from_json(args.mock_fixture, args.confidence if args.confidence is not None else 0.1)
        else:
            import torch
            from traffic_mot.detection.rtdetrv2_p2 import RTDETRv2P2Detector
            torch.set_num_threads(args.torch_threads)
            detector = RTDETRv2P2Detector(args.config, project_root=args.project_root, device=args.device,
                                        confidence_threshold=args.confidence, precision=args.precision)
        manifest = VideoPipeline(detector, output_dir=args.output_dir, camera_id=args.camera_id,
            save_video=args.save_video, save_json=args.save_json, max_frames=args.max_frames,
            seed=args.seed, clock=args.clock, fallback_fps=args.fallback_fps,
            reconnect_attempts=args.rtsp_reconnect_attempts, show_confidence=not args.hide_confidence,
            display_threshold=args.display_threshold).process(args.source)
        print(json.dumps({k: manifest[k] for k in ("status", "mock", "frames_processed", "detection_count", "output_dir")}))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted; available run manifest is marked incomplete")
        return 130
    except Exception as exc:
        logging.error("Video detection failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
