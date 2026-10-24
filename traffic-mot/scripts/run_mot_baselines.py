"""Run RT-DETRv2-P2 + stock ByteTrack/BoT-SORT on VisDrone MOT validation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--split", choices=["validation"], default="validation", help="Test reserved for frozen final evaluation")
    parser.add_argument("--sequence", action="append", help="Validation sequence name; repeat to select multiple")
    parser.add_argument("--max-frames", type=int, help="Per-sequence prefix limit; output is labeled smoke_test")
    parser.add_argument("--detection-cache", type=Path, help="Reuse a complete verified cache; otherwise create once for E0/E1")
    parser.add_argument("--bytetrack-config", type=Path, default=PROJECT_ROOT / "configs/bytetrack.yaml")
    parser.add_argument("--botsort-config", type=Path, default=PROJECT_ROOT / "configs/botsort.yaml")
    parser.add_argument("--evaluation-config", type=Path, default=PROJECT_ROOT / "configs/evaluation.yaml")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--visualization-fps", type=float, help="Display-only FPS, not camera FPS or processing throughput")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.torch_threads < 1: raise ValueError("--torch-threads must be positive")
        import torch
        torch.set_num_threads(args.torch_threads)
        from traffic_mot.tracking.baselines import run_baselines
        run = run_baselines(PROJECT_ROOT, args.detector_config.resolve(), args.manifest.resolve(), args.output_root.resolve(),
            split=args.split, sequence_names=args.sequence, max_frames=args.max_frames, cache_root=args.detection_cache,
            tracker_configs={"E0": args.bytetrack_config, "E1": args.botsort_config},
            evaluation_config=args.evaluation_config,
            save_video=args.save_video, visualization_fps=args.visualization_fps)
        print(json.dumps({"status": run["status"], "output": str(args.output_root.resolve()),
                          "smoke_test": run["smoke_test"], "summary": run["summary"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted; run marked incomplete when output directory was initialized")
        return 130
    except Exception as exc:
        logging.error("MOT baselines failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
