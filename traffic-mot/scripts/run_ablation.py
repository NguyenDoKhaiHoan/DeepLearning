"""Run validation-only E0-E3 and component ablations with a shared detector cache."""
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, help="Completed E0/E1 run; reuse cache, scope and verify baseline metrics")
    parser.add_argument("--detection-cache", type=Path, help="Complete real detection cache with matching signature")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--visualization-fps", type=float)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.torch_threads < 1: raise ValueError("--torch-threads must be positive")
        import torch
        torch.set_num_threads(args.torch_threads)
        from traffic_mot.occlusion.ablation import run_ablation
        result = run_ablation(PROJECT_ROOT, args.config.resolve(), args.detector_config.resolve(), args.manifest.resolve(),
            args.output_root.resolve(), split=args.split, sequence_names=args.sequence, max_frames=args.max_frames,
            cache_root=args.detection_cache.resolve() if args.detection_cache else None,
            baseline_run=args.baseline_run.resolve() if args.baseline_run else None,
            save_video=args.save_video, visualization_fps=args.visualization_fps)
        print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()),
                          "smoke_test": result["smoke_test"], "results": result["experiments"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted; check incomplete run manifest before retrying with a new output directory")
        return 130
    except Exception as exc:
        logging.error("Ablation failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
