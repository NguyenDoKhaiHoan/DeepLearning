"""Evaluate the frozen validation winner on test and measure real core runtime."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="selected_config.yaml from a full validation ablation")
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", help="Optional held-out prefix smoke subset")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--visualization-fps", type=float, help="Explicit playback FPS only")
    parser.add_argument("--benchmark-frames", type=int, default=100, help="Real measured core frames after warmup; 0 disables")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true", help="Verify freeze/assets without reading test GT or running inference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.torch_threads < 1: raise ValueError("--torch-threads must be positive")
        from traffic_mot.evaluation.final import run_final_evaluation, verify_frozen_selection
        if args.check_only:
            _, report = verify_frozen_selection(ROOT, args.config.resolve(), args.detector_config.resolve(), args.manifest.resolve())
            print(json.dumps(report, indent=2))
            return 0
        import torch
        torch.set_num_threads(args.torch_threads)
        result = run_final_evaluation(ROOT, args.config.resolve(), args.detector_config.resolve(), args.manifest.resolve(),
            args.output_root.resolve(), split=args.split, sequence_names=args.sequence, max_frames=args.max_frames,
            save_video=args.save_video, visualization_fps=args.visualization_fps, benchmark_frames=args.benchmark_frames)
        print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()),
            "selected_experiment": result["selected_experiment"], "smoke_test": result["smoke_test"], "metrics": result["metrics"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted; check incomplete run manifest before retrying in a new output directory")
        return 130
    except Exception as exc:
        logging.error("Final evaluation failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
