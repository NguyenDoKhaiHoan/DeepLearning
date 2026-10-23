"""Validate Phase 0 configuration and optionally existing detector assets."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from traffic_mot.utils.config import load_configs, validate_assets, validate_configs


def main() -> int:
    """Return nonzero for invalid contracts or unavailable requested assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Project containing configs/; paths resolve against this root.")
    parser.add_argument("--check-assets", action="store_true",
                        help="Read and hash local checkpoint; check labels, config and Git commit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        root = args.project_root.resolve()
        configs = load_configs(root)
        validate_configs(configs)
        result = {"phase": 0, "configurations": len(configs), "status": "passed",
                  "scope": "configuration_and_assets_only", "inference_run": False}
        if args.check_assets:
            result["assets"] = validate_assets(root, configs["detector"])
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        logging.error("Phase 0 validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
