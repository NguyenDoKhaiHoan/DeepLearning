"""Create a native VisDrone MOT manifest from ZIPs or extracted sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_mot.utils.kaggle import check_mot_dataset
from traffic_mot.utils.mot_data import prepare_mot_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Dataset mount containing exactly one val and one test-dev release")
    parser.add_argument("--validation-source", type=Path, help="Explicit val ZIP or extracted release directory")
    parser.add_argument("--test-source", type=Path, help="Explicit test-dev ZIP or extracted release directory")
    parser.add_argument("--output", type=Path, required=True, help="New manifest JSON; never overwrites")
    parser.add_argument("--check-split", choices=["validation", "test"], default="validation",
                        help="Parse selected GT only; test reserved for final evaluation")
    args = parser.parse_args()
    try:
        manifest = prepare_mot_manifest(args.output, root=args.root,
            validation_source=args.validation_source, test_source=args.test_source)
        report = check_mot_dataset(args.output, split=args.check_split)
        report_path = args.output.with_suffix(".preflight.json")
        if report_path.exists(): raise FileExistsError(f"Refusing to overwrite report: {report_path}")
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({"manifest": str(args.output.resolve()), "preflight_report": str(report_path.resolve()),
            "validation_sequences": len(manifest["validation"]), "test_sequences": len(manifest["test"]),
            "checked_split": args.check_split, "status": "passed", "tracking_implemented": False}, indent=2))
        return 0
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
