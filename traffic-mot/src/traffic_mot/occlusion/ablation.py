"""Validation-only OAR ablations through the existing shared-cache runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from traffic_mot.detection.bridge import file_sha256
from traffic_mot.occlusion.recovery import OAR_VERSION, validate_policy
from traffic_mot.tracking.baselines import _save, run_baselines
from traffic_mot.utils.config import require


def experiment_definitions(config: dict, config_dir: Path) -> tuple[dict, dict, Path]:
    """Resolve E0-E3 plus both backends' retention/reassociation component runs."""
    experiments = config["experiments"]
    require(set(experiments) == {"E0", "E1", "E2", "E3"}, "OAR study requires E0/E1/E2/E3")
    policy_path = (config_dir / config["shared"]["occlusion_config"]).resolve()
    template = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    template["enabled"] = True
    validate_policy(template)
    definitions, snapshots = {}, {}
    for name, backend, enabled in [("E0", "bytetrack", False), ("E1", "botsort", False),
                                    ("E2", "bytetrack", True), ("E3", "botsort", True)]:
        exp = experiments[name]
        require(exp["recovery_enabled"] is enabled and exp["with_reid"] is False, "Ablation must isolate OAR with ReID off")
        path = (config_dir / exp["tracker_config"]).resolve()
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        require(profile["tracker"]["tracker_type"] == backend, f"{name} tracker configuration mismatch")
        snapshots[name] = profile
        definitions[name] = {"tracker_config": path, "recovery": copy.deepcopy(template) if enabled else None}
    require(snapshots["E0"] == snapshots["E2"] and snapshots["E1"] == snapshots["E3"],
            "Baseline and OAR must use identical tracker configuration")
    require(set(config["component_ablations"]) == {"retention_only", "reassociation_only", "retention_and_reassociation"},
            "Unsupported or missing component ablation")
    for backend, main in [("BT", "E2"), ("BS", "E3")]:
        for component in ("retention_only", "reassociation_only"):
            policy = copy.deepcopy(template)
            policy["retention"]["adaptive"] = component == "retention_only"
            policy["reassociation"]["enabled"] = component == "reassociation_only"
            definitions[f"{backend}_{component}"] = {"tracker_config": definitions[main]["tracker_config"], "recovery": policy}
    return definitions, template, policy_path


def paired_bootstrap(output: Path, *, samples: int, seed: int = 42) -> dict:
    """CI for mean per-sequence paired differences, not global toolkit aggregate."""
    rng = np.random.default_rng(seed)
    comparisons = {}
    for proposed, baseline in [("E2", "E0"), ("E3", "E1")]:
        left = json.loads((output / proposed / "metrics.json").read_text(encoding="utf-8"))
        right = json.loads((output / baseline / "metrics.json").read_text(encoding="utf-8"))
        names = sorted(left["per_sequence"])
        require(names == sorted(right["per_sequence"]), "Paired bootstrap sequence set mismatch")
        # Combine classes for each sequence using toolkit counts, not arbitrary macro class scores.
        from traffic_mot.evaluation.trackeval_runner import load_metrics, summarize
        metrics, _ = load_metrics()
        differences = {metric: [] for metric in ["HOTA", "IDF1", "MOTA", "IDSW"]}
        for name in names:
            summaries = []
            for result in (left, right):
                per_class = result["raw_per_sequence"][name]
                arrays = {}
                for cls, values in per_class.items():
                    arrays[cls] = {metric_name: {k: np.asarray(v) if isinstance(v, list) else v for k, v in values[metric_name].items()}
                                   for metric_name in metrics}
                combined = {key: metric.combine_classes_det_averaged({cls: value[key] for cls, value in arrays.items()})
                            for key, metric in metrics.items()}
                summaries.append(summarize(combined))
            for key in differences: differences[key].append(summaries[0][key] - summaries[1][key])
        comparison = {"sequence_count": len(names), "unit": "sequence", "samples": samples,
                      "interpretation": "mean per-sequence proposed-minus-baseline; distinct from global combined score difference",
                      "ci95": {}, "sequence_differences": differences}
        indices = rng.integers(0, len(names), size=(samples, len(names)))
        for key, values in differences.items():
            values = np.asarray(values)
            means = values[indices].mean(axis=1)
            comparison["ci95"][key] = {"mean": float(values.mean()), "low": float(np.percentile(means, 2.5)),
                                      "high": float(np.percentile(means, 97.5)), "degenerate_one_sequence": len(names) < 2}
        comparisons[f"{proposed}_minus_{baseline}"] = comparison
    return {"seed": seed, "comparisons": comparisons}


def _baseline_scope(baseline_run: Path, manifest_path: Path) -> tuple[dict, list[str], int | None, Path]:
    """Reuse an explicit completed E0/E1 artifact; infer its exact frame subset."""
    baseline = json.loads((baseline_run / "manifest.json").read_text(encoding="utf-8"))
    require(baseline["status"] == "complete" and baseline["mock"] is False and baseline["split"] == "validation"
            and baseline["recovery_enabled"] is False, "Need a completed real validation baseline run")
    require(set(baseline["summary"]) == {"E0", "E1"}, "Baseline reference must contain only stock E0/E1")
    require(baseline["signature"]["manifest_sha256"] == file_sha256(manifest_path), "Baseline manifest/data hash differs")
    frames = baseline["frame_counts"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = {s["name"]: s["frame_count"] for s in manifest["validation"]}
    require(set(frames) <= set(totals), "Baseline sequence outside validation")
    limit = None if all(frames[name] == totals[name] for name in frames) else max(frames.values())
    require(all(frames[name] == min(limit or totals[name], totals[name]) for name in frames), "Baseline range is not a supported common prefix")
    # Artifact path can move as a bundle; prefer its local cache when available.
    local = baseline_run / "detection_cache"
    cache = local if local.is_dir() else Path(baseline["detection_cache"])
    require(cache.is_dir(), "Baseline detection cache missing; retain/cache artifact or omit --baseline-run to infer a new shared cache")
    require(file_sha256(cache / "manifest.json") == baseline["detection_cache_manifest_sha256"], "Baseline cache manifest changed")
    return baseline, list(frames), limit, cache


def run_ablation(project_root: Path, config_path: Path, detector_config: Path, manifest_path: Path, output: Path, *,
                 split: str = "validation", sequence_names: list[str] | None = None, max_frames: int | None = None,
                 cache_root: Path | None = None, baseline_run: Path | None = None,
                 save_video: bool = False, visualization_fps: float | None = None) -> dict:
    """Evaluate stock/full/components on validation only and freeze only full runs.

    With --baseline-run, reuses its verified cache AND recomputes E0/E1 before
    OAR comparison. Otherwise all baselines are recomputed from one new cache.
    """
    require(split == "validation", "Ablation/tuning uses validation only; never optimize on test")
    require(not output.exists(), f"Refusing to overwrite ablation directory: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shared = config["shared"]
    if shared.get("split_manifest") is not None:
        require((config_path.parent / shared["split_manifest"]).resolve() == manifest_path.resolve(),
                "Configured split_manifest differs from --manifest")
    if shared.get("detection_cache") is not None:
        require(cache_root is None and baseline_run is None, "Configure cache once: YAML or CLI, not both")
        cache_root = (config_path.parent / shared["detection_cache"]).resolve()
    definitions, policy, policy_path = experiment_definitions(config, config_path.parent)
    eval_path = (config_path.parent / config["shared"]["evaluation_config"]).resolve()
    reference = None
    if baseline_run is not None:
        require(cache_root is None, "Choose --baseline-run or --detection-cache")
        reference, names, limit, cache = _baseline_scope(baseline_run, manifest_path)
        if sequence_names is not None: require(set(sequence_names) == set(names), "Requested sequences differ from baseline reference")
        if max_frames is not None: require(max_frames == limit, "Requested frame limit differs from baseline reference")
        sequence_names, max_frames, cache_root = names, limit, cache
        for name in ["E0", "E1"]:
            old = json.loads((baseline_run / name / "tracker_config.json").read_text(encoding="utf-8"))
            current = yaml.safe_load(Path(definitions[name]["tracker_config"]).read_text(encoding="utf-8"))
            require(old["config"] == current, "Baseline reference tracker configuration differs")
        old_eval = json.loads((baseline_run / "evaluation_profile.json").read_text(encoding="utf-8"))
        current_eval = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
        # Descriptive implementation status is not a scoring parameter.
        old_eval["dataset"].pop("adapter_status", None)
        current_eval["dataset"].pop("adapter_status", None)
        require(old_eval == current_eval, "Baseline reference evaluation profile differs")
    run = run_baselines(project_root, detector_config, manifest_path, output, split=split,
        sequence_names=sequence_names, max_frames=max_frames, cache_root=cache_root,
        evaluation_config=eval_path, save_video=save_video, visualization_fps=visualization_fps,
        experiment_profiles=definitions, include_diagnostics=True)
    try:
        run.update(study="occlusion_ablation", oar_version=OAR_VERSION, ablation_config_sha256=file_sha256(config_path),
                   occlusion_config_sha256=file_sha256(policy_path), baseline_reference=str(baseline_run) if baseline_run else None,
                   implementation_sha256={name: file_sha256(project_root / "src/traffic_mot" / name) for name in
                       ["occlusion/evidence.py", "occlusion/recovery.py", "occlusion/ablation.py",
                        "evaluation/occlusion.py", "tracking/baselines.py", "tracking/backend_bridge.py"]})
        _save(output / "manifest.json", run)
        if reference is not None:
            for name in ["E0", "E1"]:
                for metric in ["HOTA", "IDF1", "MOTA", "IDSW", "MT", "ML", "CLR_TP", "CLR_FP", "CLR_FN"]:
                    require(np.isclose(reference["summary"][name][metric], run["summary"][name][metric], atol=1e-10, rtol=0),
                            f"Recomputed stock baseline differs: {name}/{metric}; inspect source/runtime before attributing OAR improvement")
        episodes = {}
        for name in definitions:
            diagnostics = json.loads((output / name / "diagnostics.json").read_text(encoding="utf-8"))
            episodes[name] = diagnostics["episodes_gt_sha256"]
        require(len(set(episodes.values())) == 1, "GT recovery episode denominator differs across experiments")
        eval_profile = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
        bootstrap = paired_bootstrap(output, samples=eval_profile["diagnostics"]["bootstrap_samples"],
                                     seed=eval_profile["diagnostics"]["seed"])
        bootstrap["smoke_test"] = run["smoke_test"]
        _save(output / "paired_differences.json", bootstrap)
        comparison = {"status": "complete", "split": "validation", "smoke_test": run["smoke_test"],
                      "oar_version": OAR_VERSION, "experiments": run["summary"],
                      "episodes_gt_sha256": next(iter(episodes.values())), "baseline_reference": str(baseline_run) if baseline_run else None,
                      "fixed_policy_sha256": file_sha256(policy_path), "config_sha256": file_sha256(config_path),
                      "benchmark": False, "test_evaluated": False}
        _save(output / "ablation_summary.json", comparison)
        (output / "experiments_resolved.yaml").write_text(yaml.safe_dump({"source_config": config, "oar_policy": policy,
            "experiments": {name: {"tracker_config": str(d["tracker_config"]), "recovery": d["recovery"]}
                            for name, d in definitions.items()}}, sort_keys=False), encoding="utf-8")
        candidates = ["E0", "E1", "E2", "E3"]
        # Maximize global validation HOTA, then IDF1, then fewer IDSW; stable
        # ordering favors baseline in a tie. No additional threshold search.
        best = max(candidates, key=lambda name: (run["summary"][name]["HOTA"], run["summary"][name]["IDF1"],
                                                 -run["summary"][name]["IDSW"], -candidates.index(name)))
        frozen = {"schema_version": "1.0.0", "frozen": not run["smoke_test"], "selected_experiment": best,
                  "selection_split": "validation", "selection_rule": "max HOTA, then IDF1, then min IDSW; baseline wins ties",
                  "smoke_test": run["smoke_test"], "source_manifest_sha256": run["signature"]["manifest_sha256"],
                  "source_run_manifest_sha256": file_sha256(output / "manifest.json"), "oar_version": OAR_VERSION,
                  "detector_profile": json.loads((output / "detector_profile.json").read_text(encoding="utf-8")),
                  "evaluation_profile": eval_profile,
                  "tracker_profile": yaml.safe_load(Path(definitions[best]["tracker_config"]).read_text(encoding="utf-8")),
                  "recovery_policy": definitions[best]["recovery"], "with_reid": False,
                  "validation_score": run["summary"][best], "test_evaluated": False}
        target = output / ("candidate_config.yaml" if run["smoke_test"] else "selected_config.yaml")
        target.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
        _save(output / "selection.json", {"config": target.name, "sha256": file_sha256(target),
            "frozen": frozen["frozen"], "selected_experiment": best,
            "full_validation_required_for_final_test": True})
    except BaseException as exc:
        run.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
        _save(output / "manifest.json", run)
        raise
    return comparison
