#!/usr/bin/env python3
"""
TC Detection — unified detect -> env -> classify pipeline.

Works for any dataset (FNL, CMIP6, ERA5, JRA3Q, ...) via config.json.
All dataset-specific differences are in config; this runner is generic.

IMPORTANT: detect and env MUST run in separate processes.
Fortran detect_core has module-level state that contaminates env calculations
if both run in the same Python process. The full pipeline therefore runs:
  1. detect  (current process)  -> DBTRACK saved to outdir
  2. env     (subprocess)       -> DBTRACK_envi saved to outdir
  3. classify(subprocess)       -> TC_ saved to outdir

Usage:
    # Single model
    python run_detect.py --config CONFIG --model FNL
    python run_detect.py --config CONFIG --model FNL --years 2018
    python run_detect.py --config CONFIG --model MIROC6 --scenario hist

    # Batch (all active models)
    python run_detect.py --config CONFIG --all --all-scenarios

    # Stage control
    python run_detect.py --config CONFIG --model FNL --detect-only
    python run_detect.py --config CONFIG --model FNL --env-only
    python run_detect.py --config CONFIG --model FNL --classify-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.config_loader import (
    build_model_config, load_release_config, set_config_path,
    get_active_models,
)


def parse_years(s: str) -> list:
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s)]


def _resolve_outdir(args, rcfg, mcfg, model_name, scenario):
    """Determine output directory."""
    if args.outdir:
        return Path(args.outdir)
    if "outdir" in mcfg:
        exp_map = rcfg.get("experiment_map", {"hist": "historical"})
        experiment = exp_map.get(scenario, scenario)
        return Path(mcfg["outdir"].format(
            scenario=scenario, experiment=experiment,
            model=model_name, member=mcfg.get("member", "")))
    return Path(__file__).parent.parent.parent.parent / "output" / model_name / scenario


def _build_tracker_python():
    """Python command for detect/env — always the current interpreter."""
    return [sys.executable]


def _build_classifier_python(rcfg):
    """Python command for classifier — glibc wrapper if configured."""
    cls_cfg = rcfg.get("classifier", {})
    sysroot = cls_cfg.get("sysroot")
    python = cls_cfg.get("python")
    if sysroot and python:
        return [
            f"{sysroot}/lib64/ld-linux-x86-64.so.2",
            "--library-path", f"{sysroot}/lib64:/lib64:/usr/lib64",
            python,
        ]
    return [python or sys.executable]


def run_one(args, rcfg, model_name, scenario, years=None):
    """Run detection pipeline for one model+scenario. Returns 0 on success."""
    mcfg = rcfg["models"][model_name]
    yr_range = mcfg["scenarios"][scenario]
    styr, edyr = yr_range[0], yr_range[1]

    if years is None:
        years = list(range(styr, edyr + 1))

    out_of_range = [y for y in years if y < styr or y > edyr]
    if out_of_range:
        print(f"ERROR: years {out_of_range} outside range [{styr}, {edyr}]")
        return 1

    outdir = _resolve_outdir(args, rcfg, mcfg, model_name, scenario)
    outdir.mkdir(parents=True, exist_ok=True)

    yr_str = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    base_name = f"{model_name}_{scenario}_{yr_str}"

    print(f"\n{'='*60}")
    print(f"TC2: {model_name} {scenario} ({yr_str})")
    print(f"  output: {outdir}")
    print(f"{'='*60}")

    dbtrack_file = outdir / f"DBTRACK_{base_name}.txt"
    envi_file = outdir / f"DBTRACK_{base_name}_envi.txt"
    tc_file = outdir / f"TC_{base_name}.txt"

    # Use actual requested years for ModelConfig so coord reads use correct files
    run_styr, run_edyr = years[0], years[-1]

    # --- Stage 1: Detect ---
    if not (args.classify_only or args._env_subprocess):
        caltype = mcfg.get("caltype", 1)
        cfg = build_model_config(model_name, scenario, run_styr, run_edyr, caltype)
        print(f"  ncdir: {cfg.ncdir}")

        from lib.detect import TCDetector
        detector = TCDetector(cfg)
        tracks = detector.run(years=years)
        detector.save_tracks(tracks, str(dbtrack_file))
        print(f"  {len(tracks)} tracks -> {dbtrack_file}")

        if args.detect_only:
            return 0

    # --- Stage 2: Env (always subprocess to avoid Fortran state contamination) ---
    if not (args.detect_only or args.classify_only):
        if args._env_subprocess:
            # We ARE the env subprocess — run env in-process, then return
            caltype = mcfg.get("caltype", 1)
            cfg = build_model_config(model_name, scenario, run_styr, run_edyr, caltype)

            from lib.detect import load_tracks
            tracks = load_tracks(str(dbtrack_file))

            from lib.parameters import EnvironmentalCalculator
            calc = EnvironmentalCalculator(cfg)
            calc.compute_and_save(tracks, str(envi_file))
            return 0
        else:
            # Parent process — fork subprocess for env
            _run_env_subprocess(args, rcfg, model_name, scenario, yr_str)

            if args.env_only:
                return 0

    # --- Stage 3: Classify (always subprocess) ---
    if not (args.detect_only or args.env_only):
        _run_classifier(rcfg, str(envi_file), str(dbtrack_file), str(tc_file))

    return 0


def _run_env_subprocess(args, rcfg, model_name, scenario, yr_str):
    """Run env calculation as a separate subprocess to isolate Fortran state."""
    python_cmd = _build_tracker_python()
    script = str(Path(__file__).resolve())

    cmd = python_cmd + [
        script,
        "--config", args.config,
        "--model", model_name,
        "--scenario", scenario,
        "--years", yr_str,
        "--_env-subprocess",
    ]
    if args.outdir:
        cmd += ["--outdir", args.outdir]

    print(f"\nRunning env (subprocess)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        stderr = result.stderr.rstrip() if result.stderr else ""
        print(f"Env subprocess failed (rc={result.returncode}): {stderr}")
        raise RuntimeError("Env subprocess failed")


def _run_classifier(rcfg, envi_file, anal_file, output_file):
    """Run ML classifier via subprocess with glibc wrapper if configured."""
    if not Path(envi_file).exists():
        raise FileNotFoundError(f"Classifier: envi file not found: {envi_file}")
    if not Path(anal_file).exists():
        raise FileNotFoundError(f"Classifier: anal file not found: {anal_file}")

    classify_script = str(Path(__file__).parent.parent / "classifier" / "run_classify.py")
    cmd = _build_classifier_python(rcfg) + [classify_script]

    cmd += [
        "--envi-file", envi_file,
        "--anal-file", anal_file,
        "--output-file", output_file,
    ]

    print(f"\nRunning classifier...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(f"Classifier failed (rc={result.returncode}): {result.stderr.rstrip()}")
        raise RuntimeError("Classifier failed")


def main():
    parser = argparse.ArgumentParser(description="TC Detection Pipeline")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--scenario", default=None, help="Scenario")
    parser.add_argument("--years", default=None, help="Year range (e.g., 2000-2024)")
    parser.add_argument("--outdir", default=None, help="Override output directory")
    # Batch mode
    parser.add_argument("--all", action="store_true", help="Run all active models")
    parser.add_argument("--all-scenarios", action="store_true", help="Run all scenarios")
    # Stage control
    parser.add_argument("--detect-only", action="store_true",
                        help="Detection only, no env/classify")
    parser.add_argument("--env-only", action="store_true",
                        help="Detection + env, no classify")
    parser.add_argument("--classify-only", action="store_true",
                        help="Classify from existing DBTRACK/envi in outdir")
    # Internal: env subprocess (not user-facing)
    parser.add_argument("--_env-subprocess", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    set_config_path(args.config)
    rcfg = load_release_config()

    # Build model list
    if args.all:
        models = get_active_models()
        if not models:
            print("Error: no active_models in config", file=sys.stderr)
            return 1
    elif args.model:
        models = [args.model]
    else:
        print("Error: specify --model or --all", file=sys.stderr)
        return 1

    years = parse_years(args.years) if args.years else None

    n_ok = 0
    n_fail = 0
    for model in models:
        if model not in rcfg["models"]:
            print(f"WARNING: {model} not in config, skipping")
            continue

        mcfg = rcfg["models"][model]
        scenarios_available = mcfg.get("scenarios", {})

        if args.all_scenarios:
            scenarios = list(scenarios_available.keys())
        elif args.scenario:
            scenarios = [args.scenario]
        elif len(scenarios_available) == 1:
            scenarios = list(scenarios_available.keys())
        else:
            print(f"Error: --scenario required for {model}. "
                  f"Available: {list(scenarios_available.keys())}")
            n_fail += 1
            continue

        for scenario in scenarios:
            if scenario not in scenarios_available:
                print(f"WARNING: {model} has no {scenario}, skipping")
                continue
            try:
                rc = run_one(args, rcfg, model, scenario, years)
                if rc == 0:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                print(f"ERROR: {model} {scenario}: {e}")
                import traceback
                traceback.print_exc()
                n_fail += 1

    print(f"\nDone: {n_ok} ok, {n_fail} failed")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
