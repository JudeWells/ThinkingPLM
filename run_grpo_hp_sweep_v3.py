#!/usr/bin/env python3
"""
GRPO hyperparameter sweep v3: focused grid with likelihood tracking.

Changes from v2:
  - Removed temperature BO (use adaptive heuristic instead)
  - Track top-10 best & worst sequences with their prompts
  - Every 5 cycles, evaluate model likelihood of best/worst sequences
  - Log avg likelihoods to wandb to verify GRPO shifts distribution

Grid: lr ∈ {1e-05, 2e-05, 4e-05} × beta ∈ {0.0, 0.1}  →  6 runs
Each run gets its own GPU via CUDA_VISIBLE_DEVICES.

Usage:
    python run_grpo_hp_sweep_v3.py
    # Or launch individual runs:
    python run_grpo_hp_sweep_v3.py --run v3_lr2e-05_noKL --gpu 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ── Sweep grid ──────────────────────────────────────────────────────────
LR_VALUES = [1e-5, 2e-5, 4e-5]

BETA_CONFIGS = [
    {"grpo_beta": 0.0,  "grpo_use_reference_model": False},  # no KL
    {"grpo_beta": 0.1,  "grpo_use_reference_model": True},   # moderate KL
]

MAX_CYCLES = 200
BASE_CONFIG = "configs/pipelines/pipeline_grpo_lis_hairpin.yaml"
AVAILABLE_GPUS = [1, 2, 4, 5, 6, 7]  # skip 0 and 3

# Shared overrides for all GRPO runs
SHARED_OVERRIDES = {
    "profam_temperature": 0.8,
    "profam_num_samples": 12,
    "grpo_replay_cycles": 7,
    "grpo_clip_ratio": 0.2,
    "rl_every_n_cycles": 1,
    "rl_steps_per_cycle": 1,
    "elitism": True,
    "accept_only_improvement": True,
    "wandb_enabled": True,
    "wandb_project": "profam-bagel-pipeline",
    # Likelihood tracking
    "likelihood_eval_every": 5,
    "likelihood_track_n": 10,
}

OUTPUT_BASE = "outputs/grpo_hp_sweep_v3"


def build_all_configs() -> list[tuple[str, Path]]:
    """Generate all run configs and return (name, config_path) pairs."""
    configs = []
    for lr in LR_VALUES:
        for beta_cfg in BETA_CONFIGS:
            beta = beta_cfg["grpo_beta"]
            use_ref = beta_cfg["grpo_use_reference_model"]
            beta_label = f"beta{beta}" if use_ref else "noKL"
            name = f"v3_lr{lr:.0e}_{beta_label}"

            tags = ["sweep_v3", "grpo", f"lr={lr}", f"beta={beta}",
                    "withKL" if use_ref else "noKL"]

            with open(BASE_CONFIG) as f:
                cfg = yaml.safe_load(f)

            cfg.update(SHARED_OVERRIDES)
            cfg.update({
                "grpo_enabled": True,
                "grpo_lr": lr,
                "grpo_beta": beta,
                "grpo_use_reference_model": use_ref,
                "wandb_tags": tags,
            })
            cfg["output_dir"] = f"{OUTPUT_BASE}/{name}"
            cfg["max_cycles"] = MAX_CYCLES
            cfg["wandb_run_name"] = name

            out_path = Path(f"{OUTPUT_BASE}/_configs/{name}.yaml")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            configs.append((name, out_path))
    return configs


def launch_run(name: str, config_path: Path, gpu: int) -> subprocess.Popen:
    """Launch a single run on a specific GPU, return the Popen object."""
    log_path = Path(f"{OUTPUT_BASE}/log_{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    proc = subprocess.Popen(
        [sys.executable, "run_profam_bagel_pipeline.py", "--config", str(config_path)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"  Launched {name} on GPU {gpu} (PID {proc.pid}, log: {log_path})")
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Run a single config by name (e.g. v3_lr2e-05_noKL)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU to use for --run")
    args = parser.parse_args()

    configs = build_all_configs()
    print(f"Generated {len(configs)} configs in {OUTPUT_BASE}/_configs/")

    if args.run:
        # Single run mode
        match = [(n, p) for n, p in configs if n == args.run]
        if not match:
            print(f"Unknown run: {args.run}. Available: {[n for n, _ in configs]}")
            sys.exit(1)
        name, path = match[0]
        gpu = args.gpu or AVAILABLE_GPUS[0]
        proc = launch_run(name, path, gpu)
        proc.wait()
        print(f"  {name} finished with exit code {proc.returncode}")
        return

    # Parallel launch mode: one run per GPU
    if len(configs) > len(AVAILABLE_GPUS):
        print(f"WARNING: {len(configs)} runs but only {len(AVAILABLE_GPUS)} GPUs. "
              f"Launching first {len(AVAILABLE_GPUS)}, remaining will wait.")

    processes: list[tuple[str, subprocess.Popen, int]] = []

    print(f"\nLaunching {min(len(configs), len(AVAILABLE_GPUS))} runs in parallel:")
    for i, (name, config_path) in enumerate(configs[:len(AVAILABLE_GPUS)]):
        gpu = AVAILABLE_GPUS[i]
        proc = launch_run(name, config_path, gpu)
        processes.append((name, proc, gpu))

    remaining = list(configs[len(AVAILABLE_GPUS):])

    # Monitor and launch remaining when GPUs free up
    results = []
    while processes or remaining:
        time.sleep(30)

        # Check for completed processes
        still_running = []
        for name, proc, gpu in processes:
            ret = proc.poll()
            if ret is not None:
                print(f"\n  FINISHED: {name} (GPU {gpu}, exit code {ret})")
                # Read best energy from cycle_stats
                stats_path = Path(f"{OUTPUT_BASE}/{name}/cycle_stats.json")
                best_energy = float("inf")
                if stats_path.exists():
                    with open(stats_path) as f:
                        stats = json.load(f)
                    energies = [stats[k].get("all_min_energy", float("inf"))
                                for k in sorted(stats.keys(), key=int)]
                    if energies:
                        best_energy = min(energies)
                results.append({"name": name, "best_energy": best_energy,
                                "return_code": ret, "gpu": gpu})

                # Launch a remaining run on the freed GPU
                if remaining:
                    next_name, next_config = remaining.pop(0)
                    next_proc = launch_run(next_name, next_config, gpu)
                    still_running.append((next_name, next_proc, gpu))
            else:
                still_running.append((name, proc, gpu))
        processes = still_running

    # Final summary
    print("\n" + "=" * 70)
    print("  SWEEP V3 COMPLETE")
    print("=" * 70)
    print(f"\n{'Run':<30} {'Best Energy':>12} {'Exit':>5}")
    print("─" * 50)
    for r in sorted(results, key=lambda x: x["best_energy"]):
        print(f"{r['name']:<30} {r['best_energy']:>12.4f} {r['return_code']:>5}")

    results_path = Path(f"{OUTPUT_BASE}/sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
