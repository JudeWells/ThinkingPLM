#!/usr/bin/env python3
"""
GRPO hyperparameter sweep v4: low learning rate exploration, no KL.

Based on v3 findings:
  - Lower LR consistently better
  - noKL beats withKL
  - Best v3 run was lr=1e-05_noKL (-0.8103)
  - 450 cycles to see if longer runs help

Grid: lr ∈ {1e-05, 7e-06, 5e-06, 3e-06, 1e-06, 5e-07} × noKL only → 6 runs

Usage:
    python run_grpo_hp_sweep_v4.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

LR_VALUES = [1e-5, 7e-6, 5e-6, 3e-6, 1e-6, 5e-7]
MAX_CYCLES = 450
BASE_CONFIG = "configs/pipelines/pipeline_grpo_lis_hairpin.yaml"
AVAILABLE_GPUS = [0, 1, 3, 4, 5, 6]
OUTPUT_BASE = "outputs/grpo_hp_sweep_v4"

SHARED_OVERRIDES = {
    "profam_temperature": 0.8,
    "profam_num_samples": 12,
    "grpo_replay_cycles": 7,
    "grpo_clip_ratio": 0.2,
    "rl_every_n_cycles": 1,
    "rl_steps_per_cycle": 1,
    "elitism": True,
    "accept_only_improvement": True,
    "grpo_enabled": True,
    "grpo_beta": 0.0,
    "grpo_use_reference_model": False,
    "wandb_enabled": True,
    "wandb_project": "profam-bagel-pipeline",
    "likelihood_eval_every": 5,
    "likelihood_track_n": 10,
}


def build_configs() -> list[tuple[str, Path]]:
    configs = []
    for lr in LR_VALUES:
        name = f"v4_lr{lr:.0e}_noKL"
        tags = ["sweep_v4", "grpo", f"lr={lr}", "noKL"]

        with open(BASE_CONFIG) as f:
            cfg = yaml.safe_load(f)

        cfg.update(SHARED_OVERRIDES)
        cfg["grpo_lr"] = lr
        cfg["output_dir"] = f"{OUTPUT_BASE}/{name}"
        cfg["max_cycles"] = MAX_CYCLES
        cfg["wandb_run_name"] = name
        cfg["wandb_tags"] = tags

        out_path = Path(f"{OUTPUT_BASE}/_configs/{name}.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        configs.append((name, out_path))
    return configs


def main():
    configs = build_configs()
    print(f"Generated {len(configs)} configs")

    processes = []
    for i, (name, config_path) in enumerate(configs):
        gpu = AVAILABLE_GPUS[i]
        log_path = Path(f"{OUTPUT_BASE}/log_{name}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

        proc = subprocess.Popen(
            [sys.executable, "run_profam_bagel_pipeline.py", "--config", str(config_path)],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        print(f"  {name} -> GPU {gpu} (PID {proc.pid})")
        processes.append((name, proc, gpu))

    # Monitor
    results = []
    while processes:
        time.sleep(60)
        still_running = []
        for name, proc, gpu in processes:
            ret = proc.poll()
            if ret is not None:
                stats_path = Path(f"{OUTPUT_BASE}/{name}/cycle_stats.json")
                best_energy = float("inf")
                n_cycles = 0
                if stats_path.exists():
                    with open(stats_path) as f:
                        stats = json.load(f)
                    n_cycles = len(stats)
                    energies = [stats[k].get("all_min_energy", float("inf"))
                                for k in sorted(stats.keys(), key=int)]
                    if energies:
                        best_energy = min(energies)
                results.append({"name": name, "best_energy": best_energy,
                                "n_cycles": n_cycles, "return_code": ret})
                print(f"  DONE: {name} ({n_cycles} cycles, E={best_energy:.4f}, exit={ret})")
            else:
                still_running.append((name, proc, gpu))
        processes = still_running

    print("\n" + "=" * 60)
    print(f"{'Run':<25} {'Cycles':>7} {'Best Energy':>12} {'Exit':>5}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["best_energy"]):
        print(f"{r['name']:<25} {r['n_cycles']:>7} {r['best_energy']:>12.4f} {r['return_code']:>5}")

    with open(Path(f"{OUTPUT_BASE}/sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
