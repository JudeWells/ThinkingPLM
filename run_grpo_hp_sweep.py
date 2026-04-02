#!/usr/bin/env python3
"""
GRPO hyperparameter sweep: learning rate × KL beta.

Runs a grid of campaigns sequentially on a single GPU, each for 100 cycles.
Also runs a baseline with GRPO disabled for comparison.

Each run writes a temporary YAML config with overrides (avoids CLI arg issues)
and logs to wandb for easy comparison.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_grpo_hp_sweep.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ── Sweep grid ──────────────────────────────────────────────────────────
LR_VALUES = [1e-6, 1e-5, 1e-4]

BETA_CONFIGS = [
    {"grpo_beta": 0.0,  "grpo_use_reference_model": False},  # no KL
    {"grpo_beta": 0.05, "grpo_use_reference_model": True},   # mild KL
    {"grpo_beta": 0.2,  "grpo_use_reference_model": True},   # strong KL
]

MAX_CYCLES = 100
BASE_CONFIG = "configs/pipelines/pipeline_grpo_lis_hairpin.yaml"
INCLUDE_BASELINE = True


def write_run_config(run_name: str, overrides: dict) -> Path:
    """Write a temporary YAML config with sweep overrides applied."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg.update(overrides)
    cfg["output_dir"] = f"outputs/grpo_hp_sweep/{run_name}"
    cfg["max_cycles"] = MAX_CYCLES
    cfg["wandb_run_name"] = run_name

    out_path = Path(f"outputs/grpo_hp_sweep/_configs/{run_name}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out_path


def run_campaign(run_name: str, config_path: Path, tags: list[str]) -> dict:
    """Run a single campaign, streaming output to console."""
    print("\n" + "=" * 70)
    print(f"  STARTING: {run_name}")
    print(f"  Config: {config_path}")
    print(f"  Tags: {tags}")
    print("=" * 70 + "\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "run_profam_bagel_pipeline.py", "--config", str(config_path)],
        text=True,
        timeout=3600 * 6,
    )
    elapsed = time.time() - t0

    # Read results
    output_dir = f"outputs/grpo_hp_sweep/{run_name}"
    best_energy = float("inf")
    stats_path = Path(output_dir) / "cycle_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        energies = [
            stats[k].get("all_min_energy", float("inf"))
            for k in sorted(stats.keys(), key=int)
        ]
        if energies:
            best_energy = min(energies)

    summary = {
        "run_name": run_name,
        "best_energy": best_energy,
        "elapsed_min": elapsed / 60,
        "return_code": result.returncode,
        "tags": tags,
    }

    print(f"\n{'─' * 70}")
    print(f"  FINISHED: {run_name}")
    print(f"  Best energy: {best_energy:.4f}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Exit code: {result.returncode}")
    print(f"{'─' * 70}\n")

    return summary


def main():
    results_file = Path("outputs/grpo_hp_sweep/sweep_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        completed_names = {r["run_name"] for r in all_results}
        print(f"Resuming: {len(completed_names)} runs already completed")
    else:
        completed_names = set()

    # ── Baseline (no GRPO) ──────────────────────────────────────────────
    if INCLUDE_BASELINE:
        name = "baseline_no_grpo"
        if name not in completed_names:
            config_path = write_run_config(name, {
                "grpo_enabled": False,
                "wandb_tags": ["sweep", "baseline", "no_grpo"],
            })
            summary = run_campaign(name, config_path, ["sweep", "baseline"])
            all_results.append(summary)
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Grid: lr × beta ─────────────────────────────────────────────────
    for lr in LR_VALUES:
        for beta_cfg in BETA_CONFIGS:
            beta = beta_cfg["grpo_beta"]
            use_ref = beta_cfg["grpo_use_reference_model"]
            beta_label = f"beta{beta}" if use_ref else "noKL"
            name = f"lr{lr:.0e}_{beta_label}"

            if name in completed_names:
                print(f"Skipping {name} (already completed)")
                continue

            tags = ["sweep", "grpo", f"lr={lr}", f"beta={beta}",
                    "withKL" if use_ref else "noKL"]

            config_path = write_run_config(name, {
                "grpo_enabled": True,
                "grpo_lr": lr,
                "grpo_beta": beta,
                "grpo_use_reference_model": use_ref,
                "wandb_tags": tags,
            })

            summary = run_campaign(name, config_path, tags)
            summary["grpo_lr"] = lr
            summary["grpo_beta"] = beta
            summary["grpo_use_reference_model"] = use_ref
            all_results.append(summary)

            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    print(f"\n{'Run':<30} {'Best Energy':>12} {'Time (min)':>10}")
    print("─" * 55)
    for r in sorted(all_results, key=lambda x: x["best_energy"]):
        print(f"{r['run_name']:<30} {r['best_energy']:>12.4f} {r['elapsed_min']:>10.1f}")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
