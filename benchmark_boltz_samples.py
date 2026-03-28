#!/usr/bin/env python
"""
Benchmark: 3 independent Boltz predictions vs a single call with --diffusion_samples 3.

Measures wall-clock time and compares outputs to confirm both approaches
produce valid, diverse samples.
"""

import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
import yaml
import json
import argparse


TEST_SEQ_SHORT = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH"
TEST_SEQ_NANOBODY = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)


def run_boltz(sequence: str, out_dir: Path, diffusion_samples: int = 1, seed: int = None) -> dict:
    """Run Boltz prediction and return timing + sample data."""

    input_yaml = out_dir / "input.yaml"
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}
        ],
    }
    with open(input_yaml, "w") as f:
        yaml.dump(data, f)

    cmd = [
        "python", "-c",
        "import sys, torch; "
        "_orig_load = torch.load; "
        "torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False}); "
        "from boltz.main import cli; "
        "sys.argv = ['boltz'] + sys.argv[1:]; "
        "cli()",
        "predict",
        str(input_yaml),
        "--out_dir", str(out_dir / "output"),
        "--write_full_pae",
        "--no_kernels",
        "--diffusion_samples", str(diffusion_samples),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}): {result.stderr[-500:]}")
        return {"elapsed": elapsed, "samples": [], "ok": False}

    output_dir = out_dir / "output"
    samples = []
    for model_idx in range(diffusion_samples):
        sample = {}
        plddt_files = list(output_dir.rglob(f"plddt_*_model_{model_idx}.npz"))
        pae_files = list(output_dir.rglob(f"pae_*_model_{model_idx}.npz"))
        conf_files = list(output_dir.rglob(f"confidence_*_model_{model_idx}.json"))

        if plddt_files:
            sample["plddt"] = np.load(str(plddt_files[0]))["plddt"]
        if pae_files:
            sample["pae"] = np.load(str(pae_files[0]))["pae"]
        if conf_files:
            with open(conf_files[0]) as f:
                sample["confidence"] = json.load(f)
        if sample:
            samples.append(sample)

    return {"elapsed": elapsed, "samples": samples, "ok": True}


def plddt_summary(plddt: np.ndarray) -> str:
    return f"mean={plddt.mean():.4f}, std={plddt.std():.4f}"


def benchmark_independent(sequence: str, n: int = 3, seed: int = 42) -> dict:
    """Run N independent Boltz calls (diffusion_samples=1 each)."""
    print(f"\n{'=' * 70}")
    print(f"APPROACH A: {n} independent Boltz calls (diffusion_samples=1)")
    print(f"{'=' * 70}")

    all_samples = []
    call_times = []

    t_total_start = time.perf_counter()
    for i in range(n):
        print(f"\n  Call {i + 1}/{n} (seed={seed + i}):")
        with tempfile.TemporaryDirectory(prefix=f"boltz_indep_{i}_") as tmpdir:
            res = run_boltz(sequence, Path(tmpdir), diffusion_samples=1, seed=seed + i)
            call_times.append(res["elapsed"])
            print(f"    Time: {res['elapsed']:.2f}s")
            if res["ok"] and res["samples"]:
                s = res["samples"][0]
                all_samples.append(s)
                if "plddt" in s:
                    print(f"    pLDDT: {plddt_summary(s['plddt'])}")
    t_total = time.perf_counter() - t_total_start

    print(f"\n  Per-call times: {[f'{t:.2f}s' for t in call_times]}")
    print(f"  Total wall-clock: {t_total:.2f}s")
    print(f"  Samples recovered: {len(all_samples)}/{n}")

    return {
        "total_time": t_total,
        "call_times": call_times,
        "samples": all_samples,
    }


def benchmark_batched(sequence: str, n: int = 3, seed: int = 42) -> dict:
    """Run a single Boltz call with diffusion_samples=N."""
    print(f"\n{'=' * 70}")
    print(f"APPROACH B: Single Boltz call (diffusion_samples={n})")
    print(f"{'=' * 70}")

    t_total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="boltz_batched_") as tmpdir:
        print(f"\n  Single call (seed={seed}, diffusion_samples={n}):")
        res = run_boltz(sequence, Path(tmpdir), diffusion_samples=n, seed=seed)
        print(f"    Time: {res['elapsed']:.2f}s")
        for i, s in enumerate(res["samples"]):
            if "plddt" in s:
                print(f"    Sample {i}: pLDDT {plddt_summary(s['plddt'])}")
    t_total = time.perf_counter() - t_total_start

    print(f"\n  Total wall-clock: {t_total:.2f}s")
    print(f"  Samples recovered: {len(res['samples'])}/{n}")

    return {
        "total_time": t_total,
        "call_times": [res["elapsed"]],
        "samples": res["samples"],
    }


def pairwise_plddt_diversity(samples: list[dict]) -> list[float]:
    """Compute mean absolute pLDDT differences between all pairs."""
    diffs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            p_i = samples[i].get("plddt")
            p_j = samples[j].get("plddt")
            if p_i is not None and p_j is not None:
                diffs.append(float(np.abs(p_i - p_j).mean()))
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Benchmark Boltz independent vs batched diffusion samples")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to generate (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--long", action="store_true", help="Use a longer nanobody sequence (~120 aa) instead of the short 35-aa test sequence")
    args = parser.parse_args()

    sequence = TEST_SEQ_NANOBODY if args.long else TEST_SEQ_SHORT
    print(f"Sequence length: {len(sequence)} aa")
    print(f"Number of samples: {args.n}")

    res_indep = benchmark_independent(sequence, n=args.n, seed=args.seed)
    res_batch = benchmark_batched(sequence, n=args.n, seed=args.seed)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sequence length: {len(sequence)} aa, samples: {args.n}")
    print()

    t_indep = res_indep["total_time"]
    t_batch = res_batch["total_time"]
    print(f"  Approach A ({args.n}x independent):    {t_indep:8.2f}s")
    print(f"  Approach B (diffusion_samples={args.n}): {t_batch:8.2f}s")
    if t_batch > 0:
        speedup = t_indep / t_batch
        print(f"  Speedup (A / B):                 {speedup:8.2f}x")
    print()

    div_indep = pairwise_plddt_diversity(res_indep["samples"])
    div_batch = pairwise_plddt_diversity(res_batch["samples"])
    if div_indep:
        print(f"  Diversity (mean |pLDDT diff|) independent: {np.mean(div_indep):.6f}")
    if div_batch:
        print(f"  Diversity (mean |pLDDT diff|) batched:     {np.mean(div_batch):.6f}")

    if div_indep and div_batch:
        all_identical_indep = all(d < 0.0001 for d in div_indep)
        all_identical_batch = all(d < 0.0001 for d in div_batch)
        print()
        if all_identical_indep:
            print("  WARNING: Independent samples are all identical!")
        else:
            print(f"  Independent samples are diverse (max pLDDT diff = {max(div_indep):.6f})")
        if all_identical_batch:
            print("  WARNING: Batched samples are all identical!")
        else:
            print(f"  Batched samples are diverse (max pLDDT diff = {max(div_batch):.6f})")


if __name__ == "__main__":
    main()
