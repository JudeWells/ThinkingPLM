#!/usr/bin/env python
"""
Quick test to check if Boltz is deterministic without explicit seeding.

Also tests the --diffusion_samples option for efficient ensemble generation.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
import yaml
import json


def run_boltz_prediction(sequence: str, out_dir: Path, diffusion_samples: int = 1, seed: int = None) -> dict:
    """Run Boltz and return pLDDT/PAE for comparison."""

    input_yaml = out_dir / "input.yaml"
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}
        ]
    }
    with open(input_yaml, "w") as f:
        yaml.dump(data, f)

    # Build command with torch.load patch for PyTorch 2.6+ compat
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

    print(f"    Running Boltz (diffusion_samples={diffusion_samples}, seed={seed})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-500:]}")
        return None

    # Parse results - when diffusion_samples > 1, there will be multiple model files
    output_dir = out_dir / "output"
    results = {"samples": []}

    # Find all model files
    for model_idx in range(diffusion_samples):
        sample = {}

        # Try to find files for this model index
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
            results["samples"].append(sample)

    return results


def compare_plddts(plddt1, plddt2):
    """Compare two pLDDT arrays."""
    diff = np.abs(plddt1 - plddt2)
    return {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "identical": float(diff.max()) < 0.0001,
    }


def main():
    # Short test sequence
    test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH"  # Very short for speed

    print("=" * 70)
    print("TEST 1: Two separate Boltz calls WITHOUT explicit seed")
    print("        (Testing if Boltz is internally deterministic)")
    print("=" * 70)

    results_no_seed = []
    for i in range(2):
        print(f"\n  Call {i+1}/2:")
        with tempfile.TemporaryDirectory(prefix=f"boltz_noseed_{i}_") as tmpdir:
            result = run_boltz_prediction(test_seq, Path(tmpdir), diffusion_samples=1, seed=None)
            if result and result["samples"]:
                results_no_seed.append(result["samples"][0])
                plddt = result["samples"][0].get("plddt", [])
                if len(plddt) > 0:
                    print(f"    pLDDT: mean={plddt.mean():.4f}, std={plddt.std():.4f}")

    if len(results_no_seed) == 2 and "plddt" in results_no_seed[0] and "plddt" in results_no_seed[1]:
        comp = compare_plddts(results_no_seed[0]["plddt"], results_no_seed[1]["plddt"])
        print(f"\n  Comparison of two calls without seed:")
        print(f"    pLDDT identical: {comp['identical']}")
        print(f"    pLDDT mean diff: {comp['mean_diff']:.6f}")
        print(f"    pLDDT max diff: {comp['max_diff']:.6f}")

        if comp['identical']:
            print("\n  ⚠️  WARNING: Boltz produces IDENTICAL results without seeding!")
            print("     The ensemble is NOT providing structural diversity.")
        else:
            print("\n  ✓ Boltz produces DIFFERENT results without seeding.")
            print("    Internal randomness provides some diversity.")

    print("\n" + "=" * 70)
    print("TEST 2: Single Boltz call with --diffusion_samples 3")
    print("        (Testing efficient multi-sample generation)")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="boltz_multisample_") as tmpdir:
        result = run_boltz_prediction(test_seq, Path(tmpdir), diffusion_samples=3, seed=42)
        if result:
            print(f"\n  Generated {len(result['samples'])} samples in ONE call")
            for i, sample in enumerate(result["samples"]):
                if "plddt" in sample:
                    plddt = sample["plddt"]
                    print(f"    Sample {i}: pLDDT mean={plddt.mean():.4f}, std={plddt.std():.4f}")

            if len(result["samples"]) >= 2:
                # Compare samples
                p0 = result["samples"][0].get("plddt")
                p1 = result["samples"][1].get("plddt")
                if p0 is not None and p1 is not None:
                    comp = compare_plddts(p0, p1)
                    print(f"\n  Comparison of samples 0 vs 1:")
                    print(f"    pLDDT identical: {comp['identical']}")
                    print(f"    pLDDT mean diff: {comp['mean_diff']:.6f}")

    print("\n" + "=" * 70)
    print("TEST 3: Two calls with SAME explicit seed")
    print("        (Testing reproducibility)")
    print("=" * 70)

    results_same_seed = []
    for i in range(2):
        print(f"\n  Call {i+1}/2 with seed=42:")
        with tempfile.TemporaryDirectory(prefix=f"boltz_seed42_{i}_") as tmpdir:
            result = run_boltz_prediction(test_seq, Path(tmpdir), diffusion_samples=1, seed=42)
            if result and result["samples"]:
                results_same_seed.append(result["samples"][0])
                plddt = result["samples"][0].get("plddt", [])
                if len(plddt) > 0:
                    print(f"    pLDDT: mean={plddt.mean():.4f}")

    if len(results_same_seed) == 2 and "plddt" in results_same_seed[0] and "plddt" in results_same_seed[1]:
        comp = compare_plddts(results_same_seed[0]["plddt"], results_same_seed[1]["plddt"])
        print(f"\n  Comparison with same seed:")
        print(f"    pLDDT identical: {comp['identical']}")
        if comp['identical']:
            print("    ✓ Same seed produces identical results (reproducible)")
        else:
            print(f"    ⚠️ Same seed produces different results (diff={comp['mean_diff']:.6f})")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Use --diffusion_samples N to generate N samples in ONE Boltz call
   This is more efficient than calling Boltz N times.

2. Use --seed to ensure reproducibility or control diversity:
   - Same seed = identical results
   - Different seeds = different samples

3. Use --step_scale (default ~1.6) to control diversity:
   - Lower values = more diverse samples
   - Recommended range: 1.0 - 2.0

4. In BAGEL, pass these via extra_args:
   Boltz(extra_args=["--diffusion_samples", "3", "--seed", "42"])
""")


if __name__ == "__main__":
    main()
