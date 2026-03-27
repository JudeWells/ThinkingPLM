#!/usr/bin/env python
"""
Test script to verify Boltz ensemble behavior.

Checks:
1. Are multiple Boltz predictions actually different or identical?
2. What CLI options does Boltz have for controlling randomness?
3. Is there a more efficient way to generate multiple samples?
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path


def check_boltz_cli_options():
    """Check what seed/sampling options Boltz CLI supports."""
    print("=" * 60)
    print("CHECKING BOLTZ CLI OPTIONS")
    print("=" * 60)

    result = subprocess.run(
        ["boltz", "predict", "--help"],
        capture_output=True,
        text=True,
    )

    print("\nBoltz predict --help output (relevant options):")
    for line in result.stdout.split('\n'):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['seed', 'sample', 'diffusion', 'recycl', 'model']):
            print(f"  {line}")

    return result.stdout


def run_single_boltz_prediction(sequence: str, out_dir: Path, seed: int = None) -> dict:
    """Run a single Boltz prediction and return the results."""
    import yaml

    # Create input YAML
    input_yaml = out_dir / "input.yaml"
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}
        ]
    }
    with open(input_yaml, "w") as f:
        yaml.dump(data, f)

    # Build command
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
    ]

    # Add seed if provided (check if this option exists)
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"  Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    # Parse results
    output_dir = out_dir / "output"

    # Find files
    cif_files = list(output_dir.rglob("*_model_*.cif"))
    cif_files = [f for f in cif_files if not f.name.startswith(("pae_", "plddt_"))]

    pae_files = list(output_dir.rglob("pae_*_model_*.npz"))
    plddt_files = list(output_dir.rglob("plddt_*_model_*.npz"))

    results = {}

    if cif_files:
        # Read CA coordinates
        from biotite.structure.io.pdbx import CIFFile, get_structure
        cif = CIFFile.read(str(cif_files[0]))
        structure = get_structure(cif, model=1)
        ca_atoms = structure[structure.atom_name == "CA"]
        results["ca_coords"] = ca_atoms.coord.copy()

    if pae_files:
        results["pae"] = np.load(str(pae_files[0]))["pae"]

    if plddt_files:
        results["plddt"] = np.load(str(plddt_files[0]))["plddt"]

    return results


def compare_structures(results_list: list) -> dict:
    """Compare multiple Boltz prediction results."""
    stats = {}

    # Compare CA coordinates
    if all("ca_coords" in r for r in results_list):
        coords = [r["ca_coords"] for r in results_list]

        # Compute pairwise RMSD
        rmsds = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                diff = coords[i] - coords[j]
                rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                rmsds.append(rmsd)

        stats["ca_rmsd_pairs"] = rmsds
        stats["ca_rmsd_mean"] = np.mean(rmsds) if rmsds else 0
        stats["ca_rmsd_max"] = np.max(rmsds) if rmsds else 0
        stats["structures_identical"] = all(rmsd < 0.01 for rmsd in rmsds)

    # Compare PAE matrices
    if all("pae" in r for r in results_list):
        paes = [r["pae"] for r in results_list]

        pae_diffs = []
        for i in range(len(paes)):
            for j in range(i + 1, len(paes)):
                diff = np.abs(paes[i] - paes[j]).mean()
                pae_diffs.append(diff)

        stats["pae_diff_pairs"] = pae_diffs
        stats["pae_diff_mean"] = np.mean(pae_diffs) if pae_diffs else 0
        stats["pae_identical"] = all(d < 0.001 for d in pae_diffs)

    # Compare pLDDT
    if all("plddt" in r for r in results_list):
        plddts = [r["plddt"] for r in results_list]

        plddt_diffs = []
        for i in range(len(plddts)):
            for j in range(i + 1, len(plddts)):
                diff = np.abs(plddts[i] - plddts[j]).mean()
                plddt_diffs.append(diff)

        stats["plddt_diff_pairs"] = plddt_diffs
        stats["plddt_diff_mean"] = np.mean(plddt_diffs) if plddt_diffs else 0
        stats["plddt_identical"] = all(d < 0.001 for d in plddt_diffs)

    return stats


def test_ensemble_without_seeds():
    """Test if running Boltz 3 times without seeds gives different results."""
    print("\n" + "=" * 60)
    print("TEST 1: Multiple predictions WITHOUT explicit seeds")
    print("=" * 60)

    # Short test sequence (nanobody-like)
    test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

    results = []
    for i in range(3):
        print(f"\n  Prediction {i+1}/3:")
        with tempfile.TemporaryDirectory(prefix=f"boltz_test_{i}_") as tmpdir:
            result = run_single_boltz_prediction(test_seq, Path(tmpdir))
            if result:
                results.append(result)
                if "plddt" in result:
                    print(f"    pLDDT mean: {result['plddt'].mean():.4f}")

    if len(results) >= 2:
        stats = compare_structures(results)
        print(f"\n  Results:")
        print(f"    Structures identical (RMSD < 0.01): {stats.get('structures_identical', 'N/A')}")
        print(f"    CA RMSD between predictions: {stats.get('ca_rmsd_pairs', 'N/A')}")
        print(f"    PAE identical: {stats.get('pae_identical', 'N/A')}")
        print(f"    pLDDT identical: {stats.get('plddt_identical', 'N/A')}")
        return stats
    return None


def test_ensemble_with_seeds():
    """Test if explicit seeds produce different results."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple predictions WITH different seeds")
    print("=" * 60)

    test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

    results = []
    seeds = [1, 42, 123]

    for i, seed in enumerate(seeds):
        print(f"\n  Prediction {i+1}/3 with seed={seed}:")
        with tempfile.TemporaryDirectory(prefix=f"boltz_seed_{seed}_") as tmpdir:
            try:
                result = run_single_boltz_prediction(test_seq, Path(tmpdir), seed=seed)
                if result:
                    results.append(result)
                    if "plddt" in result:
                        print(f"    pLDDT mean: {result['plddt'].mean():.4f}")
            except Exception as e:
                print(f"    Failed with seed option: {e}")
                print("    (Boltz may not support --seed flag)")
                return None

    if len(results) >= 2:
        stats = compare_structures(results)
        print(f"\n  Results:")
        print(f"    Structures identical (RMSD < 0.01): {stats.get('structures_identical', 'N/A')}")
        print(f"    CA RMSD between predictions: {stats.get('ca_rmsd_pairs', 'N/A')}")
        print(f"    PAE identical: {stats.get('pae_identical', 'N/A')}")
        print(f"    pLDDT identical: {stats.get('plddt_identical', 'N/A')}")
        return stats
    return None


def check_bagel_boltz_kwargs():
    """Check what kwargs BAGEL's Boltz oracle accepts."""
    print("\n" + "=" * 60)
    print("BAGEL BOLTZ ORACLE PARAMETERS")
    print("=" * 60)

    from bagel.oracles.folding import Boltz
    import inspect

    sig = inspect.signature(Boltz.__init__)
    print("\nBoltz.__init__ parameters:")
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        default = param.default if param.default is not inspect.Parameter.empty else "REQUIRED"
        print(f"  {name}: {default}")

    # Check if model_seeds is used anywhere
    import bagel.oracles.folding.boltz as boltz_module
    source = inspect.getsource(boltz_module)

    print("\n  'model_seeds' usage in boltz.py:")
    for i, line in enumerate(source.split('\n')):
        if 'model_seeds' in line:
            print(f"    Line {i+1}: {line.strip()}")


def main():
    print("BOLTZ ENSEMBLE BEHAVIOR TEST")
    print("=" * 60)

    # Check CLI options
    help_text = check_boltz_cli_options()

    # Check BAGEL wrapper
    check_bagel_boltz_kwargs()

    # Run actual tests
    print("\n" + "=" * 60)
    print("RUNNING PREDICTION TESTS")
    print("(This may take a few minutes per prediction)")
    print("=" * 60)

    stats_no_seed = test_ensemble_without_seeds()

    # Only try seeds if the option exists
    if "--seed" in help_text.lower():
        stats_with_seed = test_ensemble_with_seeds()
    else:
        print("\n  Boltz does not appear to support --seed flag")
        stats_with_seed = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if stats_no_seed:
        if stats_no_seed.get('structures_identical'):
            print("\n  WARNING: Multiple Boltz calls produce IDENTICAL structures!")
            print("  The ensemble is NOT providing diversity.")
            print("  Possible fixes:")
            print("    1. Use different --seed values (if supported)")
            print("    2. Vary --sampling_steps")
            print("    3. Check if Boltz has a num_samples option")
        else:
            print("\n  Multiple Boltz calls produce DIFFERENT structures.")
            print(f"  Mean CA RMSD: {stats_no_seed.get('ca_rmsd_mean', 0):.3f} Å")


if __name__ == "__main__":
    main()
