"""
Test whether SolMPNN perplexity differs significantly between monomer-only
and complex-context scoring on a real heterodimer (1YCR: MDM2 + p53 peptide).

Runs n=10 ensembles under both conditions with backbone_noise > 0 to get
variance, then compares distributions with a t-test.
"""

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PDB_PATH = Path("/mnt/disk2/ThinkingPLM/target_pdbs/1YCR.pdb")
SCORER = "/home/judewells/miniconda3/envs/profam_bagel/lib/python3.11/site-packages/bagel/scripts/proteinmpnn_scorer.py"
PROTEINMPNN = "/mnt/disk2/ProteinMPNN"
ENV = "proteinmpnn"

N_REPEATS = 10
BACKBONE_NOISE = 0.1
ENSEMBLE_N = 10


def run_scorer(pdb_path, chains_to_score, seed):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        cmd = [
            "conda", "run", "-n", ENV, "python", SCORER,
            "--pdb", str(pdb_path),
            "--chains_to_score", chains_to_score,
            "--proteinmpnn_path", PROTEINMPNN,
            "--backbone_noise", str(BACKBONE_NOISE),
            "--ensemble_n", str(ENSEMBLE_N),
            "--decoding_order", f"fixed:{seed}",
            "--output_json", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"scorer failed: {proc.stderr[-300:]}")
        with open(out) as f:
            return json.load(f)


def split_pdb_monomer(pdb_path, keep_chain, output_path):
    """Write a new PDB containing only the specified chain."""
    from biotite.structure.io.pdb import PDBFile
    pdb = PDBFile.read(str(pdb_path))
    atoms = pdb.get_structure(model=1)
    atoms = atoms[atoms.chain_id == keep_chain]
    out = PDBFile()
    out.set_structure(atoms)
    out.write(str(output_path))


def main():
    # Score the p53 peptide (chain B) in two contexts
    target_chain_to_score = "B"  # the p53 peptide
    print("=" * 70)
    print("Test: SolMPNN perplexity of p53 peptide (chain B) in 1YCR")
    print(f"  n_repeats = {N_REPEATS}")
    print(f"  backbone_noise = {BACKBONE_NOISE}")
    print(f"  ensemble_n per repeat = {ENSEMBLE_N}")
    print("=" * 70)

    # Make monomer-only PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fh:
        mono_pdb = Path(fh.name)
    split_pdb_monomer(PDB_PATH, "B", mono_pdb)

    mono_perps = []
    complex_perps = []

    print("\nRunning monomer (chain B alone)...")
    for i in range(N_REPEATS):
        r = run_scorer(mono_pdb, "B", seed=100 + i)
        mono_perps.append(r["perplexity"])
        print(f"  [{i+1:>2d}/{N_REPEATS}]  perp = {r['perplexity']:.4f}  (std of ensemble passes: {r['std_nll']:.4f})")

    print("\nRunning complex (chain B in MDM2 context)...")
    for i in range(N_REPEATS):
        r = run_scorer(PDB_PATH, "B", seed=100 + i)
        complex_perps.append(r["perplexity"])
        print(f"  [{i+1:>2d}/{N_REPEATS}]  perp = {r['perplexity']:.4f}  (std of ensemble passes: {r['std_nll']:.4f})")

    mono_arr = np.array(mono_perps)
    complex_arr = np.array(complex_perps)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Monomer:  mean={mono_arr.mean():.4f}  std={mono_arr.std(ddof=1):.4f}  n={len(mono_arr)}")
    print(f"  Complex:  mean={complex_arr.mean():.4f}  std={complex_arr.std(ddof=1):.4f}  n={len(complex_arr)}")
    print(f"  Diff (complex - monomer):  {complex_arr.mean() - mono_arr.mean():+.4f}")
    print(f"  Relative:                  {100*(complex_arr.mean() - mono_arr.mean())/mono_arr.mean():+.2f}%")

    # Welch's t-test
    mean_diff = complex_arr.mean() - mono_arr.mean()
    pooled_se = np.sqrt(mono_arr.var(ddof=1)/len(mono_arr) + complex_arr.var(ddof=1)/len(complex_arr))
    if pooled_se > 0:
        t_stat = mean_diff / pooled_se
        print(f"  Welch t-statistic:         {t_stat:+.3f}")
    else:
        print(f"  Zero variance → results are deterministic")

    os.unlink(mono_pdb)


if __name__ == "__main__":
    main()
