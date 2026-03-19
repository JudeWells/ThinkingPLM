#!/usr/bin/env python3
"""
Compare ipSAE and iPTM scores between Boltz and ColabFold for the
BindCraft rank2 starting sequence against 15-PGDH (2GDZ).

Boltz: 10 runs with different seeds to get score distributions.
ColabFold: all 5 models, report best and all.

Usage:
    python experiment_boltz_vs_colabfold.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINDER_SEQ = (
    "MEELSEKQKELKKKAEVVLKRTEEMRETDMVGHWREMQKKFGMPEEYVKMMEAVGEFVVETMKVYMEHEVTGKLRLEEVPELFERIVRPYMQPSMEATNEYNKKHFS"
)

TARGET_SEQ = (
    "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQKTLFIQCDVADQQQLRDTFRKVVDHFGRLDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ"
)

COLABFOLD_BIN = "/mnt/disk2/colabfold_2025_09/localcolabfold/colabfold-conda/bin/colabfold_batch"
PROFAM_PYTHON = "/home/judewells/miniconda3/envs/profam_bagel/bin/python"

OUTPUT_DIR = Path("outputs/experiment_boltz_vs_colabfold")

N_BOLTZ_SEEDS = 10

# ---------------------------------------------------------------------------
# ipSAE computation (from test_ipsae_agreement.py, BAGEL variant)
# ---------------------------------------------------------------------------

def calc_d0_array(L):
    L = np.maximum(np.asarray(L, dtype=float), 27.0)
    return np.maximum(1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8, 1.0)


def ipsae_one_direction(pae, source_indices, target_indices, pae_cutoff=10.0):
    if len(source_indices) == 0 or len(target_indices) == 0:
        return 0.0
    pae_cross = pae[np.ix_(source_indices, target_indices)]
    valid_mask = pae_cross < pae_cutoff
    n_partners = valid_mask.sum(axis=1).astype(np.float64)
    has_partners = n_partners > 0
    if not np.any(has_partners):
        return 0.0
    d0_per_res = calc_d0_array(n_partners)
    per_res_scores = np.zeros(len(source_indices), dtype=np.float64)
    for i in range(len(source_indices)):
        if not has_partners[i]:
            continue
        partner_pae = pae_cross[i, valid_mask[i]]
        ptm_scores = 1.0 / (1.0 + (partner_pae / d0_per_res[i]) ** 2)
        per_res_scores[i] = ptm_scores.mean()
    return float(np.max(per_res_scores))


def compute_ipsae(pae_matrix, n_binder, n_target, pae_cutoff=10.0):
    """Compute symmetric ipSAE between binder (first chain) and target (second chain)."""
    binder_idx = np.arange(n_binder)
    target_idx = np.arange(n_binder, n_binder + n_target)
    s_bt = ipsae_one_direction(pae_matrix, binder_idx, target_idx, pae_cutoff)
    s_tb = ipsae_one_direction(pae_matrix, target_idx, binder_idx, pae_cutoff)
    return max(s_bt, s_tb)


# ---------------------------------------------------------------------------
# Boltz prediction (single seed)
# ---------------------------------------------------------------------------

def run_boltz_single_seed(binder_seq, target_seq, output_dir, seed):
    """Run Boltz with a single seed and return metrics dict."""
    boltz_dir = output_dir / f"boltz_seed_{seed}"
    boltz_dir.mkdir(parents=True, exist_ok=True)

    script = boltz_dir / "_run_boltz.py"
    script.write_text(f'''
import sys
import json
import numpy as np
sys.path.insert(0, "/mnt/disk2/ThinkingPLM")

from bagel.oracles.folding.boltz import Boltz
from bagel.chain import Chain, Residue

# Build chains
binder_residues = [Residue(name=aa, chain_ID="A", index=i) for i, aa in enumerate("{binder_seq}")]
target_residues = [Residue(name=aa, chain_ID="B", index=i) for i, aa in enumerate("{target_seq}")]
binder_chain = Chain(residues=binder_residues)
target_chain = Chain(residues=target_residues)

# Run Boltz with specific seed
oracle = Boltz(model_seeds=[{seed}])
result = oracle.predict([binder_chain, target_chain])

# Save outputs
pae = result.pae[0]
np.save("{boltz_dir}/pae.npy", pae)

iptm = result.chain_pair_iptm
np.save("{boltz_dir}/chain_pair_iptm.npy", iptm)

ptm = float(result.ptm[0])
plddt_mean = float(result.local_plddt.mean())

output = {{
    "seed": {seed},
    "ptm": ptm,
    "plddt_mean": plddt_mean,
    "iptm_matrix": iptm.tolist(),
    "pae_shape": list(pae.shape),
}}
with open("{boltz_dir}/metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Boltz seed {seed}: pTM={{ptm:.4f}}, pLDDT={{plddt_mean:.4f}}, iPTM={{float(iptm[0,1]):.4f}}")
''')

    result = subprocess.run(
        [PROFAM_PYTHON, str(script)],
        capture_output=True,
        text=True,
        cwd="/mnt/disk2/ThinkingPLM",
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  Boltz seed {seed} STDERR (last 500 chars):\n{result.stderr[-500:]}")
        raise RuntimeError(f"Boltz seed {seed} failed")

    n_binder = len(binder_seq)
    n_target = len(target_seq)
    pae = np.load(boltz_dir / "pae.npy")
    iptm = np.load(boltz_dir / "chain_pair_iptm.npy")
    with open(boltz_dir / "metrics.json") as f:
        metrics = json.load(f)

    ipsae = compute_ipsae(pae, n_binder, n_target)
    iptm_val = float(iptm[0, 1]) if iptm.shape[0] >= 2 else 0.0

    return {
        "seed": seed,
        "ipSAE": ipsae,
        "iPTM": iptm_val,
        "pTM": metrics["ptm"],
        "mean_pLDDT": metrics["plddt_mean"],
    }


# ---------------------------------------------------------------------------
# ColabFold prediction (all 5 models)
# ---------------------------------------------------------------------------

def run_colabfold(binder_seq, target_seq, output_dir):
    """Run ColabFold with all 5 models and return per-model metrics."""
    print("\n" + "=" * 60)
    print("Running ColabFold (5 models)...")
    print("=" * 60)

    cf_dir = output_dir / "colabfold"
    cf_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = cf_dir / "input.fasta"
    fasta_path.write_text(f">binder_target\n{binder_seq}:{target_seq}\n")

    results_dir = cf_dir / "results"
    results_dir.mkdir(exist_ok=True)

    cmd = [
        COLABFOLD_BIN,
        str(fasta_path),
        str(results_dir),
        "--num-models", "5",
        "--num-seeds", "1",
        "--num-recycle", "3",
        "--rank", "iptm",
        "--overwrite-existing-results",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"ColabFold STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError("ColabFold prediction failed")

    # Parse all score files (one per model)
    score_files = sorted(results_dir.glob("*scores*.json"))
    if not score_files:
        raise FileNotFoundError(f"No score files found in {results_dir}")

    print(f"Found {len(score_files)} score files")

    n_binder = len(binder_seq)
    n_target = len(target_seq)
    model_results = []

    for sf in score_files:
        with open(sf) as f:
            scores = json.load(f)

        pae = np.array(scores["pae"])
        ipsae = compute_ipsae(pae, n_binder, n_target)
        iptm_val = scores.get("iptm", None)
        ptm_val = scores.get("ptm", None)
        plddt_mean = float(np.mean(scores.get("plddt", [0])))

        model_name = sf.stem.replace("binder_target_scores_", "")
        model_results.append({
            "model": model_name,
            "ipSAE": ipsae,
            "iPTM": iptm_val,
            "pTM": ptm_val,
            "mean_pLDDT": plddt_mean,
        })
        print(f"  {model_name}: ipSAE={ipsae:.4f}, iPTM={iptm_val}, pTM={ptm_val}, pLDDT={plddt_mean:.1f}")

    return model_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_comparison_plot(boltz_results, colabfold_results, output_dir):
    """Plot distributions of ipSAE and iPTM for Boltz seeds + ColabFold models."""
    plt.style.use("dark_background")

    metrics = ["ipSAE", "iPTM", "pTM", "mean_pLDDT"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        boltz_vals = [r[metric] for r in boltz_results if r.get(metric) is not None]
        cf_vals = [r[metric] for r in colabfold_results if r.get(metric) is not None]

        # Normalize pLDDT to same scale (ColabFold uses 0-100, Boltz uses 0-1)
        if metric == "mean_pLDDT" and cf_vals and max(cf_vals) > 1:
            cf_vals = [v / 100.0 for v in cf_vals]
            label_suffix = " (scaled to 0-1)"
        else:
            label_suffix = ""

        # Strip plot with jitter
        jitter_b = np.random.default_rng(42).normal(0, 0.04, len(boltz_vals))
        jitter_c = np.random.default_rng(43).normal(0, 0.04, len(cf_vals))

        ax.scatter(
            np.zeros(len(boltz_vals)) + jitter_b,
            boltz_vals,
            color="#00bfff", s=80, alpha=0.8, edgecolors="white", linewidths=0.5,
            label=f"Boltz (n={len(boltz_vals)})", zorder=3,
        )
        ax.scatter(
            np.ones(len(cf_vals)) + jitter_c,
            cf_vals,
            color="#ff6b6b", s=80, alpha=0.8, edgecolors="white", linewidths=0.5,
            label=f"ColabFold (n={len(cf_vals)})", zorder=3,
        )

        # Mean lines
        if boltz_vals:
            ax.hlines(np.mean(boltz_vals), -0.25, 0.25, colors="#00bfff", linewidths=2, zorder=4)
        if cf_vals:
            ax.hlines(np.mean(cf_vals), 0.75, 1.25, colors="#ff6b6b", linewidths=2, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Boltz", "ColabFold"])
        ax.set_ylabel(metric + label_suffix)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Boltz vs ColabFold: BindCraft rank2 binder + 15-PGDH (2GDZ)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plot_path = output_dir / "boltz_vs_colabfold_comparison.png"
    fig.savefig(plot_path, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_binder = len(BINDER_SEQ)
    n_target = len(TARGET_SEQ)
    print(f"Binder length: {n_binder}")
    print(f"Target length: {n_target}")
    print(f"Total complex: {n_binder + n_target}")

    # --- Boltz: 10 seeds ---
    print("\n" + "=" * 60)
    print(f"Running Boltz with {N_BOLTZ_SEEDS} seeds...")
    print("=" * 60)

    boltz_results = []
    for seed in range(1, N_BOLTZ_SEEDS + 1):
        try:
            res = run_boltz_single_seed(BINDER_SEQ, TARGET_SEQ, OUTPUT_DIR, seed)
            boltz_results.append(res)
        except Exception as e:
            print(f"  Boltz seed {seed} failed: {e}")

    # --- ColabFold: 5 models ---
    try:
        colabfold_results = run_colabfold(BINDER_SEQ, TARGET_SEQ, OUTPUT_DIR)
    except Exception as e:
        print(f"ColabFold failed: {e}")
        colabfold_results = []

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("BOLTZ RESULTS (per seed)")
    print("=" * 60)
    print(f"{'Seed':>6} {'ipSAE':>8} {'iPTM':>8} {'pTM':>8} {'pLDDT':>8}")
    print("-" * 42)
    for r in boltz_results:
        print(f"{r['seed']:>6} {r['ipSAE']:>8.4f} {r['iPTM']:>8.4f} {r['pTM']:>8.4f} {r['mean_pLDDT']:>8.4f}")
    if boltz_results:
        print("-" * 42)
        for metric in ["ipSAE", "iPTM", "pTM", "mean_pLDDT"]:
            vals = [r[metric] for r in boltz_results]
            print(f"  {metric:<12} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    print("\n" + "=" * 60)
    print("COLABFOLD RESULTS (per model)")
    print("=" * 60)
    for r in colabfold_results:
        iptm_str = f"{r['iPTM']:.4f}" if r['iPTM'] is not None else "N/A"
        ptm_str = f"{r['pTM']:.4f}" if r['pTM'] is not None else "N/A"
        print(f"  {r['model']:<50} ipSAE={r['ipSAE']:.4f}  iPTM={iptm_str}  pTM={ptm_str}  pLDDT={r['mean_pLDDT']:.1f}")

    if colabfold_results:
        best_cf = max(colabfold_results, key=lambda r: r.get("iPTM") or 0)
        print(f"\n  Best ColabFold (by iPTM): {best_cf['model']}")
        print(f"    ipSAE={best_cf['ipSAE']:.4f}  iPTM={best_cf['iPTM']}  pTM={best_cf['pTM']}")

    # --- Plot ---
    if boltz_results or colabfold_results:
        make_comparison_plot(boltz_results, colabfold_results, OUTPUT_DIR)

    # --- Save all results ---
    all_results = {
        "boltz": boltz_results,
        "colabfold": colabfold_results,
    }
    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {OUTPUT_DIR}/comparison_results.json")


if __name__ == "__main__":
    main()
