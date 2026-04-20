#!/usr/bin/env python3
"""Scatter plots: pipeline energy vs Boltz2 ipTM/ipSAE, and ipTM vs ipSAE.

Uses the same Boltz2 aggregation as build_boltz2_html.py (mean across the
5 diffusion samples, with ipSAE computed via Dunbrack d0res).
"""

import json
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/mnt/disk2/ThinkingPLM")
BOLTZ_OUT = BASE / "boltz2_output" / "boltz_results_boltz2_input" / "predictions"
INPUT_JSON = BASE / "colabfold_input" / "best_sequences.json"
OUT_PNG = BASE / "correlations_boltz2.png"

_utils_spec = importlib.util.spec_from_file_location(
    "pipeline_utils_standalone", BASE / "pipeline" / "utils.py"
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_mod)
compute_ipsae = _utils_mod.compute_ipsae


def get_boltz_name(r):
    safe = r["campaign"].replace("/", "_")
    return f"sc_rep3_{safe}" if r["target_id"] == "2GDZ" else f"{r['target_id']}_{safe}"


def load_sample_metrics(name, binder_len):
    sub = BOLTZ_OUT / name
    ipsaes, iptms, plddts = [], [], []
    for i in range(20):
        conf_f = sub / f"confidence_{name}_model_{i}.json"
        pae_f = sub / f"pae_{name}_model_{i}.npz"
        plddt_f = sub / f"plddt_{name}_model_{i}.npz"
        if not conf_f.exists():
            break
        with open(conf_f) as f:
            conf = json.load(f)
        iptms.append(conf.get("iptm"))
        if pae_f.exists():
            pae = np.load(pae_f)["pae"]
            a = np.arange(binder_len)
            b = np.arange(binder_len, pae.shape[0])
            ipsaes.append(compute_ipsae(pae, a, b, pae_cutoff=10.0))
        if plddt_f.exists():
            pl = np.load(plddt_f)["plddt"]
            if pl.max() <= 1.0:
                pl = pl * 100.0
            plddts.append(float(np.mean(pl[:binder_len])))
    return {
        "ipsae": float(np.mean(ipsaes)) if ipsaes else None,
        "iptm": float(np.mean(iptms)) if iptms else None,
        "binder_plddt": float(np.mean(plddts)) if plddts else None,
    }


def pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    with open(INPUT_JSON) as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        name = get_boltz_name(e)
        m = load_sample_metrics(name, len(e["binder_seq"]))
        rows.append({
            "campaign": e["campaign"],
            "target": e["target_id"],
            "method": e["campaign"].split("/")[-1],
            "energy": e["total_energy"],
            "pipeline_lis": e["lis_energy"],
            "ipsae": m["ipsae"],
            "iptm": m["iptm"],
            "binder_plddt": m["binder_plddt"],
        })

    # Color / marker by target and method
    targets = sorted({r["target"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    target_colors = {
        t: c for t, c in zip(targets, plt.cm.tab10(np.linspace(0, 1, len(targets))))
    }
    method_markers = {
        "bandit_grpo": "o",
        "bandit_bt": "s",
        "proposal_bandit": "^",
        "random_greedy": "D",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    def scatter_plot(ax, xkey, ykey, xlabel, ylabel, title):
        xs, ys = [], []
        for r in rows:
            x, y = r[xkey], r[ykey]
            if x is None or y is None or np.isnan(x) or np.isnan(y):
                continue
            xs.append(x); ys.append(y)
            ax.scatter(
                x, y,
                color=target_colors[r["target"]],
                marker=method_markers.get(r["method"], "o"),
                s=70, alpha=0.8, edgecolor="black", linewidth=0.5,
            )
        if len(xs) >= 2:
            r_val = pearson(xs, ys)
            ax.set_title(f"{title}\nPearson r = {r_val:.3f}  (n={len(xs)})")
            # best-fit line
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="gray", alpha=0.6, linewidth=1)
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    scatter_plot(axes[0], "energy", "iptm",
                 "Pipeline total energy (ESMFold LIS)", "Boltz2 ipTM (mean of 5)",
                 "Pipeline energy vs Boltz2 ipTM")
    scatter_plot(axes[1], "energy", "ipsae",
                 "Pipeline total energy (ESMFold LIS)", "Boltz2 ipSAE (Dunbrack d0res)",
                 "Pipeline energy vs Boltz2 ipSAE")
    scatter_plot(axes[2], "iptm", "ipsae",
                 "Boltz2 ipTM", "Boltz2 ipSAE",
                 "Boltz2 ipTM vs ipSAE")

    # Shared legend for targets + methods
    target_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=target_colors[t],
                   markersize=8, label=t, markeredgecolor="black", markeredgewidth=0.5)
        for t in targets
    ]
    method_handles = [
        plt.Line2D([], [], marker=m, linestyle="", color="gray",
                   markersize=8, label=method, markeredgecolor="black", markeredgewidth=0.5)
        for method, m in method_markers.items() if method in methods
    ]
    fig.legend(
        handles=target_handles + method_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(target_handles) + len(method_handles),
        frameon=False, fontsize=9,
    )
    fig.suptitle("Pipeline energy vs Boltz2 metrics (35 campaigns, 5 diffusion samples each)",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")

    # Also print correlation summary
    print("\nCorrelations:")
    for (xk, yk) in [("energy", "iptm"), ("energy", "ipsae"), ("iptm", "ipsae"),
                     ("energy", "binder_plddt"), ("binder_plddt", "ipsae")]:
        xs = [r[xk] for r in rows if r[xk] is not None and r[yk] is not None]
        ys = [r[yk] for r in rows if r[xk] is not None and r[yk] is not None]
        print(f"  {xk:>14} vs {yk:<14}  r = {pearson(xs, ys):+.3f}  (n={len(xs)})")


if __name__ == "__main__":
    main()
