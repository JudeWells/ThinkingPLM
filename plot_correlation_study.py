#!/usr/bin/env python3
"""Build correlation plots for the sampled-sequence Boltz2 study.

For each target, scatter plots of:
  - pipeline total_energy vs Boltz2 ipSAE
  - pipeline total_energy vs Boltz2 ipTM
Combined: ipTM vs ipSAE across all samples.

Uses the same Dunbrack-d0res ipSAE as build_boltz2_html.py via
pipeline.utils.compute_ipsae.
"""

import json
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/mnt/disk2/ThinkingPLM")
MANIFEST = BASE / "boltz2_corr_manifest.json"
BOLTZ_OUT = BASE / "boltz2_corr_output" / "boltz_results_boltz2_corr_input" / "predictions"
OUT_PNG = BASE / "correlation_study_boltz2.png"
OUT_PNG_PER_TARGET = BASE / "correlation_study_per_target.png"

_utils_spec = importlib.util.spec_from_file_location(
    "pipeline_utils_standalone", BASE / "pipeline" / "utils.py"
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_mod)
compute_ipsae = _utils_mod.compute_ipsae


def load_metrics(name, binder_len):
    sub = BOLTZ_OUT / name
    conf_f = sub / f"confidence_{name}_model_0.json"
    pae_f = sub / f"pae_{name}_model_0.npz"
    plddt_f = sub / f"plddt_{name}_model_0.npz"
    if not conf_f.exists():
        return None
    with open(conf_f) as f:
        conf = json.load(f)
    out = {"iptm": conf.get("iptm"), "ptm": conf.get("ptm")}
    if pae_f.exists():
        pae = np.load(pae_f)["pae"]
        a = np.arange(binder_len)
        b = np.arange(binder_len, pae.shape[0])
        out["ipsae"] = compute_ipsae(pae, a, b, pae_cutoff=10.0)
    else:
        out["ipsae"] = None
    if plddt_f.exists():
        pl = np.load(plddt_f)["plddt"]
        if pl.max() <= 1.0:
            pl = pl * 100.0
        out["binder_plddt"] = float(np.mean(pl[:binder_len]))
    else:
        out["binder_plddt"] = None
    return out


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    rows = []
    skipped = 0
    for entry in manifest:
        m = load_metrics(entry["name"], entry["binder_len"])
        if m is None:
            skipped += 1
            continue
        rows.append({**entry, **m})

    print(f"Loaded {len(rows)} predictions ({skipped} skipped/missing)")

    targets = sorted({r["target_id"] for r in rows})
    n_targets = len(targets)

    # ============================================================
    # Figure 1: per-target panels (n_targets rows × 2 cols: ipsae, iptm)
    # ============================================================
    fig, axes = plt.subplots(n_targets, 2, figsize=(11, 3.0 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    color_map = {"top20": "#C62828", "random": "#1565C0"}

    for ti, tid in enumerate(targets):
        sub = [r for r in rows if r["target_id"] == tid]
        for stype in ["random", "top20"]:
            pts = [r for r in sub if r["sample_type"] == stype]
            xs = [r["total_energy"] for r in pts]
            for col, (key, ylabel) in enumerate([
                ("ipsae", "Boltz2 ipSAE (Dunbrack)"),
                ("iptm", "Boltz2 ipTM"),
            ]):
                ys = [r[key] for r in pts if r[key] is not None]
                xs_clean = [r["total_energy"] for r in pts if r[key] is not None]
                axes[ti, col].scatter(
                    xs_clean, ys, s=42, alpha=0.75, color=color_map[stype],
                    edgecolor="black", linewidth=0.4, label=f"{stype} (n={len(ys)})",
                )

        # Add Pearson + Spearman per panel using all samples for that target
        for col, key in enumerate(["ipsae", "iptm"]):
            xs_all = [r["total_energy"] for r in sub if r[key] is not None]
            ys_all = [r[key] for r in sub if r[key] is not None]
            r_p = pearson(xs_all, ys_all)
            r_s = spearman(xs_all, ys_all)
            label = "ipSAE" if key == "ipsae" else "ipTM"
            axes[ti, col].set_title(
                f"{tid} · energy vs {label}\nPearson r = {r_p:+.3f}  Spearman ρ = {r_s:+.3f}  (n={len(xs_all)})",
                fontsize=10,
            )
            axes[ti, col].set_xlabel("Pipeline total energy (ESMFold LIS)")
            ylab = "Boltz2 ipSAE" if key == "ipsae" else "Boltz2 ipTM"
            axes[ti, col].set_ylabel(ylab)
            axes[ti, col].grid(alpha=0.3)
            if ti == 0:
                axes[ti, col].legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(
        "Pipeline energy vs Boltz2 metrics — sampled sequences per target\n"
        "(15 from top 20% by lowest energy, 15 random from full pool)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG_PER_TARGET, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PNG_PER_TARGET}")

    # ============================================================
    # Figure 2: combined 1×3 (energy-iptm, energy-ipsae, iptm-ipsae)
    # ============================================================
    target_colors = {
        t: c for t, c in zip(targets, plt.cm.tab10(np.linspace(0, 1, len(targets))))
    }
    type_marker = {"top20": "o", "random": "^"}

    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 5.2))

    def panel(ax, xkey, ykey, xlabel, ylabel, title):
        for r in rows:
            if r[xkey] is None or r[ykey] is None:
                continue
            ax.scatter(
                r[xkey], r[ykey],
                color=target_colors[r["target_id"]],
                marker=type_marker[r["sample_type"]],
                s=50, alpha=0.75, edgecolor="black", linewidth=0.4,
            )
        xs = [r[xkey] for r in rows if r[xkey] is not None and r[ykey] is not None]
        ys = [r[ykey] for r in rows if r[xkey] is not None and r[ykey] is not None]
        r_p = pearson(xs, ys)
        r_s = spearman(xs, ys)
        ax.set_title(f"{title}\nPearson r = {r_p:+.3f}  Spearman ρ = {r_s:+.3f}  (n={len(xs)})",
                     fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="gray", alpha=0.6, linewidth=1)

    panel(ax2[0], "total_energy", "iptm",
          "Pipeline total energy (ESMFold LIS)", "Boltz2 ipTM",
          "Pipeline energy vs Boltz2 ipTM")
    panel(ax2[1], "total_energy", "ipsae",
          "Pipeline total energy (ESMFold LIS)", "Boltz2 ipSAE",
          "Pipeline energy vs Boltz2 ipSAE")
    panel(ax2[2], "iptm", "ipsae",
          "Boltz2 ipTM", "Boltz2 ipSAE",
          "Boltz2 ipTM vs ipSAE")

    target_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=target_colors[t],
                   markersize=8, label=t, markeredgecolor="black", markeredgewidth=0.4)
        for t in targets
    ]
    type_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="gray",
                   markersize=8, label="top 20%", markeredgecolor="black", markeredgewidth=0.4),
        plt.Line2D([], [], marker="^", linestyle="", color="gray",
                   markersize=8, label="random", markeredgecolor="black", markeredgewidth=0.4),
    ]
    fig2.legend(
        handles=target_handles + type_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(target_handles) + 2, frameon=False, fontsize=9,
    )
    fig2.suptitle("Combined: 180 sampled sequences across 6 targets",
                  y=1.02, fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")

    # Print correlation summary
    print("\n=== Per-target correlations ===")
    print(f"{'Target':<22} {'n':>4} {'r(E,ipSAE)':>14} {'ρ(E,ipSAE)':>14} {'r(E,ipTM)':>14} {'ρ(E,ipTM)':>14}")
    for tid in targets:
        sub = [r for r in rows if r["target_id"] == tid]
        es = [r["total_energy"] for r in sub if r["ipsae"] is not None]
        ips = [r["ipsae"] for r in sub if r["ipsae"] is not None]
        ipt = [r["iptm"] for r in sub if r["iptm"] is not None]
        es2 = [r["total_energy"] for r in sub if r["iptm"] is not None]
        print(f"{tid:<22} {len(sub):>4} "
              f"{pearson(es, ips):>+14.3f} {spearman(es, ips):>+14.3f} "
              f"{pearson(es2, ipt):>+14.3f} {spearman(es2, ipt):>+14.3f}")

    # Overall
    es = [r["total_energy"] for r in rows if r["ipsae"] is not None]
    ips = [r["ipsae"] for r in rows if r["ipsae"] is not None]
    ipt_e = [r["total_energy"] for r in rows if r["iptm"] is not None]
    ipt = [r["iptm"] for r in rows if r["iptm"] is not None]
    print(f"\n{'Overall':<22} {len(rows):>4} "
          f"{pearson(es, ips):>+14.3f} {spearman(es, ips):>+14.3f} "
          f"{pearson(ipt_e, ipt):>+14.3f} {spearman(ipt_e, ipt):>+14.3f}")


if __name__ == "__main__":
    main()
