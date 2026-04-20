#!/usr/bin/env python3
"""ipTM-focused cross-predictor comparison.

ESMFold ipTM is not available — the pipeline runs only stored LIS energy and
pLDDT, not ipTM. This script compares Boltz2 ipTM vs BindCraft-AF2 ipTM
across the 180 sampled sequences, plus their relationship to pipeline energy.

Reads from boltz2_corr_manifest.json + Boltz2/AF2 outputs.
Builds:
  - iptm_comparison.png         (combined 1x4 view)
  - iptm_comparison_per_target.png  (per-target panels)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/mnt/disk2/ThinkingPLM")
MANIFEST = BASE / "boltz2_corr_manifest.json"
BOLTZ_PRED_DIR = BASE / "boltz2_corr_output" / "boltz_results_boltz2_corr_input" / "predictions"
AF2_DIR = BASE / "af2_template_corr_output"
OUT_PNG = BASE / "iptm_comparison.png"
OUT_PNG_PER_TARGET = BASE / "iptm_comparison_per_target.png"


def load_boltz_iptm(name):
    sub = BOLTZ_PRED_DIR / name
    conf_f = sub / f"confidence_{name}_model_0.json"
    if not conf_f.exists():
        return None
    with open(conf_f) as f:
        return json.load(f).get("iptm")


def load_af2_iptm(name):
    sub = AF2_DIR / name
    summary_f = sub / "summary.json"
    if not summary_f.exists():
        return None
    with open(summary_f) as f:
        s = json.load(f)
    iptms = [m["iptm"] for m in s["models"] if m.get("iptm") is not None and not np.isnan(m["iptm"])]
    return float(np.mean(iptms)) if iptms else None


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest:
        name = entry["name"]
        rows.append({
            "name": name,
            "target": entry["target_id"],
            "sample_type": entry["sample_type"],
            "energy": entry["total_energy"],
            "boltz_iptm": load_boltz_iptm(name),
            "af2_iptm": load_af2_iptm(name),
        })

    targets = sorted({r["target"] for r in rows})
    target_colors = {t: c for t, c in zip(targets, plt.cm.tab10(np.linspace(0, 1, len(targets))))}
    type_marker = {"top20": "o", "random": "^"}

    # ============================================================
    # Figure 1: combined 1x3
    #  (a) energy vs Boltz2 ipTM
    #  (b) energy vs AF2 ipTM
    #  (c) Boltz2 ipTM vs AF2 ipTM (with y=x diagonal)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    def panel(ax, xkey, ykey, xlabel, ylabel, title, identity=False):
        for r in rows:
            x, y = r.get(xkey), r.get(ykey)
            if x is None or y is None: continue
            ax.scatter(x, y,
                       color=target_colors[r["target"]],
                       marker=type_marker[r["sample_type"]],
                       s=46, alpha=0.75, edgecolor="black", linewidth=0.4)
        xs = [r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        ys = [r[ykey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        r_p, r_s = pearson(xs, ys), spearman(xs, ys)
        ax.set_title(f"{title}\nPearson r = {r_p:+.3f}  Spearman ρ = {r_s:+.3f}  (n={len(xs)})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if identity and len(xs) >= 2:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, linewidth=1, label="y=x")
            ax.set_xlim(lo - 0.02, hi + 0.02)
            ax.set_ylim(lo - 0.02, hi + 0.02)
            ax.set_aspect("equal", adjustable="box")
        elif len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="gray", alpha=0.6, linewidth=1)

    panel(axes[0], "energy", "boltz_iptm",
          "Pipeline energy (ESMFold LIS)", "Boltz2 ipTM",
          "Energy vs Boltz2 ipTM")
    panel(axes[1], "energy", "af2_iptm",
          "Pipeline energy (ESMFold LIS)", "BindCraft-AF2 ipTM",
          "Energy vs BindCraft-AF2 ipTM")
    panel(axes[2], "boltz_iptm", "af2_iptm",
          "Boltz2 ipTM", "BindCraft-AF2 ipTM",
          "Boltz2 vs BindCraft-AF2 ipTM",
          identity=True)

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
    fig.legend(handles=target_handles + type_handles,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(target_handles) + 2, frameon=False, fontsize=9)
    fig.suptitle(
        "ipTM cross-predictor comparison — 180 sampled sequences\n"
        "(ESMFold ipTM not available; pipeline records only LIS energy)",
        y=1.04, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")

    # ============================================================
    # Figure 2: per-target panels (6 rows × 3 cols, same layout)
    # ============================================================
    n_t = len(targets)
    fig2, axes2 = plt.subplots(n_t, 3, figsize=(13, 3.0 * n_t))
    if n_t == 1:
        axes2 = axes2.reshape(1, -1)
    color_map = {"top20": "#C62828", "random": "#1565C0"}

    for ti, tid in enumerate(targets):
        sub_rows = [r for r in rows if r["target"] == tid]
        for col, (xk, yk, xlab, ylab, title_name, identity) in enumerate([
            ("energy", "boltz_iptm", "Pipeline energy", "Boltz2 ipTM", "energy vs Boltz2 ipTM", False),
            ("energy", "af2_iptm", "Pipeline energy", "BindCraft-AF2 ipTM", "energy vs AF2 ipTM", False),
            ("boltz_iptm", "af2_iptm", "Boltz2 ipTM", "BindCraft-AF2 ipTM", "Boltz2 vs AF2 ipTM", True),
        ]):
            ax = axes2[ti, col]
            for stype in ["random", "top20"]:
                pts = [r for r in sub_rows if r["sample_type"] == stype and r.get(xk) is not None and r.get(yk) is not None]
                ax.scatter(
                    [r[xk] for r in pts], [r[yk] for r in pts],
                    s=38, alpha=0.75, color=color_map[stype],
                    edgecolor="black", linewidth=0.4, label=stype,
                )
            xs_all = [r[xk] for r in sub_rows if r.get(xk) is not None and r.get(yk) is not None]
            ys_all = [r[yk] for r in sub_rows if r.get(xk) is not None and r.get(yk) is not None]
            r_p, r_s = pearson(xs_all, ys_all), spearman(xs_all, ys_all)
            ax.set_title(f"{tid} · {title_name}\nr={r_p:+.3f}  ρ={r_s:+.3f}  (n={len(xs_all)})", fontsize=9)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if identity and len(xs_all) >= 2:
                lo = min(min(xs_all), min(ys_all))
                hi = max(max(xs_all), max(ys_all))
                ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, linewidth=0.8)
            if ti == 0 and col == 0:
                ax.legend(loc="best", fontsize=7, frameon=True)

    fig2.suptitle("ipTM per-target comparison — sampled set (180 sequences)",
                  y=1.0, fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_PNG_PER_TARGET, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG_PER_TARGET}")

    # Per-target correlation summary
    print("\n=== Per-target ipTM correlations (Pearson) ===")
    print(f"{'Target':<22} {'n':>4} {'r(E,B_iptm)':>13} {'r(E,A_iptm)':>13} {'r(B,A) iptm':>13}  ρ(B,A)")
    for t in targets:
        sub = [r for r in rows if r["target"] == t and r.get("boltz_iptm") is not None and r.get("af2_iptm") is not None]
        n = len(sub)
        if n < 2: continue
        es = [r["energy"] for r in sub]
        bt = [r["boltz_iptm"] for r in sub]
        at = [r["af2_iptm"] for r in sub]
        print(f"{t:<22} {n:>4} {pearson(es,bt):>+13.3f} {pearson(es,at):>+13.3f} {pearson(bt,at):>+13.3f}  {spearman(bt,at):+.3f}")

    sub = [r for r in rows if r.get("boltz_iptm") is not None and r.get("af2_iptm") is not None]
    n = len(sub)
    es = [r["energy"] for r in sub]
    bt = [r["boltz_iptm"] for r in sub]
    at = [r["af2_iptm"] for r in sub]
    print(f"\n{'Overall':<22} {n:>4} {pearson(es,bt):>+13.3f} {pearson(es,at):>+13.3f} {pearson(bt,at):>+13.3f}  {spearman(bt,at):+.3f}")


if __name__ == "__main__":
    main()
