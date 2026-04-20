#!/usr/bin/env python3
"""Correlation plots for the 180-sample sampled set, now including BindCraft-AF2.

Reads:
  - boltz2_corr_manifest.json                 (total_energy, sample_type, etc.)
  - boltz2_corr_output/.../predictions/<name>/  (Boltz2 PAE/plddt/conf)
  - af2_template_corr_output/<name>/          (AF2 model_<n>.npz, summary.json)

Builds:
  - correlation_study_with_af2.png   (1x4 combined view + per-target subplot grid)
  - correlation_study_per_target_af2.png  (6 targets x 3 cols: E-vs-Boltz2, E-vs-AF2, Boltz2-vs-AF2 for ipSAE)
  - correlation_study_with_af2.csv    (per-sequence table)
"""

import json
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/mnt/disk2/ThinkingPLM")
MANIFEST = BASE / "boltz2_corr_manifest.json"
BOLTZ_PRED_DIR = BASE / "boltz2_corr_output" / "boltz_results_boltz2_corr_input" / "predictions"
AF2_DIR = BASE / "af2_template_corr_output"
OUT_PNG_COMBINED = BASE / "correlation_study_with_af2.png"
OUT_PNG_PER_TARGET = BASE / "correlation_study_per_target_af2.png"
OUT_CSV = BASE / "correlation_study_with_af2.csv"

_utils_spec = importlib.util.spec_from_file_location("u", BASE / "pipeline" / "utils.py")
_u = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_u)
compute_ipsae = _u.compute_ipsae


def load_boltz_metrics(name, binder_len):
    sub = BOLTZ_PRED_DIR / name
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
        # Boltz2 layout: binder first (chain A), target second (chain B)
        out["ipsae"] = compute_ipsae(
            pae, np.arange(binder_len), np.arange(binder_len, pae.shape[0]),
            pae_cutoff=10.0,
        )
    else:
        out["ipsae"] = None
    if plddt_f.exists():
        pl = np.load(plddt_f)["plddt"]
        if pl.max() <= 1.0: pl = pl * 100.0
        out["binder_plddt"] = float(np.mean(pl[:binder_len]))
    else:
        out["binder_plddt"] = None
    return out


def load_af2_metrics(name, binder_len):
    sub = AF2_DIR / name
    summary_f = sub / "summary.json"
    if not summary_f.exists():
        return None
    with open(summary_f) as f:
        summary = json.load(f)
    target_len = summary["target_len"]

    ipsaes, iptms, plddts = [], [], []
    for m in summary["models"]:
        npz = sub / f"model_{m['model']}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        pae = d["pae"]
        plddt = d["plddt"]
        if plddt.max() <= 1.0: plddt = plddt * 100.0
        # AF2 layout: target first, binder second
        ipsaes.append(compute_ipsae(
            pae,
            np.arange(target_len, target_len + binder_len),
            np.arange(0, target_len),
            pae_cutoff=10.0,
        ))
        iptms.append(float(d["iptm"]))
        plddts.append(float(np.mean(plddt[target_len:])))
    if not ipsaes:
        return None
    return {
        "ipsae": float(np.mean(ipsaes)),
        "iptm": float(np.mean(iptms)),
        "binder_plddt": float(np.mean(plddts)),
    }


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2: return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    rows = []
    for entry in manifest:
        name = entry["name"]
        binder_len = entry["binder_len"]
        boltz = load_boltz_metrics(name, binder_len) or {}
        af2 = load_af2_metrics(name, binder_len) or {}
        rows.append({
            "name": name,
            "target": entry["target_id"],
            "sample_type": entry["sample_type"],
            "binder_len": binder_len,
            "energy": entry["total_energy"],
            "boltz_ipsae": boltz.get("ipsae"),
            "boltz_iptm": boltz.get("iptm"),
            "boltz_binder_plddt": boltz.get("binder_plddt"),
            "af2_ipsae": af2.get("ipsae"),
            "af2_iptm": af2.get("iptm"),
            "af2_binder_plddt": af2.get("binder_plddt"),
        })

    n_total = len(rows)
    n_boltz = sum(1 for r in rows if r["boltz_ipsae"] is not None)
    n_af2 = sum(1 for r in rows if r["af2_ipsae"] is not None)
    print(f"Loaded {n_total} entries — Boltz2: {n_boltz}, AF2: {n_af2}")

    # CSV
    import csv as csv_mod
    with open(OUT_CSV, "w") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}")

    if n_af2 < 5:
        print("Not enough AF2 results yet — re-run later.")
        return

    targets = sorted({r["target"] for r in rows})
    target_colors = {t: c for t, c in zip(targets, plt.cm.tab10(np.linspace(0, 1, len(targets))))}
    type_marker = {"top20": "o", "random": "^"}

    # ============================================================
    # Combined 1x4: energy↔boltz_ipsae, energy↔af2_ipsae, boltz_ipsae↔af2_ipsae, energy↔af2_iptm
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

    def panel(ax, xkey, ykey, xlabel, ylabel, title):
        for r in rows:
            x, y = r.get(xkey), r.get(ykey)
            if x is None or y is None: continue
            ax.scatter(x, y,
                       color=target_colors[r["target"]],
                       marker=type_marker[r["sample_type"]],
                       s=42, alpha=0.7, edgecolor="black", linewidth=0.4)
        xs = [r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        ys = [r[ykey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        if len(xs) >= 2:
            r_p, r_s = pearson(xs, ys), spearman(xs, ys)
            ax.set_title(f"{title}\nPearson r = {r_p:+.3f}  Spearman ρ = {r_s:+.3f}  (n={len(xs)})", fontsize=10)
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="gray", alpha=0.6, linewidth=1)
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    panel(axes[0], "energy", "boltz_ipsae", "Pipeline energy", "Boltz2 ipSAE", "Energy vs Boltz2 ipSAE")
    panel(axes[1], "energy", "af2_ipsae", "Pipeline energy", "BindCraft-AF2 ipSAE", "Energy vs AF2 ipSAE")
    panel(axes[2], "boltz_ipsae", "af2_ipsae", "Boltz2 ipSAE", "BindCraft-AF2 ipSAE", "Boltz2 vs AF2 ipSAE")
    panel(axes[3], "energy", "af2_iptm", "Pipeline energy", "BindCraft-AF2 ipTM", "Energy vs AF2 ipTM")

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
    fig.suptitle("Sampled correlation study (180 sequences) — with BindCraft-AF2",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG_COMBINED, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG_COMBINED}")

    # ============================================================
    # Per-target grid: 6 rows × 3 cols
    # ============================================================
    n_t = len(targets)
    fig2, axes2 = plt.subplots(n_t, 3, figsize=(13, 3.0 * n_t))
    if n_t == 1:
        axes2 = axes2.reshape(1, -1)
    color_map = {"top20": "#C62828", "random": "#1565C0"}

    for ti, tid in enumerate(targets):
        sub_rows = [r for r in rows if r["target"] == tid]
        for col, (xk, yk, xlab, ylab, title_name) in enumerate([
            ("energy", "boltz_ipsae", "Pipeline energy", "Boltz2 ipSAE", "energy vs Boltz2 ipSAE"),
            ("energy", "af2_ipsae", "Pipeline energy", "BindCraft-AF2 ipSAE", "energy vs AF2 ipSAE"),
            ("boltz_ipsae", "af2_ipsae", "Boltz2 ipSAE", "BindCraft-AF2 ipSAE", "Boltz2 vs AF2 ipSAE"),
        ]):
            ax = axes2[ti, col]
            for stype in ["random", "top20"]:
                pts = [r for r in sub_rows if r["sample_type"] == stype and r.get(xk) is not None and r.get(yk) is not None]
                ax.scatter(
                    [r[xk] for r in pts], [r[yk] for r in pts],
                    s=38, alpha=0.75, color=color_map[stype],
                    edgecolor="black", linewidth=0.4,
                    label=f"{stype}",
                )
            xs_all = [r[xk] for r in sub_rows if r.get(xk) is not None and r.get(yk) is not None]
            ys_all = [r[yk] for r in sub_rows if r.get(xk) is not None and r.get(yk) is not None]
            r_p, r_s = pearson(xs_all, ys_all), spearman(xs_all, ys_all)
            ax.set_title(f"{tid} · {title_name}\nr={r_p:+.3f}  ρ={r_s:+.3f}  (n={len(xs_all)})", fontsize=9)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            if ti == 0 and col == 0:
                ax.legend(loc="best", fontsize=7, frameon=True)

    fig2.suptitle(
        "Per-target correlations — sampled set (180 sequences) with BindCraft-AF2",
        y=1.0, fontsize=12, fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(OUT_PNG_PER_TARGET, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG_PER_TARGET}")

    # Per-target correlation summary
    print("\n=== Per-target correlations (Pearson) ===")
    print(f"{'Target':<22} {'n':>4} {'r(E,B_ips)':>12} {'r(E,A_ips)':>12} {'r(B,A_ips)':>12} {'r(E,B_ipt)':>12} {'r(E,A_ipt)':>12}")
    for t in targets:
        sub = [r for r in rows if r["target"] == t and r.get("af2_ipsae") is not None]
        n = len(sub)
        if n < 2: continue
        es = [r["energy"] for r in sub]
        bi = [r["boltz_ipsae"] for r in sub]
        ai = [r["af2_ipsae"] for r in sub]
        bt = [r["boltz_iptm"] for r in sub]
        at = [r["af2_iptm"] for r in sub]
        print(f"{t:<22} {n:>4} {pearson(es,bi):>+12.3f} {pearson(es,ai):>+12.3f} {pearson(bi,ai):>+12.3f} {pearson(es,bt):>+12.3f} {pearson(es,at):>+12.3f}")

    sub = [r for r in rows if r.get("af2_ipsae") is not None]
    n = len(sub)
    es = [r["energy"] for r in sub]
    bi = [r["boltz_ipsae"] for r in sub]
    ai = [r["af2_ipsae"] for r in sub]
    bt = [r["boltz_iptm"] for r in sub]
    at = [r["af2_iptm"] for r in sub]
    print(f"\n{'Overall':<22} {n:>4} {pearson(es,bi):>+12.3f} {pearson(es,ai):>+12.3f} {pearson(bi,ai):>+12.3f} {pearson(es,bt):>+12.3f} {pearson(es,at):>+12.3f}")


if __name__ == "__main__":
    main()
