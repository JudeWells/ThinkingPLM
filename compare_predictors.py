#!/usr/bin/env python3
"""Compare BindCraft-style AF2 ipSAE vs Boltz2 ipSAE vs ESMFold (pipeline) energy.

For each best-binder campaign, gather:
  - Pipeline metrics (total_energy, lis_energy) from best_sequences.json
  - Boltz2 metrics (mean over 5 samples, ipSAE Dunbrack, iptm, binder pLDDT)
  - BindCraft AF2 metrics (mean over 2 models, ipSAE Dunbrack, iptm, binder pLDDT)

Outputs:
  - predictor_comparison.csv     - per-campaign table
  - predictor_comparison.png     - 2x3 scatter plot grid
"""

import json
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/mnt/disk2/ThinkingPLM")
INPUT_JSON = BASE / "colabfold_input" / "best_sequences.json"
BOLTZ_PRED_DIR = BASE / "boltz2_output" / "boltz_results_boltz2_input" / "predictions"
AF2_DIR = BASE / "af2_template_output"
OUT_CSV = BASE / "predictor_comparison.csv"
OUT_PNG = BASE / "predictor_comparison.png"

_utils_spec = importlib.util.spec_from_file_location(
    "u", BASE / "pipeline" / "utils.py"
)
_u = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_u)
compute_ipsae = _u.compute_ipsae


def get_safe_name(entry):
    safe = entry["campaign"].replace("/", "_")
    return f"sc_rep3_{safe}" if entry["target_id"] == "2GDZ" else f"{entry['target_id']}_{safe}"


def load_boltz2_metrics(name, binder_len):
    """Mean ipSAE/iptm/binder_plddt over the 5 boltz2 samples."""
    sub = BOLTZ_PRED_DIR / name
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
            ipsaes.append(compute_ipsae(pae, np.arange(binder_len),
                                        np.arange(binder_len, pae.shape[0]),
                                        pae_cutoff=10.0))
        if plddt_f.exists():
            pl = np.load(plddt_f)["plddt"]
            if pl.max() <= 1.0: pl = pl * 100.0
            # In Boltz2 output, binder is chain A (first)
            plddts.append(float(np.mean(pl[:binder_len])))
    if not ipsaes:
        return None
    return {
        "ipsae": float(np.mean(ipsaes)),
        "iptm": float(np.mean(iptms)),
        "binder_plddt": float(np.mean(plddts)),
    }


def load_af2_metrics(name, binder_len):
    """Mean ipSAE/iptm/binder_plddt over the 2 BindCraft AF2 models.

    In BindCraft layout, target comes first, binder comes second:
      indices [0, target_len) = target, [target_len, total) = binder.
    """
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
        if plddt.max() <= 1.0:
            plddt = plddt * 100.0
        # binder is the trailing residues
        ipsaes.append(compute_ipsae(
            pae,
            np.arange(target_len, target_len + binder_len),  # binder
            np.arange(0, target_len),                        # target
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
    with open(INPUT_JSON) as f:
        entries = json.load(f)

    rows = []
    for entry in entries:
        name = get_safe_name(entry)
        binder_len = len(entry["binder_seq"])
        boltz = load_boltz2_metrics(name, binder_len) or {}
        af2 = load_af2_metrics(name, binder_len) or {}
        rows.append({
            "campaign": entry["campaign"],
            "target": entry["target_id"],
            "method": entry["campaign"].split("/")[-1],
            "energy": entry["total_energy"],
            "boltz_ipsae": boltz.get("ipsae"),
            "boltz_iptm": boltz.get("iptm"),
            "boltz_binder_plddt": boltz.get("binder_plddt"),
            "af2_ipsae": af2.get("ipsae"),
            "af2_iptm": af2.get("iptm"),
            "af2_binder_plddt": af2.get("binder_plddt"),
        })

    # Save CSV
    import csv as csv_mod
    with open(OUT_CSV, "w") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")

    n_with_af2 = sum(1 for r in rows if r["af2_ipsae"] is not None)
    print(f"  with Boltz2: {sum(1 for r in rows if r['boltz_ipsae'] is not None)}/{len(rows)}")
    print(f"  with AF2-tpl: {n_with_af2}/{len(rows)}")

    if n_with_af2 < 5:
        print("\nNot enough AF2 predictions yet — re-run after they finish.")
        return

    # ============================================================
    # Plot 2x3 grid:
    # row 0: energy vs (boltz_ipsae, af2_ipsae, boltz_vs_af2 ipSAE)
    # row 1: energy vs (boltz_iptm,  af2_iptm,  boltz_vs_af2 iptm)
    # ============================================================
    targets = sorted({r["target"] for r in rows})
    target_colors = {t: c for t, c in zip(targets, plt.cm.tab10(np.linspace(0, 1, len(targets))))}
    method_markers = {"bandit_grpo": "o", "bandit_bt": "s", "proposal_bandit": "^", "random_greedy": "D"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def panel(ax, xkey, ykey, xlabel, ylabel, title):
        for r in rows:
            x, y = r.get(xkey), r.get(ykey)
            if x is None or y is None: continue
            ax.scatter(x, y,
                       color=target_colors[r["target"]],
                       marker=method_markers.get(r["method"], "o"),
                       s=70, alpha=0.8, edgecolor="black", linewidth=0.5)
        xs = [r[xkey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        ys = [r[ykey] for r in rows if r.get(xkey) is not None and r.get(ykey) is not None]
        if len(xs) >= 2:
            r_p = pearson(xs, ys)
            r_s = spearman(xs, ys)
            ax.set_title(f"{title}\nPearson r = {r_p:+.3f}  Spearman ρ = {r_s:+.3f}  (n={len(xs)})", fontsize=10)
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(coef, xx), "--", color="gray", alpha=0.6, linewidth=1)
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # Row 0: ipSAE vs energy + cross-predictor
    panel(axes[0, 0], "energy", "boltz_ipsae",
          "Pipeline energy (ESMFold LIS)", "Boltz2 ipSAE",
          "Energy vs Boltz2 ipSAE")
    panel(axes[0, 1], "energy", "af2_ipsae",
          "Pipeline energy (ESMFold LIS)", "BindCraft-AF2 ipSAE",
          "Energy vs BindCraft-AF2 ipSAE")
    panel(axes[0, 2], "boltz_ipsae", "af2_ipsae",
          "Boltz2 ipSAE", "BindCraft-AF2 ipSAE",
          "Boltz2 vs BindCraft-AF2 ipSAE")

    # Row 1: ipTM vs energy + cross-predictor
    panel(axes[1, 0], "energy", "boltz_iptm",
          "Pipeline energy (ESMFold LIS)", "Boltz2 ipTM",
          "Energy vs Boltz2 ipTM")
    panel(axes[1, 1], "energy", "af2_iptm",
          "Pipeline energy (ESMFold LIS)", "BindCraft-AF2 ipTM",
          "Energy vs BindCraft-AF2 ipTM")
    panel(axes[1, 2], "boltz_iptm", "af2_iptm",
          "Boltz2 ipTM", "BindCraft-AF2 ipTM",
          "Boltz2 vs BindCraft-AF2 ipTM")

    target_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=target_colors[t],
                   markersize=8, label=t, markeredgecolor="black", markeredgewidth=0.4)
        for t in targets
    ]
    method_handles = [
        plt.Line2D([], [], marker=mk, linestyle="", color="gray",
                   markersize=8, label=m, markeredgecolor="black", markeredgewidth=0.4)
        for m, mk in method_markers.items()
    ]
    fig.legend(
        handles=target_handles + method_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.01),
        ncol=len(target_handles) + len(method_handles),
        frameon=False, fontsize=9,
    )
    fig.suptitle("Energy / Boltz2 / BindCraft-AF2 comparison — best binder per campaign",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")

    # Per-target correlation summary table
    print("\n=== Correlations: per target ===")
    print(f"{'Target':<22} {'n':>3} {'r(E,B_ips)':>12} {'r(E,A_ips)':>12} {'r(B,A) ips':>12} {'r(E,B_ipt)':>12} {'r(E,A_ipt)':>12} {'r(B,A) ipt':>12}")
    for t in targets:
        sub = [r for r in rows if r["target"] == t and r.get("af2_ipsae") is not None]
        n = len(sub)
        if n < 2: continue
        es = [r["energy"] for r in sub]
        b_i = [r["boltz_ipsae"] for r in sub]
        a_i = [r["af2_ipsae"] for r in sub]
        b_t = [r["boltz_iptm"] for r in sub]
        a_t = [r["af2_iptm"] for r in sub]
        print(f"{t:<22} {n:>3} {pearson(es,b_i):>+12.3f} {pearson(es,a_i):>+12.3f} {pearson(b_i,a_i):>+12.3f} "
              f"{pearson(es,b_t):>+12.3f} {pearson(es,a_t):>+12.3f} {pearson(b_t,a_t):>+12.3f}")

    sub = [r for r in rows if r.get("af2_ipsae") is not None]
    n = len(sub)
    es = [r["energy"] for r in sub]
    b_i = [r["boltz_ipsae"] for r in sub]
    a_i = [r["af2_ipsae"] for r in sub]
    b_t = [r["boltz_iptm"] for r in sub]
    a_t = [r["af2_iptm"] for r in sub]
    print(f"\n{'Overall':<22} {n:>3} {pearson(es,b_i):>+12.3f} {pearson(es,a_i):>+12.3f} {pearson(b_i,a_i):>+12.3f} "
          f"{pearson(es,b_t):>+12.3f} {pearson(es,a_t):>+12.3f} {pearson(b_t,a_t):>+12.3f}")


if __name__ == "__main__":
    main()
