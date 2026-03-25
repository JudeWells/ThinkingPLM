"""Compare proposal_bandit_eb5 (discount=0.95) vs proposal_bandit_eb5_d1 (discount=1.0)."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BENCH = Path("outputs/bench")
OUT = Path("outputs/bench_analysis")
OUT.mkdir(exist_ok=True)
plt.style.use("dark_background")

C_OLD = "#ff6b6b"   # d=0.95
C_NEW = "#00e676"   # d=1.0


def find_pairs():
    pairs = []
    for target in sorted(BENCH.iterdir()):
        if not target.is_dir() or target.name == "test_logs":
            continue
        for scaffold in sorted(target.iterdir()):
            if not scaffold.is_dir():
                continue
            old = scaffold / "proposal_bandit_eb5"
            new = scaffold / "proposal_bandit_eb5_d1"
            if (old / "thompson_arms.json").exists() and (new / "thompson_arms.json").exists():
                pairs.append({"target": target.name, "scaffold": scaffold.name,
                              "old": old, "new": new})
    return pairs


def load_arms(path):
    with open(path / "thompson_arms.json") as f:
        return json.load(f)


def load_min_energy(path):
    cs_path = path / "cycle_stats.json"
    if not cs_path.exists():
        return float("nan")
    with open(cs_path) as f:
        cs = json.load(f)
    return min(v.get("best_sequence", {}).get("energy", 0) for v in cs.values())


def arm_summary(arms):
    alphas = [a["alpha"] for a in arms]
    betas = [a["beta_param"] for a in arms]
    ts = [a["times_selected"] for a in arms]
    rewards = [a["total_reward_credited"] for a in arms]
    ipsae = [a["ipsae_raw"] for a in arms]
    abs_ipsae = [abs(x) for x in ipsae]
    n = len(arms)

    # Posterior spread: max(alpha) - 1 shows how much evidence accumulated
    alpha_excess = [a - 1.0 for a in alphas]
    beta_excess = [b - 1.0 for b in betas]

    # Gini on times_selected
    v = np.sort(np.array(ts, dtype=float))
    gini = float((2 * np.sum(np.arange(1, n+1) * v) - (n+1) * np.sum(v)) / (n * np.sum(v))) if v.sum() > 0 else 0

    # Correlation: |ipsae| vs times_selected
    corr_i, p_i = stats.spearmanr(abs_ipsae, ts) if np.std(ts) > 0 and np.std(abs_ipsae) > 0 else (np.nan, np.nan)
    corr_r, p_r = stats.spearmanr(rewards, ts) if np.std(ts) > 0 and np.std(rewards) > 0 else (np.nan, np.nan)

    return {
        "n_arms": n,
        "alpha_mean": np.mean(alphas),
        "alpha_max": max(alphas),
        "alpha_excess_mean": np.mean(alpha_excess),
        "alpha_excess_max": max(alpha_excess),
        "beta_mean": np.mean(betas),
        "beta_max": max(betas),
        "beta_excess_mean": np.mean(beta_excess),
        "beta_excess_max": max(beta_excess),
        "ts_max": max(ts),
        "ts_mean": np.mean(ts),
        "ts_std": np.std(ts),
        "frac_ts_ge3": sum(1 for t in ts if t >= 3) / n,
        "frac_ts_ge5": sum(1 for t in ts if t >= 5) / n,
        "frac_ts_0": sum(1 for t in ts if t == 0) / n,
        "gini": gini,
        "corr_ipsae_ts": corr_i,
        "corr_reward_ts": corr_r,
        "ts_values": ts,
        "alphas": alphas,
        "betas": betas,
        "abs_ipsae": abs_ipsae,
        "rewards": rewards,
    }


def main():
    pairs = find_pairs()
    print(f"Found {len(pairs)} paired experiments\n")

    results = []
    for p in pairs:
        old_arms = load_arms(p["old"])
        new_arms = load_arms(p["new"])
        r = {
            "target": p["target"], "scaffold": p["scaffold"],
            "old": arm_summary(old_arms), "new": arm_summary(new_arms),
            "old_min_e": load_min_energy(p["old"]),
            "new_min_e": load_min_energy(p["new"]),
        }
        results.append(r)
        print(f"  {r['target']}/{r['scaffold']}: "
              f"max_ts {r['old']['ts_max']}→{r['new']['ts_max']}, "
              f"alpha_max {r['old']['alpha_max']:.2f}→{r['new']['alpha_max']:.2f}, "
              f"gini {r['old']['gini']:.3f}→{r['new']['gini']:.3f}")

    # ---- Plot 1: Paired bar charts for key metrics ----
    labels = [f"{r['target'].split('_')[0]}\n{r['scaffold']}" for r in results]
    x = np.arange(len(results))
    w = 0.35

    metrics_bar = [
        ("ts_max", "Max Times Selected"),
        ("alpha_excess_max", "Max Alpha Excess (α - 1)"),
        ("gini", "Selection Concentration (Gini)"),
        ("corr_reward_ts", "Corr: Reward vs Times Sel"),
        ("frac_ts_ge5", "Fraction Arms Selected >= 5"),
        ("frac_ts_0", "Fraction Arms Never Selected"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Discount=0.95 vs Discount=1.0 (EB=5)", color="white", fontsize=16)
    fig.patch.set_facecolor("black")

    for ax_idx, (key, title) in enumerate(metrics_bar):
        ax = axes[ax_idx // 3][ax_idx % 3]
        ax.set_facecolor("black")
        old_vals = [r["old"][key] for r in results]
        new_vals = [r["new"][key] for r in results]
        ax.bar(x - w/2, old_vals, w, color=C_OLD, label="d=0.95", alpha=0.85)
        ax.bar(x + w/2, new_vals, w, color=C_NEW, label="d=1.0", alpha=0.85)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right", color="white")
        ax.legend(fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
    plt.tight_layout()
    fname = OUT / "discount_fix_bars.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {fname}")

    # ---- Plot 2: Alpha distribution (old vs new) overlaid histograms ----
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle("Alpha Distribution per Arm: d=0.95 vs d=1.0", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("black")
        old_a = r["old"]["alphas"]
        new_a = r["new"]["alphas"]
        max_a = max(max(old_a), max(new_a))
        bins = np.linspace(0.9, max_a + 0.1, 30)
        ax.hist(old_a, bins=bins, color=C_OLD, alpha=0.6, label=f"d=0.95 (max={max(old_a):.2f})")
        ax.hist(new_a, bins=bins, color=C_NEW, alpha=0.6, label=f"d=1.0 (max={max(new_a):.2f})")
        ax.set_title(f"{r['target'].split('_')[0]}/{r['scaffold']}", color="white", fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_xlabel("α", fontsize=8, color="white")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    fname = OUT / "discount_fix_alpha_dist.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 3: Times selected distribution ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle("Times Selected Distribution: d=0.95 vs d=1.0", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("black")
        old_ts = r["old"]["ts_values"]
        new_ts = r["new"]["ts_values"]
        max_ts = max(max(old_ts), max(new_ts))
        bins = np.arange(0, max_ts + 2) - 0.5
        ax.hist(old_ts, bins=bins, color=C_OLD, alpha=0.6, label=f"d=0.95 (max={max(old_ts)})")
        ax.hist(new_ts, bins=bins, color=C_NEW, alpha=0.6, label=f"d=1.0 (max={max(new_ts)})")
        ax.set_title(f"{r['target'].split('_')[0]}/{r['scaffold']}", color="white", fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_xlabel("Times Selected", fontsize=8, color="white")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    fname = OUT / "discount_fix_ts_dist.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 4: Min energy comparison ----
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    old_e = [r["old_min_e"] for r in results]
    new_e = [r["new_min_e"] for r in results]
    ax.bar(x - w/2, old_e, w, color=C_OLD, label="d=0.95", alpha=0.85)
    ax.bar(x + w/2, new_e, w, color=C_NEW, label="d=1.0", alpha=0.85)
    ax.set_ylabel("Min Energy", color="white")
    ax.set_title("Min Energy: d=0.95 vs d=1.0 (lower = better)", color="white", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color="white")
    ax.legend()
    ax.tick_params(colors="white")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    fname = OUT / "discount_fix_energy.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 5: |ipSAE| vs times_selected scatter ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle("|ipSAE| vs Times Selected: d=0.95 vs d=1.0", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("black")
        ax.scatter(r["old"]["abs_ipsae"], r["old"]["ts_values"], c=C_OLD, alpha=0.4, s=12, label="d=0.95")
        ax.scatter(r["new"]["abs_ipsae"], r["new"]["ts_values"], c=C_NEW, alpha=0.4, s=12, label="d=1.0")
        ax.set_title(f"{r['target'].split('_')[0]}/{r['scaffold']}", color="white", fontsize=9)
        ax.set_xlabel("|ipSAE|", fontsize=8, color="white")
        ax.set_ylabel("Times Sel", fontsize=8, color="white")
        ax.legend(fontsize=6)
        ax.tick_params(colors="white", labelsize=7)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    plt.tight_layout()
    fname = OUT / "discount_fix_scatter.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Markdown report ----
    lines = []
    lines.append("# Discount Fix Validation: d=0.95 vs d=1.0 (EB=5)\n")
    lines.append(f"Comparing {len(pairs)} paired experiments with thompson_exploit_bias=5.\n")
    lines.append("**Hypothesis**: Setting discount=1.0 allows posteriors to accumulate evidence, "
                 "leading to more exploitation of good arms (higher alpha/beta, higher max times_selected, "
                 "more concentrated selection).\n")

    # Summary table
    lines.append("## Per-Experiment Summary\n")
    lines.append("| Target | Scaffold | Metric | d=0.95 | d=1.0 | Change |")
    lines.append("|--------|----------|--------|--------|-------|--------|")
    for r in results:
        t, s = r["target"], r["scaffold"]
        for key, label, fmt in [
            ("ts_max", "Max times_sel", "d"),
            ("alpha_excess_max", "Max α-1", ".2f"),
            ("gini", "Gini", ".3f"),
            ("frac_ts_ge5", "Frac sel>=5", ".1%"),
        ]:
            o = r["old"][key]
            n_ = r["new"][key]
            if fmt == "d":
                change = f"{n_ - o:+d}"
                lines.append(f"| {t} | {s} | {label} | {o} | {n_} | {change} |")
            elif fmt == ".1%":
                change = f"{n_ - o:+.1%}"
                lines.append(f"| {t} | {s} | {label} | {o:.1%} | {n_:.1%} | {change} |")
            else:
                change = f"{n_ - o:+{fmt}}"
                lines.append(f"| {t} | {s} | {label} | {o:{fmt}} | {n_:{fmt}} | {change} |")

    # Aggregate
    lines.append("\n## Aggregate Comparison\n")
    agg_metrics = [
        ("ts_max", "Max times_selected", "d"),
        ("ts_mean", "Mean times_selected", ".2f"),
        ("alpha_excess_max", "Max alpha excess (α-1)", ".2f"),
        ("alpha_excess_mean", "Mean alpha excess", ".3f"),
        ("beta_excess_max", "Max beta excess (β-1)", ".2f"),
        ("gini", "Gini coefficient", ".3f"),
        ("frac_ts_ge3", "Frac arms sel >= 3", ".1%"),
        ("frac_ts_ge5", "Frac arms sel >= 5", ".1%"),
        ("frac_ts_0", "Frac arms never selected", ".1%"),
        ("corr_ipsae_ts", "Corr |ipSAE| vs sel", ".3f"),
        ("corr_reward_ts", "Corr reward vs sel", ".3f"),
    ]

    lines.append("| Metric | d=0.95 (mean) | d=1.0 (mean) | Change | p-value |")
    lines.append("|--------|---------------|--------------|--------|---------|")

    for key, label, fmt in agg_metrics:
        old_vals = [r["old"][key] for r in results]
        new_vals = [r["new"][key] for r in results]
        # Filter NaN for correlations
        valid = [(o, n_) for o, n_ in zip(old_vals, new_vals) if not (np.isnan(o) or np.isnan(n_))]
        if not valid:
            continue
        ov, nv = zip(*valid)
        om, nm = np.mean(ov), np.mean(nv)
        diff = nm - om
        try:
            _, p = stats.wilcoxon([b - a for a, b in valid])
        except ValueError:
            p = np.nan
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        if fmt == "d":
            lines.append(f"| {label} | {om:.1f} | {nm:.1f} | {diff:+.1f} | {p_str} |")
        elif fmt == ".1%":
            lines.append(f"| {label} | {om:.1%} | {nm:.1%} | {diff:+.1%} | {p_str} |")
        else:
            lines.append(f"| {label} | {om:{fmt}} | {nm:{fmt}} | {diff:+{fmt}} | {p_str} |")

    # Energy comparison
    lines.append("\n## Energy Outcomes\n")
    old_e_vals = [r["old_min_e"] for r in results]
    new_e_vals = [r["new_min_e"] for r in results]
    valid_e = [(o, n_) for o, n_ in zip(old_e_vals, new_e_vals) if not (np.isnan(o) or np.isnan(n_))]
    if valid_e:
        ov_e, nv_e = zip(*valid_e)
        _, p_e = stats.wilcoxon([b - a for a, b in valid_e])
        d1_wins = sum(1 for o, n_ in valid_e if n_ < o)
        lines.append(f"| Metric | d=0.95 | d=1.0 | p-value |")
        lines.append(f"|--------|--------|-------|---------|")
        lines.append(f"| Mean min energy | {np.mean(ov_e):.4f} | {np.mean(nv_e):.4f} | {p_e:.4f} |")
        lines.append(f"| Median min energy | {np.median(ov_e):.4f} | {np.median(nv_e):.4f} | |")
        lines.append(f"| d=1.0 wins | | {d1_wins}/{len(valid_e)} | |")

    # Per-experiment energy table
    lines.append("\n| Target | Scaffold | d=0.95 energy | d=1.0 energy | Winner |")
    lines.append("|--------|----------|---------------|--------------|--------|")
    for r in results:
        winner = "d=1.0" if r["new_min_e"] < r["old_min_e"] else "d=0.95"
        lines.append(f"| {r['target']} | {r['scaffold']} | {r['old_min_e']:.4f} | {r['new_min_e']:.4f} | {winner} |")

    lines.append("\n## Plots\n")
    lines.append("### Key Metrics\n![](discount_fix_bars.png)\n")
    lines.append("### Alpha Distribution\n![](discount_fix_alpha_dist.png)\n")
    lines.append("### Times Selected Distribution\n![](discount_fix_ts_dist.png)\n")
    lines.append("### |ipSAE| vs Times Selected\n![](discount_fix_scatter.png)\n")
    lines.append("### Min Energy\n![](discount_fix_energy.png)\n")

    # Key findings
    lines.append("## Key Findings\n")
    ts_old = np.mean([r["old"]["ts_max"] for r in results])
    ts_new = np.mean([r["new"]["ts_max"] for r in results])
    ae_old = np.mean([r["old"]["alpha_excess_max"] for r in results])
    ae_new = np.mean([r["new"]["alpha_excess_max"] for r in results])
    g_old = np.mean([r["old"]["gini"] for r in results])
    g_new = np.mean([r["new"]["gini"] for r in results])

    lines.append(f"1. **Max times_selected**: d=0.95 mean={ts_old:.1f} → d=1.0 mean={ts_new:.1f} "
                 f"({'confirmed' if ts_new > ts_old else 'NOT confirmed'}: more exploitation with d=1.0).\n")
    lines.append(f"2. **Max alpha excess**: {ae_old:.2f} → {ae_new:.2f} "
                 f"({'confirmed' if ae_new > ae_old else 'NOT confirmed'}: posteriors accumulate more evidence).\n")
    lines.append(f"3. **Gini concentration**: {g_old:.3f} → {g_new:.3f} "
                 f"({'confirmed' if g_new > g_old else 'NOT confirmed'}: selection more concentrated).\n")
    if valid_e:
        lines.append(f"4. **Energy outcomes**: d=1.0 wins {d1_wins}/{len(valid_e)} experiments "
                     f"(p={p_e:.4f}).\n")

    md_path = OUT / "discount_fix_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
