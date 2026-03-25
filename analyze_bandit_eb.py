"""Compare proposal bandit with exploit_bias=2 vs exploit_bias=5."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BENCH_DIR = Path("outputs/bench")
OUT_DIR = Path("outputs/bench_analysis")
OUT_DIR.mkdir(exist_ok=True)

EB2_DIR = "proposal_bandit"
EB5_DIR = "proposal_bandit_eb5"

plt.style.use("dark_background")

COLOR_EB2 = "#00bfff"
COLOR_EB5 = "#e040fb"


def discover_paired_experiments():
    """Find all target/scaffold combos that have both EB=2 and EB=5."""
    pairs = []
    for target_dir in sorted(BENCH_DIR.iterdir()):
        if not target_dir.is_dir() or target_dir.name == "test_logs":
            continue
        for scaffold_dir in sorted(target_dir.iterdir()):
            if not scaffold_dir.is_dir():
                continue
            eb2 = scaffold_dir / EB2_DIR / "thompson_arms.json"
            eb5 = scaffold_dir / EB5_DIR / "thompson_arms.json"
            if eb2.exists() and eb5.exists():
                pairs.append({
                    "target": target_dir.name,
                    "scaffold": scaffold_dir.name,
                    "eb2_path": eb2,
                    "eb5_path": eb5,
                    "eb2_stats": scaffold_dir / EB2_DIR / "cycle_stats.json",
                    "eb5_stats": scaffold_dir / EB5_DIR / "cycle_stats.json",
                })
    return pairs


def load_arms(path):
    with open(path) as f:
        return json.load(f)


def load_cycle_stats(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def arm_stats(arms):
    """Compute summary stats for a list of arms."""
    times_selected = [a["times_selected"] for a in arms]
    ipsae = [a["ipsae_raw"] for a in arms]
    reward = [a["total_reward_credited"] for a in arms]
    n_arms = len(arms)
    n_selected_ge3 = sum(1 for t in times_selected if t >= 3)
    frac_selected_ge3 = n_selected_ge3 / n_arms if n_arms > 0 else 0

    # Correlation between |ipsae_raw| and times_selected
    abs_ipsae = [abs(x) for x in ipsae]
    if len(abs_ipsae) > 2 and np.std(abs_ipsae) > 0 and np.std(times_selected) > 0:
        corr_ipsae_ts, p_ipsae_ts = stats.spearmanr(abs_ipsae, times_selected)
    else:
        corr_ipsae_ts, p_ipsae_ts = float("nan"), float("nan")

    # Correlation between total_reward_credited and times_selected
    if len(reward) > 2 and np.std(reward) > 0 and np.std(times_selected) > 0:
        corr_reward_ts, p_reward_ts = stats.spearmanr(reward, times_selected)
    else:
        corr_reward_ts, p_reward_ts = float("nan"), float("nan")

    return {
        "n_arms": n_arms,
        "times_selected": times_selected,
        "ipsae_raw": ipsae,
        "total_reward": reward,
        "abs_ipsae": abs_ipsae,
        "mean_times_selected": np.mean(times_selected),
        "max_times_selected": max(times_selected) if times_selected else 0,
        "n_selected_ge3": n_selected_ge3,
        "frac_selected_ge3": frac_selected_ge3,
        "corr_ipsae_ts": corr_ipsae_ts,
        "p_ipsae_ts": p_ipsae_ts,
        "corr_reward_ts": corr_reward_ts,
        "p_reward_ts": p_reward_ts,
        "min_ipsae": min(ipsae) if ipsae else 0,
    }


def get_min_energy(cycle_stats):
    if cycle_stats is None:
        return float("nan")
    best = 0.0
    for val in cycle_stats.values():
        e = val.get("best_sequence", {}).get("energy", 0)
        if e < best:
            best = e
    return best


def main():
    pairs = discover_paired_experiments()
    print(f"Found {len(pairs)} paired experiments")

    results = []
    for p in pairs:
        eb2_arms = load_arms(p["eb2_path"])
        eb5_arms = load_arms(p["eb5_path"])
        eb2_cs = load_cycle_stats(p["eb2_stats"])
        eb5_cs = load_cycle_stats(p["eb5_stats"])

        r = {
            "target": p["target"],
            "scaffold": p["scaffold"],
            "eb2": arm_stats(eb2_arms),
            "eb5": arm_stats(eb5_arms),
            "eb2_min_energy": get_min_energy(eb2_cs),
            "eb5_min_energy": get_min_energy(eb5_cs),
        }
        results.append(r)
        print(f"  {r['target']}/{r['scaffold']}: EB2 {r['eb2']['n_arms']} arms, EB5 {r['eb5']['n_arms']} arms")

    # ---- Plot 1: Fraction of arms with times_selected >= 3 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    labels = [f"{r['target'].split('_')[0]}\n{r['scaffold']}" for r in results]
    x = np.arange(len(results))
    w = 0.35

    eb2_frac = [r["eb2"]["frac_selected_ge3"] for r in results]
    eb5_frac = [r["eb5"]["frac_selected_ge3"] for r in results]
    axes[0].bar(x - w/2, eb2_frac, w, color=COLOR_EB2, label="EB=2", alpha=0.85)
    axes[0].bar(x + w/2, eb5_frac, w, color=COLOR_EB5, label="EB=5", alpha=0.85)
    axes[0].set_ylabel("Fraction", color="white")
    axes[0].set_title("Fraction of Arms Selected >= 3 Times", color="white", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color="white")
    axes[0].legend()
    axes[0].tick_params(colors="white")

    # ---- Plot 2: Spearman correlation |ipsae| vs times_selected ----
    eb2_corr = [r["eb2"]["corr_ipsae_ts"] for r in results]
    eb5_corr = [r["eb5"]["corr_ipsae_ts"] for r in results]
    axes[1].bar(x - w/2, eb2_corr, w, color=COLOR_EB2, label="EB=2", alpha=0.85)
    axes[1].bar(x + w/2, eb5_corr, w, color=COLOR_EB5, label="EB=5", alpha=0.85)
    axes[1].set_ylabel("Spearman r", color="white")
    axes[1].set_title("Corr: |ipSAE| vs Times Selected", color="white", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color="white")
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].legend()
    axes[1].tick_params(colors="white")

    # ---- Plot 3: Spearman correlation total_reward vs times_selected ----
    eb2_corr_r = [r["eb2"]["corr_reward_ts"] for r in results]
    eb5_corr_r = [r["eb5"]["corr_reward_ts"] for r in results]
    axes[2].bar(x - w/2, eb2_corr_r, w, color=COLOR_EB2, label="EB=2", alpha=0.85)
    axes[2].bar(x + w/2, eb5_corr_r, w, color=COLOR_EB5, label="EB=5", alpha=0.85)
    axes[2].set_ylabel("Spearman r", color="white")
    axes[2].set_title("Corr: Total Reward vs Times Selected", color="white", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color="white")
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].legend()
    axes[2].tick_params(colors="white")

    plt.tight_layout()
    fname = OUT_DIR / "bandit_eb_comparison_bars.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 4: Scatter plots of |ipsae| vs times_selected for EB2 and EB5 ----
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("|ipSAE| vs Times Selected per Arm", fontsize=16, color="white", y=1.01)
    fig.patch.set_facecolor("black")

    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("black")

        eb2 = r["eb2"]
        eb5 = r["eb5"]
        ax.scatter(eb2["abs_ipsae"], eb2["times_selected"], c=COLOR_EB2, alpha=0.5, s=20, label="EB=2")
        ax.scatter(eb5["abs_ipsae"], eb5["times_selected"], c=COLOR_EB5, alpha=0.5, s=20, label="EB=5")
        ax.set_xlabel("|ipSAE|", fontsize=8, color="white")
        ax.set_ylabel("Times Selected", fontsize=8, color="white")
        ax.set_title(f"{r['target'].split('_')[0]}/{r['scaffold']}", fontsize=9, color="white")
        ax.legend(fontsize=7)
        ax.tick_params(colors="white", labelsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    fname = OUT_DIR / "bandit_eb_scatter_ipsae_vs_selected.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 5: Distribution of times_selected (histogram) EB2 vs EB5 ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Distribution of Times Selected per Arm", fontsize=16, color="white", y=1.01)
    fig.patch.set_facecolor("black")

    for idx, r in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("black")
        max_ts = max(max(r["eb2"]["times_selected"], default=0), max(r["eb5"]["times_selected"], default=0))
        bins = np.arange(0, max_ts + 2) - 0.5
        ax.hist(r["eb2"]["times_selected"], bins=bins, color=COLOR_EB2, alpha=0.6, label="EB=2")
        ax.hist(r["eb5"]["times_selected"], bins=bins, color=COLOR_EB5, alpha=0.6, label="EB=5")
        ax.set_xlabel("Times Selected", fontsize=8, color="white")
        ax.set_ylabel("Count", fontsize=8, color="white")
        ax.set_title(f"{r['target'].split('_')[0]}/{r['scaffold']}", fontsize=9, color="white")
        ax.legend(fontsize=7)
        ax.tick_params(colors="white", labelsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    fname = OUT_DIR / "bandit_eb_hist_times_selected.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 6: Min energy comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    eb2_e = [r["eb2_min_energy"] for r in results]
    eb5_e = [r["eb5_min_energy"] for r in results]
    ax.bar(x - w/2, eb2_e, w, color=COLOR_EB2, label="EB=2", alpha=0.85)
    ax.bar(x + w/2, eb5_e, w, color=COLOR_EB5, label="EB=5", alpha=0.85)
    ax.set_ylabel("Min Energy", color="white")
    ax.set_title("Min Energy Achieved: EB=2 vs EB=5", color="white", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right", color="white")
    ax.legend()
    ax.tick_params(colors="white")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    fname = OUT_DIR / "bandit_eb_min_energy.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Generate markdown ----
    lines = []
    lines.append("# Proposal Bandit: Exploit Bias 2 vs 5\n")
    lines.append("Comparing Thompson sampling proposal bandit with exploitation bias=2 (default) vs bias=5 (higher exploitation).\n")
    lines.append("## Hypotheses\n")
    lines.append("1. Higher exploit bias concentrates selection on fewer, better-performing arms.")
    lines.append("2. EB=5 should show **more arms with times_selected >= 3** (i.e. stronger re-selection of good arms).")
    lines.append("3. EB=5 should show **higher correlation** between |ipSAE|/total_reward and times_selected (the bandit better tracks quality).\n")

    # Summary table
    lines.append("## Per-Experiment Summary\n")
    lines.append("| Target | Scaffold | EB | Arms | Mean Sel | Max Sel | Frac>=3 | r(|ipSAE|,sel) | p | r(reward,sel) | p | Min Energy |")
    lines.append("|--------|----------|----|------|----------|---------|---------|----------------|---|---------------|---|------------|")

    for r in results:
        for eb_label, eb_key, min_e in [("2", "eb2", r["eb2_min_energy"]), ("5", "eb5", r["eb5_min_energy"])]:
            s = r[eb_key]
            p_ipsae = f"{s['p_ipsae_ts']:.3f}" if not np.isnan(s["p_ipsae_ts"]) else "N/A"
            p_reward = f"{s['p_reward_ts']:.3f}" if not np.isnan(s["p_reward_ts"]) else "N/A"
            corr_i = f"{s['corr_ipsae_ts']:.3f}" if not np.isnan(s["corr_ipsae_ts"]) else "N/A"
            corr_r = f"{s['corr_reward_ts']:.3f}" if not np.isnan(s["corr_reward_ts"]) else "N/A"
            lines.append(
                f"| {r['target']} | {r['scaffold']} | {eb_label} | {s['n_arms']} | "
                f"{s['mean_times_selected']:.1f} | {s['max_times_selected']} | "
                f"{s['frac_selected_ge3']:.1%} | {corr_i} | {p_ipsae} | "
                f"{corr_r} | {p_reward} | {min_e:.4f} |"
            )

    # Aggregate comparison
    lines.append("\n## Aggregate Comparison\n")

    eb2_fracs = [r["eb2"]["frac_selected_ge3"] for r in results]
    eb5_fracs = [r["eb5"]["frac_selected_ge3"] for r in results]
    eb2_corrs_i = [r["eb2"]["corr_ipsae_ts"] for r in results if not np.isnan(r["eb2"]["corr_ipsae_ts"])]
    eb5_corrs_i = [r["eb5"]["corr_ipsae_ts"] for r in results if not np.isnan(r["eb5"]["corr_ipsae_ts"])]
    eb2_corrs_r = [r["eb2"]["corr_reward_ts"] for r in results if not np.isnan(r["eb2"]["corr_reward_ts"])]
    eb5_corrs_r = [r["eb5"]["corr_reward_ts"] for r in results if not np.isnan(r["eb5"]["corr_reward_ts"])]
    eb2_energies = [r["eb2_min_energy"] for r in results]
    eb5_energies = [r["eb5_min_energy"] for r in results]

    lines.append("| Metric | EB=2 (mean) | EB=5 (mean) | Difference | Better? |")
    lines.append("|--------|-------------|-------------|------------|---------|")

    diff_frac = np.mean(eb5_fracs) - np.mean(eb2_fracs)
    lines.append(f"| Frac arms selected >= 3 | {np.mean(eb2_fracs):.1%} | {np.mean(eb5_fracs):.1%} | "
                 f"{diff_frac:+.1%} | {'EB=5' if diff_frac > 0 else 'EB=2'} |")

    diff_corr_i = np.mean(eb5_corrs_i) - np.mean(eb2_corrs_i)
    lines.append(f"| Corr |ipSAE| vs times_sel | {np.mean(eb2_corrs_i):.3f} | {np.mean(eb5_corrs_i):.3f} | "
                 f"{diff_corr_i:+.3f} | {'EB=5' if diff_corr_i > 0 else 'EB=2'} |")

    diff_corr_r = np.mean(eb5_corrs_r) - np.mean(eb2_corrs_r)
    lines.append(f"| Corr reward vs times_sel | {np.mean(eb2_corrs_r):.3f} | {np.mean(eb5_corrs_r):.3f} | "
                 f"{diff_corr_r:+.3f} | {'EB=5' if diff_corr_r > 0 else 'EB=2'} |")

    diff_e = np.mean(eb5_energies) - np.mean(eb2_energies)
    lines.append(f"| Min energy (mean) | {np.mean(eb2_energies):.4f} | {np.mean(eb5_energies):.4f} | "
                 f"{diff_e:+.4f} | {'EB=5' if diff_e < 0 else 'EB=2'} |")

    # Paired test on min energy
    t_stat, p_val = stats.wilcoxon([r["eb5_min_energy"] - r["eb2_min_energy"] for r in results])
    lines.append(f"\nWilcoxon signed-rank test on min energy difference (EB5 - EB2): "
                 f"W={t_stat:.1f}, p={p_val:.4f}\n")

    # Win count
    eb5_wins_energy = sum(1 for r in results if r["eb5_min_energy"] < r["eb2_min_energy"])
    lines.append(f"EB=5 achieves lower min energy in {eb5_wins_energy}/{len(results)} experiments.\n")

    # Paired test on correlations
    if len(eb2_corrs_r) > 2 and len(eb5_corrs_r) > 2:
        # Match lengths
        paired_corr_diff = []
        for r in results:
            c2 = r["eb2"]["corr_reward_ts"]
            c5 = r["eb5"]["corr_reward_ts"]
            if not np.isnan(c2) and not np.isnan(c5):
                paired_corr_diff.append(c5 - c2)
        if len(paired_corr_diff) > 2:
            t_c, p_c = stats.wilcoxon(paired_corr_diff)
            lines.append(f"Wilcoxon test on reward-selection correlation difference: W={t_c:.1f}, p={p_c:.4f}\n")

    # Plots
    lines.append("## Plots\n")
    lines.append("### Selection Frequency, |ipSAE| Correlation, Reward Correlation\n")
    lines.append("![Bar comparison](bandit_eb_comparison_bars.png)\n")
    lines.append("### |ipSAE| vs Times Selected (scatter)\n")
    lines.append("![Scatter](bandit_eb_scatter_ipsae_vs_selected.png)\n")
    lines.append("### Distribution of Times Selected\n")
    lines.append("![Histogram](bandit_eb_hist_times_selected.png)\n")
    lines.append("### Min Energy Achieved\n")
    lines.append("![Min Energy](bandit_eb_min_energy.png)\n")

    # Key findings
    lines.append("## Key Findings\n")
    finding = 1

    lines.append(f"{finding}. **Frac arms selected >= 3**: EB=2 mean {np.mean(eb2_fracs):.1%} vs EB=5 mean {np.mean(eb5_fracs):.1%} "
                 f"({diff_frac:+.1%}).\n")
    finding += 1

    lines.append(f"{finding}. **Reward-selection correlation**: EB=2 mean r={np.mean(eb2_corrs_r):.3f} vs "
                 f"EB=5 mean r={np.mean(eb5_corrs_r):.3f} ({diff_corr_r:+.3f}).\n")
    finding += 1

    lines.append(f"{finding}. **Min energy**: EB=2 mean {np.mean(eb2_energies):.4f} vs "
                 f"EB=5 mean {np.mean(eb5_energies):.4f}. EB=5 wins {eb5_wins_energy}/{len(results)} experiments.\n")
    finding += 1

    md_path = OUT_DIR / "bandit_eb_analysis.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
