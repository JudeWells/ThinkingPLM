#!/usr/bin/env python
"""
Analyze results from the 2GDZ_15PGDH campaign.

Strategies:
- random_greedy: random mutations, greedy selection
- random_thompson: random mutations, Thompson selection
- thompson_eb8_bandit_rel: Thompson + proposal bandit (EB=8, relative reward)
- thompson_eb16_bandit_rel: Thompson + proposal bandit (EB=16, relative reward)

Focus on:
1. Which scaffolds work best for 2GDZ
2. Which strategies achieve ipSAE < -0.6
3. Convergence analysis
4. Best sequences found
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IPSAE_THRESHOLD = -0.6
BASE_DIR = Path("outputs/2gdz_campaign")

# Dark mode styling
plt.style.use("dark_background")
COLORS = {
    "random_greedy": "#ff6b6b",
    "random_thompson": "#ffab40",
    "thompson_eb8_bandit_rel": "#00bfff",
    "thompson_eb16_bandit_rel": "#00e676",
}
STRATEGY_LABELS = {
    "random_greedy": "Random Greedy",
    "random_thompson": "Random Thompson",
    "thompson_eb8_bandit_rel": "Thompson EB8 + Bandit",
    "thompson_eb16_bandit_rel": "Thompson EB16 + Bandit",
}

STRATEGIES = [
    "random_greedy",
    "random_thompson",
    "thompson_eb8_bandit_rel",
    "thompson_eb16_bandit_rel",
]


def load_all_results() -> pd.DataFrame:
    """Load all results into a DataFrame."""
    rows = []

    for stats_file in BASE_DIR.rglob("cycle_stats.json"):
        parts = stats_file.parts
        try:
            idx = parts.index("2gdz_campaign")
            scaffold = parts[idx + 1]
            strategy = parts[idx + 2]
        except (ValueError, IndexError):
            continue

        with open(stats_file) as f:
            stats = json.load(f)

        for cycle_key, cycle_data in stats.items():
            cycle = int(cycle_key)
            best_energy = cycle_data.get("all_min_energy", float("inf"))

            best_seq_data = cycle_data.get("best_sequence", {})
            if isinstance(best_seq_data, dict):
                best_seq = best_seq_data.get("sequence", "")
                energy_terms = best_seq_data.get("energy_terms", {})
                if isinstance(energy_terms, dict) and "ipSAE" in energy_terms:
                    ipsae = energy_terms.get("ipSAE", best_energy)
                elif isinstance(energy_terms, dict) and "energy_terms" in energy_terms:
                    ipsae = energy_terms.get("energy_terms", {}).get("ipSAE", best_energy)
                else:
                    ipsae = best_energy
            else:
                best_seq = ""
                ipsae = best_energy

            rows.append({
                "scaffold": scaffold,
                "strategy": strategy,
                "cycle": cycle,
                "best_ipsae": ipsae,
                "best_sequence": best_seq,
                "run_id": f"{scaffold}/{strategy}",
            })

    return pd.DataFrame(rows)


def analyze_threshold_achievement(df: pd.DataFrame):
    """Analyze which runs achieved the threshold."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD ACHIEVEMENT (ipSAE < {IPSAE_THRESHOLD})")
    print(f"{'='*70}")

    # Best per run
    best_per_run = df.groupby(["scaffold", "strategy"]).agg({
        "best_ipsae": "min",
        "cycle": "max",
    }).reset_index()
    best_per_run["achieved"] = best_per_run["best_ipsae"] < IPSAE_THRESHOLD

    # By scaffold
    print(f"\n--- By Scaffold ---")
    scaffold_summary = best_per_run.groupby("scaffold").agg({
        "achieved": ["sum", "count", "mean"],
        "best_ipsae": ["min", "mean"],
    }).round(4)
    scaffold_summary.columns = ["n_achieved", "n_total", "pct_achieved", "best", "mean_best"]
    scaffold_summary = scaffold_summary.sort_values("best")
    print(scaffold_summary.to_string())

    # By strategy
    print(f"\n--- By Strategy ---")
    strategy_summary = best_per_run.groupby("strategy").agg({
        "achieved": ["sum", "count", "mean"],
        "best_ipsae": ["min", "mean"],
    }).round(4)
    strategy_summary.columns = ["n_achieved", "n_total", "pct_achieved", "best", "mean_best"]
    strategy_summary = strategy_summary.sort_values("mean_best")
    print(strategy_summary.to_string())

    # Cross-tabulation
    print(f"\n--- Best ipSAE by Scaffold × Strategy ---")
    pivot = best_per_run.pivot_table(
        index="scaffold",
        columns="strategy",
        values="best_ipsae",
        aggfunc="min"
    ).round(4)
    # Reorder columns
    col_order = [c for c in STRATEGIES if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string())

    return best_per_run


def analyze_convergence(df: pd.DataFrame):
    """Analyze convergence over cycles."""
    print(f"\n{'='*70}")
    print(f"CONVERGENCE ANALYSIS")
    print(f"{'='*70}")

    # Compute cumulative best per run
    trajectory_data = []
    for run_id in df["run_id"].unique():
        run_df = df[df["run_id"] == run_id].sort_values("cycle")
        cumulative_best = float("inf")
        for _, row in run_df.iterrows():
            cumulative_best = min(cumulative_best, row["best_ipsae"])
            trajectory_data.append({
                "run_id": run_id,
                "scaffold": row["scaffold"],
                "strategy": row["strategy"],
                "cycle": row["cycle"],
                "cumulative_best": cumulative_best,
            })

    traj_df = pd.DataFrame(trajectory_data)

    # Average by strategy at key cycles
    key_cycles = [0, 50, 100, 200, 300, 400, 500, 600]
    print(f"\n--- Mean Cumulative Best by Strategy at Key Cycles ---")
    cycle_summary = []
    for cycle in key_cycles:
        cycle_data = traj_df[traj_df["cycle"] == cycle]
        if len(cycle_data) > 0:
            by_strategy = cycle_data.groupby("strategy")["cumulative_best"].mean()
            cycle_summary.append(by_strategy)

    if cycle_summary:
        summary_df = pd.DataFrame(cycle_summary, index=key_cycles).T.round(4)
        # Reorder rows
        row_order = [r for r in STRATEGIES if r in summary_df.index]
        summary_df = summary_df.loc[row_order]
        print(summary_df.to_string())

    # Average by scaffold at key cycles
    print(f"\n--- Mean Cumulative Best by Scaffold at Key Cycles ---")
    cycle_summary = []
    for cycle in key_cycles:
        cycle_data = traj_df[traj_df["cycle"] == cycle]
        if len(cycle_data) > 0:
            by_scaffold = cycle_data.groupby("scaffold")["cumulative_best"].mean()
            cycle_summary.append(by_scaffold)

    if cycle_summary:
        summary_df = pd.DataFrame(cycle_summary, index=key_cycles).T.round(4)
        summary_df = summary_df.sort_values(key_cycles[-1] if key_cycles[-1] in summary_df.columns else summary_df.columns[-1])
        print(summary_df.to_string())

    return traj_df


def analyze_random_vs_thompson(df: pd.DataFrame, best_per_run: pd.DataFrame):
    """Compare random mutation strategies vs Thompson+bandit strategies."""
    print(f"\n{'='*70}")
    print(f"RANDOM MUTATION vs THOMPSON+BANDIT COMPARISON")
    print(f"{'='*70}")

    # Categorize strategies
    best_per_run = best_per_run.copy()
    best_per_run["category"] = best_per_run["strategy"].apply(
        lambda x: "random" if x.startswith("random") else "thompson_bandit"
    )
    best_per_run["selection"] = best_per_run["strategy"].apply(
        lambda x: "greedy" if "greedy" in x else "thompson"
    )

    # Compare categories
    print(f"\n--- By Category ---")
    cat_summary = best_per_run.groupby("category").agg({
        "achieved": ["sum", "mean"],
        "best_ipsae": ["min", "mean", "median"],
    }).round(4)
    cat_summary.columns = ["n_achieved", "pct_achieved", "best", "mean", "median"]
    print(cat_summary.to_string())

    # Compare selection methods
    print(f"\n--- By Selection Method ---")
    sel_summary = best_per_run.groupby("selection").agg({
        "achieved": ["sum", "mean"],
        "best_ipsae": ["min", "mean", "median"],
    }).round(4)
    sel_summary.columns = ["n_achieved", "pct_achieved", "best", "mean", "median"]
    print(sel_summary.to_string())


def analyze_exploit_bias(df: pd.DataFrame, best_per_run: pd.DataFrame):
    """Compare EB=8 vs EB=16 for Thompson+bandit strategies."""
    print(f"\n{'='*70}")
    print(f"EXPLOIT BIAS COMPARISON (EB=8 vs EB=16)")
    print(f"{'='*70}")

    # Filter to Thompson+bandit strategies only
    thompson_runs = best_per_run[best_per_run["strategy"].str.contains("thompson_eb")]

    if len(thompson_runs) == 0:
        print("No Thompson+bandit runs found.")
        return

    # Extract EB value
    thompson_runs = thompson_runs.copy()
    thompson_runs["eb"] = thompson_runs["strategy"].apply(
        lambda x: 16 if "eb16" in x else 8
    )

    # Compare EB values
    eb_summary = thompson_runs.groupby("eb").agg({
        "achieved": ["sum", "mean"],
        "best_ipsae": ["min", "mean", "median"],
    }).round(4)
    eb_summary.columns = ["n_achieved", "pct_achieved", "best", "mean", "median"]
    print(eb_summary.to_string())

    # Per-scaffold comparison
    print(f"\n--- Per-Scaffold: EB=8 vs EB=16 ---")
    for scaffold in sorted(thompson_runs["scaffold"].unique()):
        scaffold_data = thompson_runs[thompson_runs["scaffold"] == scaffold]
        eb8_val = scaffold_data[scaffold_data["eb"] == 8]["best_ipsae"].values
        eb16_val = scaffold_data[scaffold_data["eb"] == 16]["best_ipsae"].values

        if len(eb8_val) > 0 and len(eb16_val) > 0:
            winner = "EB8" if eb8_val[0] < eb16_val[0] else "EB16" if eb16_val[0] < eb8_val[0] else "TIE"
            print(f"  {scaffold:20s}: EB8={eb8_val[0]:.4f}, EB16={eb16_val[0]:.4f} → {winner}")


def find_best_sequences(df: pd.DataFrame, top_n: int = 20):
    """Find the best sequences across all runs."""
    print(f"\n{'='*70}")
    print(f"TOP {top_n} BEST SEQUENCES")
    print(f"{'='*70}")

    # Get best per run
    best_per_run = df.loc[df.groupby("run_id")["best_ipsae"].idxmin()]
    best_per_run = best_per_run.sort_values("best_ipsae").head(top_n)

    for i, (_, row) in enumerate(best_per_run.iterrows(), 1):
        achieved = "✓" if row['best_ipsae'] < IPSAE_THRESHOLD else "✗"
        print(f"\n{i}. {achieved} ipSAE = {row['best_ipsae']:.4f}")
        print(f"   Scaffold: {row['scaffold']}, Strategy: {row['strategy']}, Cycle: {row['cycle']}")
        seq = row['best_sequence']
        if len(seq) > 80:
            print(f"   Sequence ({len(seq)} aa): {seq[:40]}...{seq[-40:]}")
        else:
            print(f"   Sequence ({len(seq)} aa): {seq}")


def generate_summary(df: pd.DataFrame, best_per_run: pd.DataFrame):
    """Generate overall summary and recommendations."""
    print(f"\n{'='*70}")
    print(f"SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*70}")

    # Best scaffolds
    scaffold_perf = best_per_run.groupby("scaffold")["best_ipsae"].min().sort_values()
    print(f"\n--- Best Scaffolds for 2GDZ ---")
    for scaffold, ipsae in scaffold_perf.items():
        achieved = "✓" if ipsae < IPSAE_THRESHOLD else "✗"
        print(f"  {achieved} {scaffold}: {ipsae:.4f}")

    # Best strategies
    strategy_perf = best_per_run.groupby("strategy")["best_ipsae"].agg(["min", "mean"]).sort_values("mean")
    print(f"\n--- Strategy Performance ---")
    for strategy, row in strategy_perf.iterrows():
        print(f"  {strategy}: best={row['min']:.4f}, mean={row['mean']:.4f}")

    # Overall best
    best_idx = best_per_run["best_ipsae"].idxmin()
    best_row = best_per_run.loc[best_idx]
    print(f"\n--- Overall Best ---")
    print(f"  ipSAE: {best_row['best_ipsae']:.4f}")
    print(f"  Scaffold: {best_row['scaffold']}")
    print(f"  Strategy: {best_row['strategy']}")

    # Achievement rate
    n_achieved = best_per_run["achieved"].sum()
    n_total = len(best_per_run)
    print(f"\n--- Threshold Achievement ---")
    print(f"  Runs achieving ipSAE < {IPSAE_THRESHOLD}: {n_achieved}/{n_total} ({n_achieved/n_total:.1%})")

    # Recommendations
    print(f"\n--- Recommendations ---")

    # Best scaffold
    best_scaffold = scaffold_perf.index[0]
    print(f"  1. Best scaffold: {best_scaffold} (ipSAE={scaffold_perf[best_scaffold]:.4f})")

    # Best strategy
    best_strategy = strategy_perf.index[0]
    print(f"  2. Best strategy: {best_strategy}")

    # Thompson vs random
    random_mean = best_per_run[best_per_run["strategy"].str.contains("random")]["best_ipsae"].mean()
    thompson_mean = best_per_run[best_per_run["strategy"].str.contains("thompson_eb")]["best_ipsae"].mean()
    if thompson_mean < random_mean:
        print(f"  3. Thompson+bandit outperforms random ({thompson_mean:.4f} vs {random_mean:.4f})")
    else:
        print(f"  3. Random mutation outperforms Thompson+bandit ({random_mean:.4f} vs {thompson_mean:.4f})")


def plot_trajectories_by_scaffold(df: pd.DataFrame, traj_df: pd.DataFrame):
    """Plot convergence trajectories grouped by scaffold."""
    scaffolds = sorted(df["scaffold"].unique())
    n_scaffolds = len(scaffolds)

    # Calculate grid size
    n_cols = 4
    n_rows = (n_scaffolds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_scaffolds > 1 else [axes]

    for i, scaffold in enumerate(scaffolds):
        ax = axes[i]
        scaffold_data = traj_df[traj_df["scaffold"] == scaffold]

        for strategy in STRATEGIES:
            strat_data = scaffold_data[scaffold_data["strategy"] == strategy]
            if len(strat_data) == 0:
                continue
            strat_data = strat_data.sort_values("cycle")
            color = COLORS.get(strategy, "#ffffff")
            label = STRATEGY_LABELS.get(strategy, strategy)
            ax.plot(strat_data["cycle"], strat_data["cumulative_best"],
                   color=color, label=label, linewidth=1.5)

        ax.axhline(y=IPSAE_THRESHOLD, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(scaffold, fontsize=10, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("Best ipSAE", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-1.0, 0.1)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add legend to the last visible subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("2GDZ Campaign: Convergence by Scaffold", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fname = BASE_DIR / "trajectories_by_scaffold.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_heatmap(best_per_run: pd.DataFrame):
    """Plot heatmap of best ipSAE by scaffold × strategy."""
    pivot = best_per_run.pivot_table(
        index="scaffold",
        columns="strategy",
        values="best_ipsae",
        aggfunc="min"
    )

    # Reorder columns to match STRATEGIES
    col_order = [c for c in STRATEGIES if c in pivot.columns]
    pivot = pivot[col_order]

    # Sort rows by best overall
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with custom colormap (green = good, red = bad)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ipsae", ["#00e676", "#ffab40", "#ff6b6b"])

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-0.8, vmax=-0.1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Best ipSAE", fontsize=10)

    # Set ticks and labels
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in col_order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(col_order)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text_color = "black" if val < -0.5 else "white"
                marker = "✓" if val < IPSAE_THRESHOLD else ""
                ax.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                       fontsize=8, color=text_color, fontweight="bold")

    # Add threshold line annotation
    ax.axhline(y=-0.5, color="white", linestyle=":", alpha=0)  # invisible, just for spacing

    plt.title("2GDZ Campaign: Best ipSAE by Scaffold × Strategy\n(✓ = crossed -0.6 threshold)",
              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    fname = BASE_DIR / "heatmap_scaffold_strategy.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_bar_comparison(best_per_run: pd.DataFrame):
    """Plot bar chart comparing strategies across scaffolds."""
    # Best by scaffold
    scaffold_best = best_per_run.groupby("scaffold")["best_ipsae"].min().sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scaffold bar chart
    colors = ["#00e676" if v < IPSAE_THRESHOLD else "#ff6b6b" for v in scaffold_best.values]
    bars1 = ax1.barh(range(len(scaffold_best)), scaffold_best.values, color=colors)
    ax1.set_yticks(range(len(scaffold_best)))
    ax1.set_yticklabels(scaffold_best.index, fontsize=9)
    ax1.axvline(x=IPSAE_THRESHOLD, color="white", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Best ipSAE", fontsize=10)
    ax1.set_title("Best ipSAE by Scaffold", fontsize=11, fontweight="bold")
    ax1.set_xlim(-0.9, 0)

    # Strategy bar chart
    strategy_best = best_per_run.groupby("strategy")["best_ipsae"].min()
    strategy_best = strategy_best.reindex([s for s in STRATEGIES if s in strategy_best.index])

    colors2 = [COLORS.get(s, "#ffffff") for s in strategy_best.index]
    bars2 = ax2.barh(range(len(strategy_best)), strategy_best.values, color=colors2)
    ax2.set_yticks(range(len(strategy_best)))
    ax2.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategy_best.index], fontsize=9)
    ax2.axvline(x=IPSAE_THRESHOLD, color="white", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Best ipSAE", fontsize=10)
    ax2.set_title("Best ipSAE by Strategy", fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.9, 0)

    plt.suptitle("2GDZ Campaign: Performance Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fname = BASE_DIR / "bar_comparison.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_aggregate_trajectories(traj_df: pd.DataFrame):
    """Plot aggregate trajectories averaged across scaffolds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in STRATEGIES:
        strat_data = traj_df[traj_df["strategy"] == strategy]
        if len(strat_data) == 0:
            continue

        # Average across scaffolds at each cycle
        avg_by_cycle = strat_data.groupby("cycle")["cumulative_best"].mean()
        color = COLORS.get(strategy, "#ffffff")
        label = STRATEGY_LABELS.get(strategy, strategy)
        ax.plot(avg_by_cycle.index, avg_by_cycle.values, color=color, label=label, linewidth=2)

    ax.axhline(y=IPSAE_THRESHOLD, color="white", linestyle="--", alpha=0.5, linewidth=1.5, label="Threshold (-0.6)")
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Mean Best ipSAE (across scaffolds)", fontsize=11)
    ax.set_title("2GDZ Campaign: Aggregate Convergence by Strategy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.8, 0)
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    fname = BASE_DIR / "aggregate_trajectories.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def load_proposal_method_data() -> pd.DataFrame:
    """Load proposal method selection data from thompson_eb* runs.

    Uses optimized line-by-line parsing to avoid loading large JSON files fully.
    """
    import re
    rows = []

    for scaffold_dir in BASE_DIR.iterdir():
        if not scaffold_dir.is_dir():
            continue
        scaffold = scaffold_dir.name

        for strategy_dir in scaffold_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name

            # Only process thompson_eb* strategies (which have proposal bandit)
            if not strategy.startswith("thompson_eb"):
                continue

            stats_file = strategy_dir / "cycle_stats.json"
            if not stats_file.exists():
                continue

            # Read file and extract proposal_method with line-by-line parsing
            with open(stats_file) as f:
                content = f.read()

            current_cycle = None
            for line in content.split('\n'):
                # Check for cycle start: "N": {
                cycle_match = re.match(r'\s*"(\d+)":\s*\{', line)
                if cycle_match:
                    current_cycle = int(cycle_match.group(1))

                # Check for proposal_method
                method_match = re.search(r'"proposal_method":\s*"([^"]+)"', line)
                if method_match and current_cycle is not None:
                    proposal_method = method_match.group(1)
                    rows.append({
                        "scaffold": scaffold,
                        "strategy": strategy,
                        "cycle": current_cycle,
                        "proposal_method": proposal_method,
                        "is_profam": 1 if proposal_method == "profam" else 0,
                        "run_id": f"{scaffold}/{strategy}",
                    })

    return pd.DataFrame(rows)


def plot_proposal_method_selection(proposal_df: pd.DataFrame, window_size: int = 50):
    """Plot proportion of ProFam selection over cycles for each experiment."""
    if len(proposal_df) == 0:
        print("No proposal method data available")
        return None

    # Get unique runs
    runs = sorted(proposal_df["run_id"].unique())
    n_runs = len(runs)

    if n_runs == 0:
        print("No thompson_eb* runs found")
        return None

    # Calculate grid size
    n_cols = 4
    n_rows = (n_runs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten() if n_runs > 1 else [axes]

    for i, run_id in enumerate(runs):
        ax = axes[i]
        run_data = proposal_df[proposal_df["run_id"] == run_id].sort_values("cycle")

        if len(run_data) == 0:
            continue

        cycles = run_data["cycle"].values
        is_profam = run_data["is_profam"].values

        # Calculate rolling average
        smoothed = pd.Series(is_profam).rolling(window=window_size, min_periods=1, center=True).mean()

        # Plot raw data as scatter (semi-transparent)
        ax.scatter(cycles, is_profam, alpha=0.1, s=5, color="#00bfff", label="Raw")

        # Plot smoothed line
        strategy = run_data["strategy"].iloc[0]
        color = COLORS.get(strategy, "#ffffff")
        ax.plot(cycles, smoothed.values, color=color, linewidth=2, label=f"Smoothed (w={window_size})")

        # Add 50% reference line
        ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.3, linewidth=1)

        # Formatting
        scaffold = run_data["scaffold"].iloc[0]
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        ax.set_title(f"{scaffold}\n{strategy_label}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("ProFam Proportion", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, max(cycles) + 10)

        # Add summary stats
        total_profam = is_profam.sum()
        total = len(is_profam)
        pct = total_profam / total * 100 if total > 0 else 0
        ax.text(0.98, 0.02, f"ProFam: {pct:.1f}%", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="#00e676")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Proposal Method Selection: ProFam vs Random Mutation\n(Smoothed rolling average)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    fname = BASE_DIR / "proposal_method_selection.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_aggregate_proposal_method(proposal_df: pd.DataFrame, window_size: int = 50):
    """Plot aggregate ProFam selection proportion across all experiments."""
    if len(proposal_df) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in ["thompson_eb8_bandit_rel", "thompson_eb16_bandit_rel"]:
        strat_data = proposal_df[proposal_df["strategy"] == strategy]
        if len(strat_data) == 0:
            continue

        # Average across scaffolds at each cycle
        avg_by_cycle = strat_data.groupby("cycle")["is_profam"].mean().sort_index()

        # Smooth
        smoothed = avg_by_cycle.rolling(window=window_size, min_periods=1, center=True).mean()

        color = COLORS.get(strategy, "#ffffff")
        label = STRATEGY_LABELS.get(strategy, strategy)
        ax.plot(smoothed.index, smoothed.values, color=color, linewidth=2, label=label)

    ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.5, linewidth=1.5, label="50% baseline")
    ax.set_xlabel("Cycle", fontsize=11)
    ax.set_ylabel("Mean ProFam Selection Rate", fontsize=11)
    ax.set_title("Aggregate Proposal Method Selection by Strategy\n(Averaged across scaffolds)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 600)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    fname = BASE_DIR / "proposal_method_aggregate.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def generate_markdown_report(df: pd.DataFrame, best_per_run: pd.DataFrame):
    """Generate a markdown report with all results."""
    lines = []
    lines.append("# 2GDZ_15PGDH Campaign Analysis\n")
    lines.append(f"**Target:** 15-PGDH (PDB: 2GDZ)\n")
    lines.append(f"**Threshold:** ipSAE < {IPSAE_THRESHOLD}\n")
    lines.append(f"**Total Runs:** {len(best_per_run)}\n")

    # Summary stats
    n_achieved = (best_per_run["best_ipsae"] < IPSAE_THRESHOLD).sum()
    lines.append(f"**Runs Achieving Threshold:** {n_achieved}/{len(best_per_run)} ({n_achieved/len(best_per_run):.1%})\n")

    # Best overall
    best_idx = best_per_run["best_ipsae"].idxmin()
    best_row = best_per_run.loc[best_idx]
    lines.append(f"\n## Best Result\n")
    lines.append(f"- **ipSAE:** {best_row['best_ipsae']:.4f}\n")
    lines.append(f"- **Scaffold:** {best_row['scaffold']}\n")
    lines.append(f"- **Strategy:** {best_row['strategy']}\n")

    # Scaffold ranking
    lines.append(f"\n## Scaffold Ranking\n")
    lines.append("| Rank | Scaffold | Best ipSAE | Achieved |\n")
    lines.append("|------|----------|------------|----------|\n")
    scaffold_best = best_per_run.groupby("scaffold")["best_ipsae"].min().sort_values()
    for rank, (scaffold, ipsae) in enumerate(scaffold_best.items(), 1):
        achieved = "✓" if ipsae < IPSAE_THRESHOLD else "✗"
        lines.append(f"| {rank} | {scaffold} | {ipsae:.4f} | {achieved} |\n")

    # Strategy ranking
    lines.append(f"\n## Strategy Ranking\n")
    lines.append("| Strategy | Best ipSAE | Mean ipSAE | % Achieved |\n")
    lines.append("|----------|------------|------------|------------|\n")
    strategy_stats = best_per_run.groupby("strategy").agg({
        "best_ipsae": ["min", "mean"],
        "achieved": "mean"
    })
    strategy_stats.columns = ["best", "mean", "pct"]
    strategy_stats = strategy_stats.sort_values("mean")
    for strategy, row in strategy_stats.iterrows():
        label = STRATEGY_LABELS.get(strategy, strategy)
        lines.append(f"| {label} | {row['best']:.4f} | {row['mean']:.4f} | {row['pct']:.1%} |\n")

    # Top sequences
    lines.append(f"\n## Top 10 Sequences\n")
    top_seqs = df.loc[df.groupby("run_id")["best_ipsae"].idxmin()].sort_values("best_ipsae").head(10)
    for rank, (_, row) in enumerate(top_seqs.iterrows(), 1):
        achieved = "✓" if row['best_ipsae'] < IPSAE_THRESHOLD else "✗"
        lines.append(f"\n### {rank}. {achieved} ipSAE = {row['best_ipsae']:.4f}\n")
        lines.append(f"- Scaffold: {row['scaffold']}\n")
        lines.append(f"- Strategy: {row['strategy']}\n")
        lines.append(f"- Cycle: {row['cycle']}\n")
        seq = row['best_sequence']
        lines.append(f"- Length: {len(seq)} aa\n")
        lines.append(f"```\n{seq}\n```\n")

    # Plots
    lines.append(f"\n## Visualizations\n")
    lines.append(f"\n### Convergence by Scaffold\n")
    lines.append(f"![Trajectories](trajectories_by_scaffold.png)\n")
    lines.append(f"\n### Aggregate Convergence\n")
    lines.append(f"![Aggregate](aggregate_trajectories.png)\n")
    lines.append(f"\n### Performance Comparison\n")
    lines.append(f"![Bar Comparison](bar_comparison.png)\n")
    lines.append(f"\n### Heatmap\n")
    lines.append(f"![Heatmap](heatmap_scaffold_strategy.png)\n")

    lines.append(f"\n### Proposal Method Selection (ProFam vs Random)\n")
    lines.append(f"![Proposal Method](proposal_method_selection.png)\n")
    lines.append(f"\n### Aggregate Proposal Method Selection\n")
    lines.append(f"![Proposal Aggregate](proposal_method_aggregate.png)\n")

    # Write file
    md_path = BASE_DIR / "campaign_analysis.md"
    with open(md_path, "w") as f:
        f.write("".join(lines))
    print(f"Saved: {md_path}")
    return md_path


def main():
    print("=" * 70)
    print("2GDZ_15PGDH CAMPAIGN ANALYSIS")
    print("=" * 70)
    print(f"\nStrategies: {STRATEGIES}")

    if not BASE_DIR.exists():
        print(f"\nOutput directory not found: {BASE_DIR}")
        print("Jobs may still be running.")
        return

    df = load_all_results()

    if len(df) == 0:
        print("\nNo results found yet. Jobs may still be running.")
        return

    print(f"\nLoaded {len(df)} cycle records from {df['run_id'].nunique()} runs")
    print(f"Scaffolds ({len(df['scaffold'].unique())}): {sorted(df['scaffold'].unique())}")
    print(f"Strategies ({len(df['strategy'].unique())}): {sorted(df['strategy'].unique())}")

    # Check completion
    max_cycles = df.groupby("run_id")["cycle"].max()
    completed = (max_cycles >= 590).sum()  # Allow some margin
    print(f"Completed runs (>=590 cycles): {completed}/{len(max_cycles)}")

    best_per_run = analyze_threshold_achievement(df)
    traj_df = analyze_convergence(df)
    analyze_random_vs_thompson(df, best_per_run)
    analyze_exploit_bias(df, best_per_run)
    find_best_sequences(df)
    generate_summary(df, best_per_run)

    # Save results
    output_path = BASE_DIR / "campaign_analysis.csv"
    best_per_run.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    plot_trajectories_by_scaffold(df, traj_df)
    plot_aggregate_trajectories(traj_df)
    plot_bar_comparison(best_per_run)
    plot_heatmap(best_per_run)

    # Generate proposal method plots
    print(f"\n--- Proposal Method Analysis ---")
    proposal_df = load_proposal_method_data()
    if len(proposal_df) > 0:
        print(f"Loaded {len(proposal_df)} proposal method records from {proposal_df['run_id'].nunique()} runs")
        plot_proposal_method_selection(proposal_df)
        plot_aggregate_proposal_method(proposal_df)
    else:
        print("No proposal method data found (only thompson_eb* runs have this)")

    # Generate markdown report
    print(f"\n{'='*70}")
    print("GENERATING MARKDOWN REPORT")
    print(f"{'='*70}")
    generate_markdown_report(df, best_per_run)


if __name__ == "__main__":
    main()
