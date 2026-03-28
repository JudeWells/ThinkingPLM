#!/usr/bin/env python
"""
Analyze benchmark results comparing different optimization strategies.

Focus on:
1. How often each strategy achieves ipSAE < -0.6 (strong binding)
2. Convergence speed to good solutions
3. Best scores achieved per target/scaffold combination
4. Insights for improving Thompson sampling
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Threshold for "good" binding (ipSAE is negative, more negative = better)
IPSAE_THRESHOLD = -0.6


def load_all_results(base_dir: Path) -> pd.DataFrame:
    """Load all results into a DataFrame."""
    rows = []

    for stats_file in base_dir.rglob("cycle_stats.json"):
        parts = stats_file.parts
        try:
            idx = parts.index("bench_ensemble_v2")
            target = parts[idx + 1]
            scaffold = parts[idx + 2]
            strategy = parts[idx + 3]
        except (ValueError, IndexError):
            continue

        with open(stats_file) as f:
            stats = json.load(f)

        for cycle_key, cycle_data in stats.items():
            cycle = int(cycle_key)

            # Get best energy for this cycle
            best_energy = cycle_data.get("all_min_energy", float("inf"))
            avg_energy = cycle_data.get("all_avg_energy", float("inf"))

            # Get best sequence info
            best_seq_data = cycle_data.get("best_sequence", {})
            if isinstance(best_seq_data, dict):
                best_seq = best_seq_data.get("sequence", "")
                # Handle nested energy_terms structure
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
                "target": target,
                "scaffold": scaffold,
                "strategy": strategy,
                "cycle": cycle,
                "best_ipsae": ipsae,
                "avg_ipsae": avg_energy,
                "best_sequence": best_seq,
                "run_id": f"{target}/{scaffold}/{strategy}",
            })

    return pd.DataFrame(rows)


def analyze_threshold_achievement(df: pd.DataFrame, threshold: float = IPSAE_THRESHOLD):
    """Analyze how often each strategy achieves the threshold."""
    print(f"\n{'='*70}")
    print(f"THRESHOLD ACHIEVEMENT ANALYSIS (ipSAE < {threshold})")
    print(f"{'='*70}")

    # Get the best ipSAE achieved per run
    best_per_run = df.groupby(["target", "scaffold", "strategy"]).agg({
        "best_ipsae": "min",  # min because more negative is better
        "cycle": "max",  # total cycles
    }).reset_index()

    # Check which runs achieved the threshold
    best_per_run["achieved_threshold"] = best_per_run["best_ipsae"] < threshold

    # Summary by strategy
    print(f"\n--- Runs Achieving ipSAE < {threshold} by Strategy ---")
    strategy_summary = best_per_run.groupby("strategy").agg({
        "achieved_threshold": ["sum", "count", "mean"],
        "best_ipsae": ["min", "mean", "median"],
    }).round(4)
    strategy_summary.columns = ["n_achieved", "n_total", "pct_achieved",
                                 "best_ipsae", "mean_best_ipsae", "median_best_ipsae"]
    strategy_summary = strategy_summary.sort_values("pct_achieved", ascending=False)
    print(strategy_summary.to_string())

    # Summary by target
    print(f"\n--- Runs Achieving Threshold by Target ---")
    target_summary = best_per_run.groupby("target").agg({
        "achieved_threshold": ["sum", "count", "mean"],
        "best_ipsae": "min",
    }).round(4)
    target_summary.columns = ["n_achieved", "n_total", "pct_achieved", "best_ipsae"]
    print(target_summary.to_string())

    # Summary by scaffold
    print(f"\n--- Runs Achieving Threshold by Scaffold ---")
    scaffold_summary = best_per_run.groupby("scaffold").agg({
        "achieved_threshold": ["sum", "count", "mean"],
        "best_ipsae": "min",
    }).round(4)
    scaffold_summary.columns = ["n_achieved", "n_total", "pct_achieved", "best_ipsae"]
    print(scaffold_summary.to_string())

    # Cross-tabulation: strategy x target
    print(f"\n--- Best ipSAE by Strategy x Target ---")
    pivot = best_per_run.pivot_table(
        index="strategy",
        columns="target",
        values="best_ipsae",
        aggfunc="min"
    ).round(4)
    print(pivot.to_string())

    # Cross-tabulation: strategy x scaffold
    print(f"\n--- Best ipSAE by Strategy x Scaffold ---")
    pivot = best_per_run.pivot_table(
        index="strategy",
        columns="scaffold",
        values="best_ipsae",
        aggfunc="min"
    ).round(4)
    print(pivot.to_string())

    return best_per_run


def analyze_convergence(df: pd.DataFrame, threshold: float = IPSAE_THRESHOLD):
    """Analyze convergence speed to threshold."""
    print(f"\n{'='*70}")
    print(f"CONVERGENCE SPEED ANALYSIS")
    print(f"{'='*70}")

    # For each run, find first cycle that achieved threshold
    convergence_data = []

    for run_id in df["run_id"].unique():
        run_df = df[df["run_id"] == run_id].sort_values("cycle")

        # Track cumulative best
        cumulative_best = float("inf")
        first_threshold_cycle = None

        for _, row in run_df.iterrows():
            cumulative_best = min(cumulative_best, row["best_ipsae"])
            if cumulative_best < threshold and first_threshold_cycle is None:
                first_threshold_cycle = row["cycle"]

        convergence_data.append({
            "run_id": run_id,
            "target": run_df.iloc[0]["target"],
            "scaffold": run_df.iloc[0]["scaffold"],
            "strategy": run_df.iloc[0]["strategy"],
            "first_threshold_cycle": first_threshold_cycle,
            "final_best": cumulative_best,
            "total_cycles": run_df["cycle"].max(),
        })

    conv_df = pd.DataFrame(convergence_data)

    # Among runs that achieved threshold, average cycles to reach it
    achieved = conv_df[conv_df["first_threshold_cycle"].notna()]

    print(f"\n--- Cycles to Reach Threshold (among successful runs) ---")
    if len(achieved) > 0:
        conv_summary = achieved.groupby("strategy").agg({
            "first_threshold_cycle": ["mean", "median", "min", "max", "count"],
        }).round(1)
        conv_summary.columns = ["mean_cycles", "median_cycles", "min_cycles", "max_cycles", "n_achieved"]
        conv_summary = conv_summary.sort_values("median_cycles")
        print(conv_summary.to_string())
    else:
        print("No runs achieved the threshold.")

    return conv_df


def analyze_trajectory(df: pd.DataFrame):
    """Analyze optimization trajectories."""
    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS")
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
                "strategy": row["strategy"],
                "target": row["target"],
                "scaffold": row["scaffold"],
                "cycle": row["cycle"],
                "cumulative_best": cumulative_best,
            })

    traj_df = pd.DataFrame(trajectory_data)

    # Average cumulative best by strategy at key cycles
    key_cycles = [0, 10, 25, 50, 75, 100]

    print(f"\n--- Average Cumulative Best ipSAE at Key Cycles ---")
    cycle_summary = []
    for cycle in key_cycles:
        cycle_data = traj_df[traj_df["cycle"] == cycle]
        if len(cycle_data) > 0:
            by_strategy = cycle_data.groupby("strategy")["cumulative_best"].mean()
            cycle_summary.append(by_strategy)

    if cycle_summary:
        summary_df = pd.DataFrame(cycle_summary, index=key_cycles).T.round(4)
        summary_df.columns = [f"cycle_{c}" for c in key_cycles]
        print(summary_df.to_string())

    # Improvement from cycle 0 to final
    print(f"\n--- Improvement from Start to End ---")
    start_end = traj_df.groupby(["run_id", "strategy"]).agg({
        "cumulative_best": ["first", "last"],
    })
    start_end.columns = ["start", "end"]
    start_end["improvement"] = start_end["start"] - start_end["end"]

    improvement_by_strategy = start_end.groupby("strategy")["improvement"].agg(["mean", "median", "std"]).round(4)
    print(improvement_by_strategy.to_string())

    return traj_df


def analyze_thompson_specific(df: pd.DataFrame):
    """Analyze Thompson sampling specific metrics."""
    print(f"\n{'='*70}")
    print(f"THOMPSON SAMPLING ANALYSIS")
    print(f"{'='*70}")

    thompson_strategies = ["bandit_thompson", "random_thompson", "thompson_eb8_bandit_rel"]
    non_thompson = ["bandit_greedy", "random_greedy"]

    # Compare Thompson vs non-Thompson
    best_per_run = df.groupby(["target", "scaffold", "strategy"]).agg({
        "best_ipsae": "min",
    }).reset_index()

    print(f"\n--- Thompson vs Greedy Comparison ---")
    for scaffold in df["scaffold"].unique():
        print(f"\n  Scaffold: {scaffold}")
        scaffold_data = best_per_run[best_per_run["scaffold"] == scaffold]

        for strategy in scaffold_data["strategy"].unique():
            strat_data = scaffold_data[scaffold_data["strategy"] == strategy]
            mean_best = strat_data["best_ipsae"].mean()
            min_best = strat_data["best_ipsae"].min()
            print(f"    {strategy:30s}: mean_best={mean_best:.4f}, min_best={min_best:.4f}")


def generate_recommendations(df: pd.DataFrame, best_per_run: pd.DataFrame):
    """Generate recommendations for improving strategies."""
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR IMPROVEMENT")
    print(f"{'='*70}")

    # Identify best performing strategy overall
    strategy_perf = best_per_run.groupby("strategy").agg({
        "best_ipsae": ["mean", "min"],
        "achieved_threshold": "mean",
    })
    strategy_perf.columns = ["mean_best", "overall_best", "pct_achieved"]
    strategy_perf = strategy_perf.sort_values("mean_best")

    best_strategy = strategy_perf.index[0]
    worst_strategy = strategy_perf.index[-1]

    print(f"\n1. STRATEGY RANKING (by mean best ipSAE):")
    for i, (strategy, row) in enumerate(strategy_perf.iterrows(), 1):
        print(f"   {i}. {strategy}: mean={row['mean_best']:.4f}, best={row['overall_best']:.4f}, "
              f"threshold_rate={row['pct_achieved']:.1%}")

    print(f"\n2. KEY OBSERVATIONS:")

    # Check if Thompson helps
    thompson_mean = strategy_perf.loc[strategy_perf.index.str.contains("thompson"), "mean_best"].mean()
    greedy_mean = strategy_perf.loc[strategy_perf.index.str.contains("greedy"), "mean_best"].mean()

    if thompson_mean < greedy_mean:
        print(f"   - Thompson sampling outperforms greedy (mean: {thompson_mean:.4f} vs {greedy_mean:.4f})")
    else:
        print(f"   - Greedy outperforms Thompson sampling (mean: {greedy_mean:.4f} vs {thompson_mean:.4f})")

    # Check if bandit (ProFam) helps vs random mutation
    bandit_mean = strategy_perf.loc[strategy_perf.index.str.contains("bandit"), "mean_best"].mean()
    random_mean = strategy_perf.loc[strategy_perf.index.str.contains("random") &
                                     ~strategy_perf.index.str.contains("bandit"), "mean_best"].mean()

    if bandit_mean < random_mean:
        print(f"   - ProFam (bandit) outperforms random mutation (mean: {bandit_mean:.4f} vs {random_mean:.4f})")
    else:
        print(f"   - Random mutation outperforms ProFam (mean: {random_mean:.4f} vs {bandit_mean:.4f})")

    # Best target/scaffold combinations
    print(f"\n3. EASIEST TARGETS (highest success rate):")
    target_success = best_per_run.groupby("target")["achieved_threshold"].mean().sort_values(ascending=False)
    for target, rate in target_success.items():
        print(f"   - {target}: {rate:.1%}")

    print(f"\n4. BEST SCAFFOLDS:")
    scaffold_success = best_per_run.groupby("scaffold")["achieved_threshold"].mean().sort_values(ascending=False)
    for scaffold, rate in scaffold_success.items():
        print(f"   - {scaffold}: {rate:.1%}")

    print(f"\n5. SUGGESTIONS FOR THOMPSON SAMPLING IMPROVEMENTS:")
    print(f"   a) Explore-exploit balance:")
    print(f"      - If Thompson underperforms, try increasing exploit_bias")
    print(f"      - Consider temperature annealing for exploration->exploitation transition")
    print(f"   b) Arm management:")
    print(f"      - Review thompson_max_arms setting - may need more diversity")
    print(f"      - Consider different max_identity thresholds for pruning")
    print(f"   c) Reward signal:")
    print(f"      - Ensemble averaging (done) reduces noise in rewards")
    print(f"      - Consider reward shaping (e.g., bonus for crossing thresholds)")
    print(f"   d) Prior tuning:")
    print(f"      - Adjust Beta prior (alpha, beta) based on observed reward distribution")
    print(f"   e) Hybrid approaches:")
    print(f"      - thompson_eb8_bandit_rel combines proposal bandit with Thompson")
    print(f"      - Consider adaptive switching between strategies")


def main():
    base_dir = Path("outputs/bench_ensemble_v2")

    print("Loading results...")
    df = load_all_results(base_dir)
    print(f"Loaded {len(df)} cycle records from {df['run_id'].nunique()} runs")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Targets: {df['target'].unique().tolist()}")
    print(f"Scaffolds: {df['scaffold'].unique().tolist()}")

    # Main analyses
    best_per_run = analyze_threshold_achievement(df)
    conv_df = analyze_convergence(df)
    traj_df = analyze_trajectory(df)
    analyze_thompson_specific(df)
    generate_recommendations(df, best_per_run)

    # Save detailed results
    best_per_run.to_csv("outputs/bench_ensemble_v2/strategy_comparison.csv", index=False)
    print(f"\nDetailed results saved to outputs/bench_ensemble_v2/strategy_comparison.csv")


if __name__ == "__main__":
    main()
