#!/usr/bin/env python
"""
Analyze the relationship between number of mutations and ipSAE improvement.

For random_mutation proposals, count how many mutations were applied and
compare the resulting ipSAE to the parent/prompt sequence.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def count_mutations(name: str) -> int:
    """Count mutations from name like 'random_mutant_0_G48S+E62A+K59P'."""
    if "random_mutant" not in name:
        return 0

    # Extract mutation part after the last underscore group
    match = re.search(r'random_mutant_\d+_(.+)', name)
    if not match:
        return 0

    mutation_str = match.group(1)
    # Count mutations by splitting on '+'
    mutations = mutation_str.split('+')
    return len(mutations)


def load_and_analyze(base_dir: Path):
    """Load all CSV files and analyze mutations vs improvement."""

    all_data = []

    for csv_file in base_dir.rglob("all_sequences.csv"):
        # Extract run info from path
        parts = csv_file.parts
        try:
            idx = parts.index("bench_ensemble_v2")
            target = parts[idx + 1]
            scaffold = parts[idx + 2]
            strategy = parts[idx + 3]
        except (ValueError, IndexError):
            continue

        df = pd.read_csv(csv_file)

        # Track the best ipSAE seen so far (cumulative best)
        cumulative_best = float('inf')
        prev_best = float('inf')

        for _, row in df.iterrows():
            cycle = row['cycle']
            proposal_method = row['proposal_method']
            name = row['name']
            ipsae = row['ipSAE']

            # Update cumulative best
            if cycle == 0:
                cumulative_best = ipsae
                prev_best = ipsae
                continue

            # Only analyze random_mutation proposals
            if proposal_method != 'random_mutation':
                if ipsae < cumulative_best:
                    cumulative_best = ipsae
                prev_best = cumulative_best
                continue

            n_mutations = count_mutations(name)
            if n_mutations == 0:
                continue

            # Calculate improvement (negative = better, so improvement = prev - current)
            improvement = prev_best - ipsae

            all_data.append({
                'target': target,
                'scaffold': scaffold,
                'strategy': strategy,
                'cycle': cycle,
                'n_mutations': n_mutations,
                'ipsae': ipsae,
                'prev_best': prev_best,
                'improvement': improvement,
                'is_improvement': improvement > 0,
            })

            # Update cumulative best
            if ipsae < cumulative_best:
                cumulative_best = ipsae
            prev_best = cumulative_best

    return pd.DataFrame(all_data)


def analyze_results(df: pd.DataFrame):
    """Analyze and print results."""

    print("=" * 70)
    print("MUTATIONS VS IMPROVEMENT ANALYSIS")
    print("=" * 70)

    print(f"\nTotal random_mutation samples analyzed: {len(df)}")
    print(f"Samples by number of mutations:")
    print(df['n_mutations'].value_counts().sort_index().to_string())

    # Filter out zeros (no improvement possible from inf)
    df_valid = df[df['prev_best'] < 0].copy()  # Only where we have a valid baseline
    print(f"\nSamples with valid baseline (prev_best < 0): {len(df_valid)}")

    if len(df_valid) == 0:
        print("Not enough data with valid baselines.")
        # Analyze all data instead
        df_valid = df.copy()

    print("\n" + "-" * 70)
    print("IMPROVEMENT BY NUMBER OF MUTATIONS")
    print("-" * 70)

    # Group by n_mutations
    grouped = df.groupby('n_mutations').agg({
        'improvement': ['mean', 'std', 'median', 'count'],
        'is_improvement': ['sum', 'mean'],
        'ipsae': ['mean', 'min'],
    }).round(4)

    grouped.columns = ['mean_impr', 'std_impr', 'median_impr', 'count',
                       'n_improvements', 'pct_improvement',
                       'mean_ipsae', 'min_ipsae']

    print(grouped.to_string())

    print("\n" + "-" * 70)
    print("KEY METRICS BY MUTATION COUNT")
    print("-" * 70)

    for n_mut in sorted(df['n_mutations'].unique()):
        subset = df[df['n_mutations'] == n_mut]
        n_total = len(subset)
        n_improved = subset['is_improvement'].sum()
        pct_improved = n_improved / n_total * 100 if n_total > 0 else 0

        mean_impr = subset['improvement'].mean()
        mean_ipsae = subset['ipsae'].mean()
        best_ipsae = subset['ipsae'].min()

        # Among improvements, what's the average improvement?
        improvements_only = subset[subset['is_improvement']]
        avg_when_improved = improvements_only['improvement'].mean() if len(improvements_only) > 0 else 0

        print(f"\n{n_mut} mutation(s):")
        print(f"  Total samples: {n_total}")
        print(f"  Improved: {n_improved} ({pct_improved:.1f}%)")
        print(f"  Mean improvement: {mean_impr:.4f}")
        print(f"  Avg improvement (when improved): {avg_when_improved:.4f}")
        print(f"  Mean ipSAE: {mean_ipsae:.4f}")
        print(f"  Best ipSAE: {best_ipsae:.4f}")

    # Statistical test: is there a significant difference?
    print("\n" + "-" * 70)
    print("CORRELATION ANALYSIS")
    print("-" * 70)

    # Correlation between n_mutations and improvement
    corr = df['n_mutations'].corr(df['improvement'])
    print(f"Correlation (n_mutations vs improvement): {corr:.4f}")

    # Correlation between n_mutations and absolute ipSAE
    corr_ipsae = df['n_mutations'].corr(df['ipsae'])
    print(f"Correlation (n_mutations vs ipSAE): {corr_ipsae:.4f}")

    # Correlation between n_mutations and probability of improvement
    corr_prob = df.groupby('n_mutations')['is_improvement'].mean()
    print(f"\nProbability of improvement by mutation count:")
    for n, p in corr_prob.items():
        print(f"  {n} mutations: {p:.1%}")

    print("\n" + "-" * 70)
    print("BREAKDOWN BY TARGET")
    print("-" * 70)

    for target in df['target'].unique():
        target_df = df[df['target'] == target]
        print(f"\n{target}:")

        by_mut = target_df.groupby('n_mutations').agg({
            'is_improvement': 'mean',
            'improvement': 'mean',
            'ipsae': 'min',
        }).round(4)
        by_mut.columns = ['pct_improved', 'mean_impr', 'best_ipsae']
        print(by_mut.to_string())

    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)

    # Find optimal mutation count
    improvement_rate = df.groupby('n_mutations')['is_improvement'].mean()
    best_n = improvement_rate.idxmax()
    best_rate = improvement_rate.max()

    mean_improvement_when_good = df.groupby('n_mutations').apply(
        lambda x: x[x['is_improvement']]['improvement'].mean() if x['is_improvement'].sum() > 0 else 0
    )

    # Expected improvement = P(improvement) * E[improvement | improvement]
    expected_improvement = improvement_rate * mean_improvement_when_good
    best_expected_n = expected_improvement.idxmax()

    print(f"\nHighest improvement rate: {best_n} mutations ({best_rate:.1%})")
    print(f"Best expected improvement: {best_expected_n} mutations ({expected_improvement[best_expected_n]:.4f})")

    print(f"\nExpected improvement by mutation count:")
    for n in sorted(expected_improvement.index):
        print(f"  {n} mutations: {expected_improvement[n]:.4f} "
              f"(rate={improvement_rate[n]:.1%}, avg_when_good={mean_improvement_when_good[n]:.4f})")


def main():
    base_dir = Path("outputs/bench_ensemble_v2")

    print("Loading data...")
    df = load_and_analyze(base_dir)

    if len(df) == 0:
        print("No random_mutation data found.")
        return

    analyze_results(df)

    # Save detailed data
    output_path = base_dir / "mutations_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed data saved to {output_path}")


if __name__ == "__main__":
    main()
