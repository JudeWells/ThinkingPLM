#!/usr/bin/env python
"""
Analyze ensemble results to evaluate if averaging reduces MAD of ipSAE scores.

For sequences that appear multiple times (same target + binder sequence),
compare:
1. MAD of the mean (averaged) ipSAE scores
2. MAD of the non-aggregated (individual) ipSAE scores
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_all_ensemble_data(base_dir: Path):
    """
    Load all ensemble data from cycle_stats.json files.

    Returns dict: (target, sequence) -> list of dicts with keys:
        - mean_ipsae: averaged score
        - individual_ipsaes: list of 3 individual scores
        - run_path: path to the run for debugging
    """
    data = defaultdict(list)

    for stats_file in base_dir.rglob("cycle_stats.json"):
        # Extract target from path: outputs/bench_ensemble_v2/TARGET/scaffold/strategy/
        parts = stats_file.parts
        try:
            # Find bench_ensemble_v2 in path and get target after it
            idx = parts.index("bench_ensemble_v2")
            target = parts[idx + 1]
        except (ValueError, IndexError):
            continue

        run_path = str(stats_file.parent)

        with open(stats_file) as f:
            stats = json.load(f)

        for cycle_key, cycle_data in stats.items():
            # Get sequences from best_sequence and selected_sequences
            sequences_to_check = []

            if "best_sequence" in cycle_data and isinstance(cycle_data["best_sequence"], dict):
                sequences_to_check.append(cycle_data["best_sequence"])

            if "selected_sequences" in cycle_data:
                for seq_data in cycle_data["selected_sequences"]:
                    if isinstance(seq_data, dict):
                        sequences_to_check.append(seq_data)

            for seq_data in sequences_to_check:
                sequence = seq_data.get("sequence")
                if not sequence:
                    continue

                # Get energy data - handle nested structure
                energy_terms = seq_data.get("energy_terms", {})
                if isinstance(energy_terms, dict) and "energy_terms" in energy_terms:
                    # Nested structure
                    inner = energy_terms
                    mean_ipsae = inner.get("energy", seq_data.get("energy"))
                    individual_ipsaes = inner.get("ensemble_energies", [])
                else:
                    # Flat structure
                    mean_ipsae = seq_data.get("energy")
                    individual_ipsaes = seq_data.get("ensemble_energies", [])

                if mean_ipsae is None:
                    continue

                # Skip if no ensemble data
                if not individual_ipsaes or len(individual_ipsaes) < 2:
                    continue

                data[(target, sequence)].append({
                    "mean_ipsae": mean_ipsae,
                    "individual_ipsaes": individual_ipsaes,
                    "run_path": run_path,
                    "cycle": cycle_key,
                })

    return data


def calculate_mad(values):
    """Calculate Mean Absolute Deviation from the mean."""
    values = np.array(values)
    return np.mean(np.abs(values - np.mean(values)))


def analyze_ensemble_mad(base_dir: Path):
    """Main analysis function."""
    print("Loading ensemble data...")
    data = load_all_ensemble_data(base_dir)

    print(f"Found {len(data)} unique (target, sequence) pairs")

    # Find sequences that appear multiple times
    duplicates = {k: v for k, v in data.items() if len(v) > 1}
    print(f"Found {len(duplicates)} sequences appearing multiple times")

    # Collect all mean ipSAE values and individual ipSAE values
    # for sequences that appear multiple times
    all_mean_ipsaes = []
    all_individual_ipsaes = []

    # Also collect per-sequence statistics
    per_sequence_stats = []

    for (target, sequence), observations in duplicates.items():
        means = []
        individuals = []

        for obs in observations:
            mean_val = obs["mean_ipsae"]
            ind_vals = obs["individual_ipsaes"]

            # Skip zero values as requested
            if mean_val == 0.0:
                continue
            if all(v == 0.0 for v in ind_vals):
                continue

            means.append(mean_val)
            individuals.extend(ind_vals)

        if len(means) >= 2:  # Need at least 2 observations for MAD
            mad_mean = calculate_mad(means)
            mad_individual = calculate_mad(individuals)

            all_mean_ipsaes.extend(means)
            all_individual_ipsaes.extend(individuals)

            per_sequence_stats.append({
                "target": target,
                "sequence": sequence[:30] + "...",
                "n_observations": len(means),
                "mad_mean": mad_mean,
                "mad_individual": mad_individual,
                "reduction": (mad_individual - mad_mean) / mad_individual * 100 if mad_individual > 0 else 0,
            })

    print(f"\n{'='*60}")
    print("ENSEMBLE MAD ANALYSIS")
    print(f"{'='*60}")

    if not per_sequence_stats:
        print("\nNo sequences found with multiple non-zero observations.")
        print("Let me check for any sequences with ensemble data...")

        # Alternative: look at within-ensemble variance
        ensemble_stds = []
        ensemble_ranges = []
        count = 0

        for (target, sequence), observations in data.items():
            for obs in observations:
                ind_vals = obs["individual_ipsaes"]
                mean_val = obs["mean_ipsae"]

                # Skip zeros
                if mean_val == 0.0 or all(v == 0.0 for v in ind_vals):
                    continue

                if len(ind_vals) >= 2:
                    ensemble_stds.append(np.std(ind_vals))
                    ensemble_ranges.append(max(ind_vals) - min(ind_vals))
                    count += 1

        print(f"\nFound {count} sequences with ensemble data (ensemble_n >= 2)")
        if ensemble_stds:
            print(f"\nWithin-ensemble statistics:")
            print(f"  Mean ensemble std:   {np.mean(ensemble_stds):.4f}")
            print(f"  Median ensemble std: {np.median(ensemble_stds):.4f}")
            print(f"  Mean ensemble range: {np.mean(ensemble_ranges):.4f}")
            print(f"  Median ensemble range: {np.median(ensemble_ranges):.4f}")

            print(f"\nThis shows the typical noise in ipSAE across 3 Boltz runs")
            print(f"Averaging reduces this noise by ~1/sqrt(3) ≈ 42%")

        return

    # Overall statistics
    overall_mad_mean = calculate_mad(all_mean_ipsaes)
    overall_mad_individual = calculate_mad(all_individual_ipsaes)

    print(f"\nSequences with multiple observations: {len(per_sequence_stats)}")
    print(f"Total mean ipSAE values: {len(all_mean_ipsaes)}")
    print(f"Total individual ipSAE values: {len(all_individual_ipsaes)}")

    print(f"\n--- Overall MAD Comparison ---")
    print(f"MAD of averaged (mean) ipSAE scores:     {overall_mad_mean:.6f}")
    print(f"MAD of individual (non-avg) ipSAE scores: {overall_mad_individual:.6f}")

    if overall_mad_individual > 0:
        reduction = (overall_mad_individual - overall_mad_mean) / overall_mad_individual * 100
        print(f"Reduction in MAD from averaging:          {reduction:.1f}%")

    print(f"\n--- Per-sequence Statistics ---")
    avg_reduction = np.mean([s["reduction"] for s in per_sequence_stats])
    print(f"Average MAD reduction per sequence: {avg_reduction:.1f}%")

    # Show a few examples
    print(f"\nTop 5 sequences with highest MAD reduction:")
    sorted_stats = sorted(per_sequence_stats, key=lambda x: x["reduction"], reverse=True)[:5]
    for s in sorted_stats:
        print(f"  {s['target']}: {s['sequence']} | n={s['n_observations']} | "
              f"MAD_mean={s['mad_mean']:.4f} | MAD_ind={s['mad_individual']:.4f} | "
              f"reduction={s['reduction']:.1f}%")


def analyze_within_ensemble_variance(base_dir: Path):
    """
    Alternative analysis: Look at variance WITHIN each ensemble of 3.

    This directly measures the noise in Boltz predictions that ensemble
    averaging is meant to reduce.
    """
    print("\n" + "="*60)
    print("WITHIN-ENSEMBLE VARIANCE ANALYSIS")
    print("="*60)

    data = load_all_ensemble_data(base_dir)

    ensemble_stds = []
    ensemble_ranges = []
    ensemble_mads = []
    mean_values = []

    for (target, sequence), observations in data.items():
        for obs in observations:
            ind_vals = np.array(obs["individual_ipsaes"])
            mean_val = obs["mean_ipsae"]

            # Skip zeros
            if mean_val == 0.0 or all(v == 0.0 for v in ind_vals):
                continue

            if len(ind_vals) >= 2:
                ensemble_stds.append(np.std(ind_vals))
                ensemble_ranges.append(np.max(ind_vals) - np.min(ind_vals))
                ensemble_mads.append(calculate_mad(ind_vals))
                mean_values.append(mean_val)

    if not ensemble_stds:
        print("No ensemble data found.")
        return

    ensemble_stds = np.array(ensemble_stds)
    ensemble_ranges = np.array(ensemble_ranges)
    ensemble_mads = np.array(ensemble_mads)
    mean_values = np.array(mean_values)

    print(f"\nAnalyzed {len(ensemble_stds)} sequences with ensemble data")
    print(f"\nWithin-ensemble noise statistics (across 3 Boltz runs):")
    print(f"  Mean std:    {np.mean(ensemble_stds):.4f} (median: {np.median(ensemble_stds):.4f})")
    print(f"  Mean range:  {np.mean(ensemble_ranges):.4f} (median: {np.median(ensemble_ranges):.4f})")
    print(f"  Mean MAD:    {np.mean(ensemble_mads):.4f} (median: {np.median(ensemble_mads):.4f})")

    # Calculate coefficient of variation (relative noise)
    # Avoid division by zero for values close to 0
    valid_mask = np.abs(mean_values) > 0.01
    if np.sum(valid_mask) > 0:
        cv = ensemble_stds[valid_mask] / np.abs(mean_values[valid_mask])
        print(f"\nRelative noise (CV = std/|mean|) for |mean| > 0.01:")
        print(f"  Mean CV:   {np.mean(cv):.2%}")
        print(f"  Median CV: {np.median(cv):.2%}")

    # Theoretical benefit of averaging
    print(f"\nTheoretical benefit of averaging 3 samples:")
    print(f"  Expected std reduction: 1/sqrt(3) = {1/np.sqrt(3):.2%} of original")
    print(f"  Expected MAD of mean vs MAD of individuals: {1/np.sqrt(3):.2%}")

    # Distribution of noise
    print(f"\nDistribution of within-ensemble std:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(ensemble_stds, p):.4f}")


if __name__ == "__main__":
    base_dir = Path("outputs/bench_ensemble_v2")

    # Main analysis
    analyze_ensemble_mad(base_dir)

    # Within-ensemble variance analysis
    analyze_within_ensemble_variance(base_dir)
