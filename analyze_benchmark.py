#!/usr/bin/env python
"""
Analyze benchmark results from the iterative prompt-updating experiment.

Reads all_sequences.csv files from the output tree and produces:
1. Convergence curves — cumulative-best ipSAE vs cycle for all 4 conditions
2. Summary table — best ipSAE, cycle achieved, improvement over seed
3. Paired comparison — delta ipSAE across targets with statistical test
4. Diversity — similarity_to_initial over cycles per condition
5. PAE heatmaps — best structure per condition per target
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

# ── Configuration ───────────────────────────────────────────────────────────

TARGETS = ["2GDZ", "4ZQK", "3DI2", "1WWW", "4LXV", "1VPF"]
CONDITIONS = ["profam_update", "profam_frozen", "random_update", "random_frozen"]
INITS = {
    "2GDZ": ["scaffold", "denovo"],
    "4ZQK": ["scaffold"],
    "3DI2": ["scaffold"],
    "1WWW": ["scaffold"],
    "4LXV": ["scaffold"],
    "1VPF": ["scaffold"],
}

CONDITION_COLORS = {
    "profam_update": "#00bfff",
    "profam_frozen": "#ff6b6b",
    "random_update": "#00e676",
    "random_frozen": "#ffab40",
}

CONDITION_LABELS = {
    "profam_update": "ProFam + update",
    "profam_frozen": "ProFam + frozen",
    "random_update": "Random + update",
    "random_frozen": "Random + frozen",
}

BENCH_ROOT = Path("outputs/bench")
ANALYSIS_DIR = Path("outputs/bench/analysis")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load all_sequences.csv into a list of row dicts."""
    if not csv_path.is_file():
        return []
    with csv_path.open("r") as f:
        return list(csv.DictReader(f))


def compute_cumulative_best(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float]]:
    """Return (cycles, cumulative_best_energy) from CSV rows."""
    if not rows:
        return [], []
    # Group by cycle, find min energy per cycle, then cumulative min.
    cycle_mins: Dict[int, float] = {}
    for row in rows:
        try:
            cycle = int(row["cycle"])
            energy = float(row["total_energy"])
        except (ValueError, KeyError):
            continue
        if energy == float("inf"):
            continue
        if cycle not in cycle_mins or energy < cycle_mins[cycle]:
            cycle_mins[cycle] = energy

    if not cycle_mins:
        return [], []

    cycles = sorted(cycle_mins.keys())
    cum_best = []
    best_so_far = float("inf")
    for c in cycles:
        best_so_far = min(best_so_far, cycle_mins[c])
        cum_best.append(best_so_far)

    return cycles, cum_best


def detect_metric_columns(rows: List[Dict[str, str]]) -> List[str]:
    """Detect all numeric metric columns in the CSV (energy terms + structural metrics)."""
    if not rows:
        return []
    skip = {"cycle", "name", "sequence", "length"}
    candidates = [k for k in rows[0].keys() if k not in skip]
    # Keep only columns that have at least one parseable float value.
    metric_cols = []
    for col in candidates:
        for row in rows:
            val = row.get(col, "")
            if val == "":
                continue
            try:
                float(val)
                metric_cols.append(col)
                break
            except ValueError:
                break
    return metric_cols


def compute_per_cycle_metric(
    rows: List[Dict[str, str]],
    column: str,
    agg: str = "best",
) -> Tuple[List[int], List[float]]:
    """Return (cycles, values) for a given column.

    agg='best' returns the value from the row with the best (lowest)
    total_energy in each cycle. agg='mean' returns the cycle mean.
    """
    if not rows:
        return [], []

    cycle_data: Dict[int, List[Tuple[float, float]]] = {}  # cycle -> [(energy, metric)]
    for row in rows:
        try:
            cycle = int(row["cycle"])
            val = float(row.get(column, ""))
            energy = float(row.get("total_energy", "inf"))
        except (ValueError, KeyError):
            continue
        cycle_data.setdefault(cycle, []).append((energy, val))

    if not cycle_data:
        return [], []

    cycles = sorted(cycle_data.keys())
    values = []
    for c in cycles:
        pairs = cycle_data[c]
        if agg == "mean":
            values.append(float(np.mean([v for _, v in pairs])))
        else:  # best — value from the row with lowest energy
            pairs.sort(key=lambda x: x[0])
            values.append(pairs[0][1])
    return cycles, values


def compute_diversity(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float]]:
    """Return (cycles, mean_similarity_to_initial) per cycle."""
    if not rows:
        return [], []
    cycle_sims: Dict[int, List[float]] = {}
    for row in rows:
        try:
            cycle = int(row["cycle"])
            sim = float(row["similarity_to_initial"])
        except (ValueError, KeyError):
            continue
        cycle_sims.setdefault(cycle, []).append(sim)

    cycles = sorted(cycle_sims.keys())
    means = [float(np.mean(cycle_sims[c])) for c in cycles]
    return cycles, means


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_convergence_curves(target: str, init: str):
    """Plot cumulative-best ipSAE vs cycle for all 4 conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        csv_path = BENCH_ROOT / target / init / cond / "all_sequences.csv"
        rows = load_csv(csv_path)
        cycles, cum_best = compute_cumulative_best(rows)
        if cycles:
            ax.plot(
                cycles, cum_best,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                linewidth=2,
            )

    ax.set_xlabel("Cycle", fontsize=13)
    ax.set_ylabel("Cumulative Best ipSAE Energy", fontsize=13)
    ax.set_title(f"Convergence: {target} ({init})", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    out_path = ANALYSIS_DIR / f"convergence_{target}_{init}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_diversity(target: str, init: str):
    """Plot mean similarity to initial sequence vs cycle."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        csv_path = BENCH_ROOT / target / init / cond / "all_sequences.csv"
        rows = load_csv(csv_path)
        cycles, means = compute_diversity(rows)
        if cycles:
            ax.plot(
                cycles, means,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                linewidth=2,
            )

    ax.set_xlabel("Cycle", fontsize=13)
    ax.set_ylabel("Mean Similarity to Initial", fontsize=13)
    ax.set_title(f"Diversity: {target} ({init})", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    out_path = ANALYSIS_DIR / f"diversity_{target}_{init}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_summary_table():
    """Print and save a summary table: best ipSAE, cycle achieved, improvement."""
    header = f"{'Target':<8} {'Init':<10} {'Condition':<18} {'Best ipSAE':>12} {'@ Cycle':>8} {'Δ vs seed':>10}"
    lines = [header, "-" * len(header)]

    for target in TARGETS:
        for init in INITS[target]:
            # Load seed energy from cycle_stats.json.
            seed_energy = None
            stats_path = BENCH_ROOT / target / init / "profam_update" / "cycle_stats.json"
            if stats_path.is_file():
                with stats_path.open("r") as f:
                    stats = json.load(f)
                if "0" in stats:
                    seed_energy = stats["0"].get("all_min_energy")

            for cond in CONDITIONS:
                csv_path = BENCH_ROOT / target / init / cond / "all_sequences.csv"
                rows = load_csv(csv_path)
                cycles, cum_best = compute_cumulative_best(rows)
                if cum_best:
                    best_e = cum_best[-1]
                    best_cycle = cycles[cum_best.index(best_e)] if best_e in cum_best else cycles[-1]
                    # Find the first cycle where cum_best reached its final value
                    for i, (c, e) in enumerate(zip(cycles, cum_best)):
                        if e == best_e:
                            best_cycle = c
                            break
                    delta = (best_e - seed_energy) if seed_energy is not None else float("nan")
                    lines.append(
                        f"{target:<8} {init:<10} {cond:<18} {best_e:>12.4f} {best_cycle:>8d} {delta:>10.4f}"
                    )
                else:
                    lines.append(
                        f"{target:<8} {init:<10} {cond:<18} {'N/A':>12} {'N/A':>8} {'N/A':>10}"
                    )

    table = "\n".join(lines)
    print("\n" + table + "\n")

    out_path = ANALYSIS_DIR / "summary_table.txt"
    with out_path.open("w") as f:
        f.write(table + "\n")
    print(f"  Saved {out_path}")


def paired_comparison():
    """
    Compare profam_update vs each baseline across targets.
    Uses Wilcoxon signed-rank test if scipy is available.
    """
    from collections import defaultdict

    results: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for target in TARGETS:
        for init in INITS[target]:
            # Load profam_update best
            csv_pu = BENCH_ROOT / target / init / "profam_update" / "all_sequences.csv"
            rows_pu = load_csv(csv_pu)
            _, cum_pu = compute_cumulative_best(rows_pu)
            if not cum_pu:
                continue
            best_pu = cum_pu[-1]

            for baseline in ["profam_frozen", "random_update", "random_frozen"]:
                csv_bl = BENCH_ROOT / target / init / baseline / "all_sequences.csv"
                rows_bl = load_csv(csv_bl)
                _, cum_bl = compute_cumulative_best(rows_bl)
                if not cum_bl:
                    continue
                best_bl = cum_bl[-1]
                results[baseline].append((best_pu, best_bl))

    lines = ["Paired Comparison: profam_update vs baselines", "=" * 50]

    for baseline, pairs in results.items():
        deltas = [pu - bl for pu, bl in pairs]
        mean_delta = np.mean(deltas)
        lines.append(
            f"\nvs {baseline} (n={len(pairs)}):"
            f"\n  Mean Δ (update - baseline) = {mean_delta:.4f}"
            f"\n  All Δ: {[f'{d:.4f}' for d in deltas]}"
        )
        # Negative delta means profam_update is better (lower energy).
        wins = sum(1 for d in deltas if d < 0)
        lines.append(f"  Wins: {wins}/{len(deltas)}")

        try:
            from scipy.stats import wilcoxon
            if len(deltas) >= 5:
                stat, pval = wilcoxon(deltas)
                lines.append(f"  Wilcoxon p-value: {pval:.4f}")
            else:
                lines.append(f"  (too few pairs for Wilcoxon test)")
        except ImportError:
            lines.append("  (scipy not available for statistical test)")

    report = "\n".join(lines)
    print("\n" + report + "\n")

    out_path = ANALYSIS_DIR / "paired_comparison.txt"
    with out_path.open("w") as f:
        f.write(report + "\n")
    print(f"  Saved {out_path}")


def plot_pae_heatmaps(target: str, init: str):
    """Plot PAE heatmap for the best structure per condition."""
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 5))
    if len(CONDITIONS) == 1:
        axes = [axes]

    for ax, cond in zip(axes, CONDITIONS):
        # Find best structure from cycle_stats.json.
        stats_path = BENCH_ROOT / target / init / cond / "cycle_stats.json"
        if not stats_path.is_file():
            ax.set_title(f"{CONDITION_LABELS[cond]}\n(no data)", fontsize=10)
            ax.axis("off")
            continue

        with stats_path.open("r") as f:
            stats = json.load(f)

        # Find cycle with global best energy.
        best_cycle = None
        best_energy = float("inf")
        for cycle_key, entry in stats.items():
            if cycle_key == "0":
                continue
            e = entry.get("all_min_energy", float("inf"))
            if e < best_energy:
                best_energy = e
                best_cycle = int(cycle_key)

        if best_cycle is None:
            ax.set_title(f"{CONDITION_LABELS[cond]}\n(no cycles)", fontsize=10)
            ax.axis("off")
            continue

        # Look for PAE file.
        pae_path = (
            BENCH_ROOT / target / init / cond
            / f"cycle_{best_cycle:03d}"
            / f"sequences_cycle_all_{best_cycle}"
            / "sequence_0000.pae"
        )
        if not pae_path.is_file():
            # Try .npy variant.
            pae_path = pae_path.with_suffix(".npy")

        if pae_path.is_file():
            pae = np.load(str(pae_path))
            im = ax.imshow(pae, cmap="RdYlGn_r", vmin=0, vmax=30)
            ax.set_title(
                f"{CONDITION_LABELS[cond]}\nE={best_energy:.3f} (c{best_cycle})",
                fontsize=10,
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(
                f"{CONDITION_LABELS[cond]}\nE={best_energy:.3f} (no PAE)",
                fontsize=10,
            )
            ax.axis("off")

    fig.suptitle(f"PAE Heatmaps: {target} ({init})", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = ANALYSIS_DIR / f"pae_heatmaps_{target}_{init}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_per_term_absolute(target: str, init: str):
    """Plot absolute value of each metric vs cycle for all conditions.

    Produces one subplot per metric column found in the CSV. Each subplot
    shows the metric value from the best-energy sequence at each cycle,
    with one line per condition.
    """
    # First pass: discover all metric columns across conditions.
    all_rows_by_cond: Dict[str, List[Dict[str, str]]] = {}
    all_metric_cols: List[str] = []
    for cond in CONDITIONS:
        csv_path = BENCH_ROOT / target / init / cond / "all_sequences.csv"
        rows = load_csv(csv_path)
        all_rows_by_cond[cond] = rows
        cols = detect_metric_columns(rows)
        for c in cols:
            if c not in all_metric_cols:
                all_metric_cols.append(c)

    if not all_metric_cols:
        return

    n_metrics = len(all_metric_cols)
    n_cols = min(3, n_metrics)
    n_rows_grid = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows_grid, n_cols,
        figsize=(6 * n_cols, 4 * n_rows_grid),
        squeeze=False,
    )

    for i, metric in enumerate(all_metric_cols):
        ax = axes[i // n_cols][i % n_cols]
        for cond in CONDITIONS:
            rows = all_rows_by_cond[cond]
            cycles, vals = compute_per_cycle_metric(rows, metric, agg="best")
            if cycles:
                ax.plot(
                    cycles, vals,
                    color=CONDITION_COLORS[cond],
                    label=CONDITION_LABELS[cond],
                    linewidth=1.5, alpha=0.85,
                )
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Cycle", fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots.
    for j in range(n_metrics, n_rows_grid * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        f"Per-Term Values (best seq): {target} ({init})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    out_path = ANALYSIS_DIR / f"per_term_{target}_{init}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_per_term_mean(target: str, init: str):
    """Same as plot_per_term_absolute but shows the cycle mean instead of best."""
    all_rows_by_cond: Dict[str, List[Dict[str, str]]] = {}
    all_metric_cols: List[str] = []
    for cond in CONDITIONS:
        csv_path = BENCH_ROOT / target / init / cond / "all_sequences.csv"
        rows = load_csv(csv_path)
        all_rows_by_cond[cond] = rows
        cols = detect_metric_columns(rows)
        for c in cols:
            if c not in all_metric_cols:
                all_metric_cols.append(c)

    if not all_metric_cols:
        return

    n_metrics = len(all_metric_cols)
    n_cols = min(3, n_metrics)
    n_rows_grid = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows_grid, n_cols,
        figsize=(6 * n_cols, 4 * n_rows_grid),
        squeeze=False,
    )

    for i, metric in enumerate(all_metric_cols):
        ax = axes[i // n_cols][i % n_cols]
        for cond in CONDITIONS:
            rows = all_rows_by_cond[cond]
            cycles, vals = compute_per_cycle_metric(rows, metric, agg="mean")
            if cycles:
                ax.plot(
                    cycles, vals,
                    color=CONDITION_COLORS[cond],
                    label=CONDITION_LABELS[cond],
                    linewidth=1.5, alpha=0.85,
                )
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Cycle", fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_metrics, n_rows_grid * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        f"Per-Term Values (cycle mean): {target} ({init})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    out_path = ANALYSIS_DIR / f"per_term_mean_{target}_{init}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating convergence curves ===")
    for target in TARGETS:
        for init in INITS[target]:
            plot_convergence_curves(target, init)

    print("\n=== Generating diversity plots ===")
    for target in TARGETS:
        for init in INITS[target]:
            plot_diversity(target, init)

    print("\n=== Per-term absolute values (best seq) ===")
    for target in TARGETS:
        for init in INITS[target]:
            plot_per_term_absolute(target, init)

    print("\n=== Per-term mean values ===")
    for target in TARGETS:
        for init in INITS[target]:
            plot_per_term_mean(target, init)

    print("\n=== Summary table ===")
    generate_summary_table()

    print("\n=== Paired comparison ===")
    paired_comparison()

    print("\n=== PAE heatmaps ===")
    for target in TARGETS:
        for init in INITS[target]:
            plot_pae_heatmaps(target, init)

    print("\nDone! All outputs in", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
