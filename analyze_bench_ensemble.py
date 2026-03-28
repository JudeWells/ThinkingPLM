"""Comparative analysis of ensemble benchmark experiments across targets and strategies."""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path("outputs/bench_ensemble_v2")
OUT_DIR = Path("outputs/bench_ensemble_analysis_v2")
OUT_DIR.mkdir(exist_ok=True)

STRATEGY_ORDER = ["bandit_greedy", "bandit_thompson",  "thompson_eb8_bandit_rel", "random_greedy", "random_thompson",]
STRATEGY_COLORS = {
    "bandit_greedy": "#76ff03",
    "bandit_thompson": "#00bfff",
    "random_greedy": "#ff6b6b",
    "random_thompson": "#ffab40",
    "thompson_eb8_bandit_rel": "#ffffff",
}
STRATEGY_LABELS = {
    "bandit_greedy": "Bandit Greedy",
    "bandit_thompson": "Bandit Thompson",
    "random_greedy": "Random Greedy",
    "random_thompson": "Random Thompson",
    "thompson_eb8_bandit_rel": "Bandit Thompson EB8"
}


def load_experiment(exp_dir):
    """Load cycle_stats.json and all_sequences.csv for an experiment."""
    stats_path = exp_dir / "cycle_stats.json"
    csv_path = exp_dir / "all_sequences.csv"

    if not csv_path.exists():
        return None

    csv_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(row)

    cycle_energies = {}
    ensemble_stds = {}
    bandit_arm_counts = {}

    if stats_path.exists():
        with open(stats_path) as f:
            cycle_stats = json.load(f)
        for key, val in cycle_stats.items():
            cycle_num = int(key)
            best_seq = val.get("best_sequence", {})
            best_energy = best_seq.get("energy", None)
            if best_energy is not None:
                cycle_energies[cycle_num] = best_energy

            # Ensemble std from best sequence's energy_terms or directly
            e_std = best_seq.get("ensemble_std", None)
            if e_std is None:
                et = best_seq.get("energy_terms", {})
                e_std = et.get("ensemble_std", None)
            if e_std is not None:
                ensemble_stds[cycle_num] = e_std

            n_arms = val.get("thompson_num_arms", None)
            if n_arms is not None:
                bandit_arm_counts[cycle_num] = n_arms
    else:
        cycle_stats = {}
        for row in csv_rows:
            try:
                c = int(row.get("cycle", -1))
                e = float(row.get("total_energy", 0))
            except (ValueError, TypeError):
                continue
            if c not in cycle_energies or e < cycle_energies[c]:
                cycle_energies[c] = e

    sorted_cycles = sorted(cycle_energies.keys())
    if not sorted_cycles:
        return None

    running_best = cycle_energies[sorted_cycles[0]]
    n_improvements = 0
    improvement_cycles = []
    running_best_trace = [running_best]

    for c in sorted_cycles[1:]:
        e = cycle_energies[c]
        if e < running_best:
            n_improvements += 1
            running_best = e
            improvement_cycles.append(c)
        running_best_trace.append(running_best)

    n_swaps_accepted = sum(
        1 for v in cycle_stats.values()
        if v.get("swap_accepted") is True
    )

    ipsae_values = []
    for row in csv_rows:
        try:
            ipsae_values.append(float(row.get("ipSAE", 0)))
        except (ValueError, TypeError):
            pass

    total_cycles = max(sorted_cycles) if sorted_cycles else 0

    return {
        "cycle_energies": cycle_energies,
        "sorted_cycles": sorted_cycles,
        "running_best_trace": running_best_trace,
        "min_energy": min(cycle_energies.values()),
        "seed_energy": cycle_energies.get(sorted_cycles[0], 0),
        "n_improvements": n_improvements,
        "n_swaps_accepted": n_swaps_accepted,
        "improvement_cycles": improvement_cycles,
        "total_cycles": total_cycles,
        "min_ipsae": min(ipsae_values) if ipsae_values else None,
        "csv_rows": csv_rows,
        "ensemble_stds": ensemble_stds,
        "bandit_arm_counts": bandit_arm_counts,
    }


def discover_experiments():
    """Walk bench dir and find all target/scaffold/strategy combos."""
    experiments = {}

    for target_dir in sorted(BENCH_DIR.iterdir()):
        if not target_dir.is_dir() or target_dir.name == "test_logs":
            continue
        target_name = target_dir.name

        for scaffold_dir in sorted(target_dir.iterdir()):
            if not scaffold_dir.is_dir():
                continue
            scaffold_name = scaffold_dir.name

            for strategy_dir in sorted(scaffold_dir.iterdir()):
                if not strategy_dir.is_dir():
                    continue
                strategy_name = strategy_dir.name

                if strategy_name not in STRATEGY_ORDER:
                    continue

                data = load_experiment(strategy_dir)
                if data is None:
                    continue

                key = (target_name, scaffold_name)
                if key not in experiments:
                    experiments[key] = {}
                experiments[key][strategy_name] = data

    return experiments


def make_energy_trajectory_plot(experiments):
    """One subplot per target/scaffold showing energy trajectories for each strategy."""
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target, scaffolds in sorted(by_target.items()):
        n = len(scaffolds)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle(f"Energy Trajectories — {target}", fontsize=16, color="white", y=1.02)

        for idx, (scaffold, strategies) in enumerate(sorted(scaffolds)):
            ax = axes[idx // cols][idx % cols]
            for strat in STRATEGY_ORDER:
                if strat not in strategies:
                    continue
                d = strategies[strat]
                cycles = d["sorted_cycles"]
                trace = d["running_best_trace"]
                ax.plot(cycles, trace, color=STRATEGY_COLORS[strat],
                        label=STRATEGY_LABELS[strat], linewidth=2, alpha=0.9)
            ax.set_title(scaffold, color="white", fontsize=12)
            ax.set_xlabel("Cycle", color="white")
            ax.set_ylabel("Best Energy", color="white")
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(colors="white")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.patch.set_facecolor("black")
        for row in axes:
            for ax in row:
                ax.set_facecolor("black")
        plt.tight_layout()
        fname = OUT_DIR / f"trajectories_{target}.png"
        fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def make_bar_charts(experiments):
    """Bar chart: min energy achieved per strategy, grouped by target/scaffold."""
    all_keys = sorted(experiments.keys())
    labels = [f"{t}/{s}" for t, s in all_keys]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(labels) * 1.2), 10))
    fig.patch.set_facecolor("black")
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")

    x = np.arange(len(labels))
    width = 0.2
    present_strategies = [s for s in STRATEGY_ORDER
                          if any(s in experiments[k] for k in all_keys)]

    for i, strat in enumerate(present_strategies):
        vals = []
        for k in all_keys:
            d = experiments[k].get(strat)
            vals.append(d["min_energy"] if d else 0)
        offset = (i - len(present_strategies) / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=STRATEGY_LABELS[strat],
                color=STRATEGY_COLORS[strat], alpha=0.85)

    ax1.set_ylabel("Min Energy (lower is better)", color="white")
    ax1.set_title("Minimum Energy Achieved", color="white", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="white")
    ax1.legend(fontsize=9)
    ax1.tick_params(colors="white")
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    for i, strat in enumerate(present_strategies):
        vals = []
        for k in all_keys:
            d = experiments[k].get(strat)
            vals.append(d["n_improvements"] if d else 0)
        offset = (i - len(present_strategies) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=STRATEGY_LABELS[strat],
                color=STRATEGY_COLORS[strat], alpha=0.85)

    ax2.set_ylabel("Number of Improvements", color="white")
    ax2.set_title("Improvements Over Running Best", color="white", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="white")
    ax2.legend(fontsize=9)
    ax2.tick_params(colors="white")

    plt.tight_layout()
    fname = OUT_DIR / "bar_comparison.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def make_heatmap(experiments):
    """Heatmap of min energy: rows=target/scaffold, cols=strategy."""
    all_keys = sorted(experiments.keys())
    present_strategies = [s for s in STRATEGY_ORDER
                          if any(s in experiments[k] for k in all_keys)]

    data = []
    row_labels = []
    for k in all_keys:
        row = []
        for strat in present_strategies:
            d = experiments[k].get(strat)
            row.append(d["min_energy"] if d else np.nan)
        data.append(row)
        row_labels.append(f"{k[0]}/{k[1]}")

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(max(8, len(present_strategies) * 2.5),
                                     max(6, len(row_labels) * 0.5)))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(len(present_strategies)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in present_strategies],
                       rotation=45, ha="right", color="white")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8, color="white")

    for i in range(len(row_labels)):
        for j in range(len(present_strategies)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="black" if val < -0.2 else "white", fontsize=8)

    ax.set_title("Min Energy Heatmap (lower = better)", color="white", fontsize=14)
    plt.tight_layout()
    fname = OUT_DIR / "heatmap_min_energy.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def make_aggregate_strategy_plot(experiments):
    """Aggregate normalized trajectories per strategy across all experiments."""
    strategy_traces = defaultdict(list)

    for (target, scaffold), strategies in experiments.items():
        for strat, d in strategies.items():
            if d["total_cycles"] < 5:
                continue
            cycles = np.array(d["sorted_cycles"], dtype=float)
            trace = np.array(d["running_best_trace"])
            if cycles[-1] > 0:
                norm_cycles = cycles / cycles[-1]
                strategy_traces[strat].append((norm_cycles, trace))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for strat in STRATEGY_ORDER:
        if strat not in strategy_traces:
            continue
        traces = strategy_traces[strat]
        x_grid = np.linspace(0, 1, 100)
        interp_traces = []
        for norm_c, trace in traces:
            interp_traces.append(np.interp(x_grid, norm_c, trace))
        interp_traces = np.array(interp_traces)
        mean_trace = np.mean(interp_traces, axis=0)
        std_trace = np.std(interp_traces, axis=0)

        color = STRATEGY_COLORS[strat]
        ax.plot(x_grid, mean_trace, color=color, linewidth=2,
                label=f"{STRATEGY_LABELS[strat]} (n={len(traces)})")
        ax.fill_between(x_grid, mean_trace - std_trace, mean_trace + std_trace,
                        color=color, alpha=0.15)

    ax.set_xlabel("Normalized Cycle Progress", color="white")
    ax.set_ylabel("Best Energy", color="white")
    ax.set_title("Aggregate Energy Trajectories (mean ± std)", color="white", fontsize=14)
    ax.legend(fontsize=9)
    ax.tick_params(colors="white")
    plt.tight_layout()
    fname = OUT_DIR / "aggregate_trajectories.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def make_ensemble_uncertainty_plot(experiments):
    """Per-target subplot grid showing ensemble std (uncertainty) over cycles."""
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target, scaffolds in sorted(by_target.items()):
        n = len(scaffolds)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle(f"Ensemble Std (Uncertainty) — {target}", fontsize=16, color="white", y=1.02)

        for idx, (scaffold, strategies) in enumerate(sorted(scaffolds)):
            ax = axes[idx // cols][idx % cols]
            for strat in STRATEGY_ORDER:
                if strat not in strategies:
                    continue
                d = strategies[strat]
                estds = d.get("ensemble_stds", {})
                if not estds:
                    continue
                sorted_c = sorted(estds.keys())
                vals = [estds[c] for c in sorted_c]
                ax.plot(sorted_c, vals, color=STRATEGY_COLORS[strat],
                        label=STRATEGY_LABELS[strat], linewidth=1.5, alpha=0.8)
            ax.set_title(scaffold, color="white", fontsize=12)
            ax.set_xlabel("Cycle", color="white")
            ax.set_ylabel("Ensemble Std", color="white")
            ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(colors="white")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.patch.set_facecolor("black")
        for row in axes:
            for ax in row:
                ax.set_facecolor("black")
        plt.tight_layout()
        fname = OUT_DIR / f"ensemble_std_{target}.png"
        fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def make_bandit_arm_count_plot(experiments):
    """Per-target subplot grid showing bandit arm counts over cycles."""
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target, scaffolds in sorted(by_target.items()):
        n = len(scaffolds)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                                 squeeze=False)
        fig.suptitle(f"Bandit Arm Count — {target}", fontsize=16, color="white", y=1.02)

        for idx, (scaffold, strategies) in enumerate(sorted(scaffolds)):
            ax = axes[idx // cols][idx % cols]
            for strat in STRATEGY_ORDER:
                if strat not in strategies:
                    continue
                d = strategies[strat]
                arms = d.get("bandit_arm_counts", {})
                if not arms:
                    continue
                sorted_c = sorted(arms.keys())
                vals = [arms[c] for c in sorted_c]
                ax.plot(sorted_c, vals, color=STRATEGY_COLORS[strat],
                        label=STRATEGY_LABELS[strat], linewidth=1.5, alpha=0.8)
            ax.set_title(scaffold, color="white", fontsize=12)
            ax.set_xlabel("Cycle", color="white")
            ax.set_ylabel("Num Arms", color="white")
            ax.legend(fontsize=6, loc="upper left")
            ax.tick_params(colors="white")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.patch.set_facecolor("black")
        for row in axes:
            for ax in row:
                ax.set_facecolor("black")
        plt.tight_layout()
        fname = OUT_DIR / f"bandit_arms_{target}.png"
        fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def make_proposal_method_plots(experiments):
    """Per-target plot showing profam selection fraction over cycles for each strategy."""
    SMOOTHING_WINDOW = 15

    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target, scaffolds in sorted(by_target.items()):
        n = len(scaffolds)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)
        fig.suptitle(f"ProFam Selection Rate — {target}", fontsize=16, color="white", y=1.02)

        for idx, (scaffold, strategies) in enumerate(sorted(scaffolds)):
            ax = axes[idx // cols][idx % cols]
            for strat in STRATEGY_ORDER:
                if strat not in strategies:
                    continue
                csv_rows = strategies[strat].get("csv_rows", [])
                if not csv_rows:
                    continue

                cycle_methods = {}
                for row in csv_rows:
                    try:
                        c = int(row.get("cycle", -1))
                    except (ValueError, TypeError):
                        continue
                    pm = row.get("proposal_method", "")
                    cycle_methods[c] = pm

                sorted_c = sorted(cycle_methods.keys())
                if len(sorted_c) < 3:
                    continue

                is_profam = np.array([1.0 if cycle_methods[c] == "profam" else 0.0
                                      for c in sorted_c])
                cycles_arr = np.array(sorted_c)

                w = min(SMOOTHING_WINDOW, len(sorted_c))
                kernel = np.ones(w) / w
                smoothed = np.convolve(is_profam, kernel, mode="same")[:len(sorted_c)]
                ax.plot(cycles_arr, smoothed, color=STRATEGY_COLORS[strat], linewidth=2,
                        label=STRATEGY_LABELS[strat], alpha=0.9)

            ax.set_title(scaffold, color="white", fontsize=12)
            ax.set_xlabel("Cycle", color="white")
            ax.set_ylabel("ProFam Fraction", color="white")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
            ax.tick_params(colors="white")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.patch.set_facecolor("black")
        for row in axes:
            for ax in row:
                ax.set_facecolor("black")
        plt.tight_layout()
        fname = OUT_DIR / f"proposal_method_{target}.png"
        fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


def make_greedy_vs_thompson_scatter(experiments):
    """Scatter plot comparing greedy vs thompson min energy for bandit and random."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("black")

    for ax, (greedy_key, thompson_key, title) in zip(
        [ax1, ax2],
        [("bandit_greedy", "bandit_thompson", "Bandit: Greedy vs Thompson"),
         ("random_greedy", "random_thompson", "Random: Greedy vs Thompson")],
    ):
        ax.set_facecolor("black")
        greedy_vals, thompson_vals, labels = [], [], []
        for (target, scaffold), strategies in sorted(experiments.items()):
            if greedy_key in strategies and thompson_key in strategies:
                greedy_vals.append(strategies[greedy_key]["min_energy"])
                thompson_vals.append(strategies[thompson_key]["min_energy"])
                labels.append(f"{target[:4]}/{scaffold[:4]}")

        if not greedy_vals:
            continue

        greedy_vals = np.array(greedy_vals)
        thompson_vals = np.array(thompson_vals)

        ax.scatter(greedy_vals, thompson_vals, c="#00bfff", s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (greedy_vals[i], thompson_vals[i]), fontsize=6,
                        color="white", alpha=0.7, xytext=(4, 4), textcoords="offset points")

        lim_min = min(greedy_vals.min(), thompson_vals.min()) * 1.1
        lim_max = max(greedy_vals.max(), thompson_vals.max()) * 0.9
        if lim_max > lim_min:
            lim_min, lim_max = lim_max, lim_min
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", linewidth=1, alpha=0.5)

        ax.set_xlabel("Greedy Min Energy", color="white")
        ax.set_ylabel("Thompson Min Energy", color="white")
        ax.set_title(title, color="white", fontsize=13)
        ax.tick_params(colors="white")

        n_thompson_wins = np.sum(thompson_vals < greedy_vals)
        n_greedy_wins = np.sum(greedy_vals < thompson_vals)
        n_tie = np.sum(greedy_vals == thompson_vals)
        ax.text(0.05, 0.95, f"Thompson wins: {n_thompson_wins}\nGreedy wins: {n_greedy_wins}\nTie: {n_tie}",
                transform=ax.transAxes, fontsize=9, color="white", va="top",
                bbox=dict(boxstyle="round", facecolor="black", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    fname = OUT_DIR / "greedy_vs_thompson_scatter.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def make_bandit_vs_random_scatter(experiments):
    """Scatter plot comparing bandit vs random min energy for greedy and thompson."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("black")

    for ax, (bandit_key, random_key, title) in zip(
        [ax1, ax2],
        [("bandit_greedy", "random_greedy", "Greedy: Bandit vs Random"),
         ("bandit_thompson", "random_thompson", "Thompson: Bandit vs Random")],
    ):
        ax.set_facecolor("black")
        bandit_vals, random_vals, labels = [], [], []
        for (target, scaffold), strategies in sorted(experiments.items()):
            if bandit_key in strategies and random_key in strategies:
                bandit_vals.append(strategies[bandit_key]["min_energy"])
                random_vals.append(strategies[random_key]["min_energy"])
                labels.append(f"{target[:4]}/{scaffold[:4]}")

        if not bandit_vals:
            continue

        bandit_vals = np.array(bandit_vals)
        random_vals = np.array(random_vals)

        ax.scatter(bandit_vals, random_vals, c="#ff6b6b", s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (bandit_vals[i], random_vals[i]), fontsize=6,
                        color="white", alpha=0.7, xytext=(4, 4), textcoords="offset points")

        lim_min = min(bandit_vals.min(), random_vals.min()) * 1.1
        lim_max = max(bandit_vals.max(), random_vals.max()) * 0.9
        if lim_max > lim_min:
            lim_min, lim_max = lim_max, lim_min
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", linewidth=1, alpha=0.5)

        ax.set_xlabel("Bandit Min Energy", color="white")
        ax.set_ylabel("Random Min Energy", color="white")
        ax.set_title(title, color="white", fontsize=13)
        ax.tick_params(colors="white")

        n_bandit_wins = np.sum(bandit_vals < random_vals)
        n_random_wins = np.sum(random_vals < bandit_vals)
        n_tie = np.sum(bandit_vals == random_vals)
        ax.text(0.05, 0.95, f"Bandit wins: {n_bandit_wins}\nRandom wins: {n_random_wins}\nTie: {n_tie}",
                transform=ax.transAxes, fontsize=9, color="white", va="top",
                bbox=dict(boxstyle="round", facecolor="black", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    fname = OUT_DIR / "bandit_vs_random_scatter.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def generate_markdown(experiments):
    """Generate the analysis report as markdown."""
    lines = []
    lines.append("# Ensemble Benchmark Analysis: Strategy Comparison\n")
    lines.append("Comparative analysis of ensemble-scored protein design strategies across targets and scaffolds.\n")
    lines.append("## Strategies\n")
    lines.append("| Strategy | Description |")
    lines.append("|----------|-------------|")
    lines.append("| **Bandit Greedy** | ProFam/random bandit with greedy (argmax) arm selection |")
    lines.append("| **Bandit Thompson** | ProFam/random bandit with Thompson sampling arm selection |")
    lines.append("| **Random Greedy** | Random mutations with greedy (argmax) selection |")
    lines.append("| **Random Thompson** | Random mutations with Thompson sampling selection |")
    lines.append("| **Bandit Thompson EB8** | ProFam/random bandit with Thompson sampling, ensemble budget=8 |")
    lines.append("")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Target | Scaffold | Strategy | Cycles | Seed Energy | Min Energy | Improvements | Improvement Rate |")
    lines.append("|--------|----------|----------|--------|-------------|------------|--------------|------------------|")

    wins = defaultdict(int)
    all_keys = sorted(experiments.keys())

    for (target, scaffold) in all_keys:
        strategies = experiments[(target, scaffold)]
        best_strat = min(strategies.keys(), key=lambda s: strategies[s]["min_energy"])
        wins[best_strat] += 1

        for strat in STRATEGY_ORDER:
            if strat not in strategies:
                continue
            d = strategies[strat]
            rate = f"{d['n_improvements']/d['total_cycles']:.1%}" if d["total_cycles"] > 0 else "N/A"
            marker = " **BEST**" if strat == best_strat else ""
            lines.append(
                f"| {target} | {scaffold} | {STRATEGY_LABELS[strat]} | "
                f"{d['total_cycles']} | {d['seed_energy']:.4f} | {d['min_energy']:.4f}{marker} | "
                f"{d['n_improvements']} | {rate} |"
            )

    lines.append("\n## Strategy Win Count (Lowest Min Energy)\n")
    lines.append("| Strategy | Wins |")
    lines.append("|----------|------|")
    for strat in STRATEGY_ORDER:
        if strat in wins:
            lines.append(f"| {STRATEGY_LABELS[strat]} | {wins[strat]} |")

    total = sum(wins.values())
    lines.append(f"| **Total experiments** | **{total}** |")

    # Aggregate stats
    lines.append("\n## Aggregate Statistics\n")
    strategy_min_energies = defaultdict(list)
    strategy_improvements = defaultdict(list)
    strategy_rates = defaultdict(list)
    strategy_energy_gain = defaultdict(list)

    for (target, scaffold), strategies in experiments.items():
        for strat, d in strategies.items():
            strategy_min_energies[strat].append(d["min_energy"])
            strategy_improvements[strat].append(d["n_improvements"])
            if d["total_cycles"] > 0:
                strategy_rates[strat].append(d["n_improvements"] / d["total_cycles"])
            strategy_energy_gain[strat].append(d["seed_energy"] - d["min_energy"])

    lines.append("| Strategy | N | Mean Min Energy | Median Min Energy | Mean Improvements | Mean Rate | Mean Energy Gain |")
    lines.append("|----------|---|-----------------|-------------------|-------------------|-----------|------------------|")
    for strat in STRATEGY_ORDER:
        if strat not in strategy_min_energies:
            continue
        me = strategy_min_energies[strat]
        imp = strategy_improvements[strat]
        rates = strategy_rates[strat]
        gains = strategy_energy_gain[strat]
        lines.append(
            f"| {STRATEGY_LABELS[strat]} | {len(me)} | {np.mean(me):.4f} | {np.median(me):.4f} | "
            f"{np.mean(imp):.1f} | {np.mean(rates):.1%} | {np.mean(gains):.4f} |"
        )

    # Head-to-head: greedy vs thompson
    lines.append("\n## Head-to-Head: Greedy vs Thompson\n")
    for prefix, label in [("bandit", "Bandit"), ("random", "Random")]:
        greedy_key = f"{prefix}_greedy"
        thompson_key = f"{prefix}_thompson"
        greedy_wins, thompson_wins, ties = 0, 0, 0
        for (target, scaffold), strategies in experiments.items():
            if greedy_key in strategies and thompson_key in strategies:
                ge = strategies[greedy_key]["min_energy"]
                te = strategies[thompson_key]["min_energy"]
                if ge < te:
                    greedy_wins += 1
                elif te < ge:
                    thompson_wins += 1
                else:
                    ties += 1
        lines.append(f"**{label}**: Greedy wins {greedy_wins}, Thompson wins {thompson_wins}, Ties {ties}\n")

    # Head-to-head: bandit vs random
    lines.append("\n## Head-to-Head: Bandit vs Random\n")
    for suffix, label in [("greedy", "Greedy"), ("thompson", "Thompson")]:
        bandit_key = f"bandit_{suffix}"
        random_key = f"random_{suffix}"
        bandit_wins, random_wins, ties = 0, 0, 0
        for (target, scaffold), strategies in experiments.items():
            if bandit_key in strategies and random_key in strategies:
                be = strategies[bandit_key]["min_energy"]
                re = strategies[random_key]["min_energy"]
                if be < re:
                    bandit_wins += 1
                elif re < be:
                    random_wins += 1
                else:
                    ties += 1
        lines.append(f"**{label}**: Bandit wins {bandit_wins}, Random wins {random_wins}, Ties {ties}\n")

    # Per-target breakdown
    lines.append("\n## Per-Target Analysis\n")
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target in sorted(by_target.keys()):
        scaffolds = by_target[target]
        lines.append(f"### {target}\n")
        lines.append(f"Scaffolds tested: {', '.join(s for s, _ in sorted(scaffolds))}\n")

        target_best = {}
        for scaffold, strategies in scaffolds:
            for strat, d in strategies.items():
                if strat not in target_best or d["min_energy"] < target_best[strat]["min_energy"]:
                    target_best[strat] = {"min_energy": d["min_energy"], "scaffold": scaffold, **d}

        lines.append("| Strategy | Best Scaffold | Min Energy | Improvements | Cycles |")
        lines.append("|----------|---------------|------------|--------------|--------|")
        for strat in STRATEGY_ORDER:
            if strat not in target_best:
                continue
            tb = target_best[strat]
            lines.append(
                f"| {STRATEGY_LABELS[strat]} | {tb['scaffold']} | "
                f"{tb['min_energy']:.4f} | {tb['n_improvements']} | {tb['total_cycles']} |"
            )

        lines.append(f"\n![Trajectories](trajectories_{target}.png)\n")
        lines.append(f"![Ensemble Std](ensemble_std_{target}.png)\n")
        lines.append(f"![Bandit Arms](bandit_arms_{target}.png)\n")
        lines.append(f"![Proposal Method](proposal_method_{target}.png)\n")

    # Figures
    lines.append("## Comparative Plots\n")
    lines.append("### Min Energy and Improvements Bar Chart\n")
    lines.append("![Bar Comparison](bar_comparison.png)\n")
    lines.append("### Min Energy Heatmap\n")
    lines.append("![Heatmap](heatmap_min_energy.png)\n")
    lines.append("### Aggregate Trajectories\n")
    lines.append("![Aggregate](aggregate_trajectories.png)\n")
    lines.append("### Greedy vs Thompson Scatter\n")
    lines.append("![Greedy vs Thompson](greedy_vs_thompson_scatter.png)\n")
    lines.append("### Bandit vs Random Scatter\n")
    lines.append("![Bandit vs Random](bandit_vs_random_scatter.png)\n")

    # Key findings
    lines.append("\n## Key Findings\n")
    finding_num = 1

    if wins:
        best_overall = max(wins.keys(), key=lambda s: wins[s])
        lines.append(f"{finding_num}. **{STRATEGY_LABELS[best_overall]}** achieves the lowest energy "
                     f"in {wins[best_overall]}/{total} experiments "
                     f"({wins[best_overall]/total:.0%}).\n")
        finding_num += 1

    ranked = sorted(
        [(s, np.mean(strategy_min_energies[s])) for s in strategy_min_energies],
        key=lambda x: x[1]
    )
    ranking_str = " > ".join(
        f"**{STRATEGY_LABELS[s]}** ({v:.4f})" for s, v in ranked
    )
    lines.append(f"{finding_num}. Mean min energy ranking: {ranking_str}.\n")
    finding_num += 1

    rate_strs = []
    for strat in STRATEGY_ORDER:
        if strat in strategy_rates:
            rate_strs.append(f"{STRATEGY_LABELS[strat]} {np.mean(strategy_rates[strat]):.1%}")
    if rate_strs:
        lines.append(f"{finding_num}. Improvement rates: {', '.join(rate_strs)}.\n")
        finding_num += 1

    gain_strs = []
    for strat in STRATEGY_ORDER:
        if strat in strategy_energy_gain:
            gain_strs.append(f"{STRATEGY_LABELS[strat]} ({np.mean(strategy_energy_gain[strat]):.4f})")
    if gain_strs:
        lines.append(f"{finding_num}. Mean energy gain from seed: {', '.join(gain_strs)}.\n")
        finding_num += 1

    md_text = "\n".join(lines)
    md_path = OUT_DIR / "benchmark_analysis.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Saved {md_path}")


def main():
    plt.style.use("dark_background")
    print("Discovering experiments...")
    experiments = discover_experiments()
    print(f"Found {len(experiments)} target/scaffold combinations")

    for (target, scaffold), strategies in sorted(experiments.items()):
        strats = ", ".join(strategies.keys())
        print(f"  {target}/{scaffold}: {strats}")

    print("\nGenerating plots...")
    make_energy_trajectory_plot(experiments)
    make_bar_charts(experiments)
    make_heatmap(experiments)
    make_aggregate_strategy_plot(experiments)
    make_ensemble_uncertainty_plot(experiments)
    make_bandit_arm_count_plot(experiments)
    make_proposal_method_plots(experiments)
    make_greedy_vs_thompson_scatter(experiments)
    make_bandit_vs_random_scatter(experiments)
    print("\nGenerating markdown report...")
    generate_markdown(experiments)
    print("\nDone! Results in", OUT_DIR)


if __name__ == "__main__":
    main()
