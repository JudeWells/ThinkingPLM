"""Comparative analysis of ensemble benchmark experiments across targets and strategies."""

import json
import csv
import os
import base64
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


def embed_image_base64(img_path):
    """Read an image file and return a base64 data URI."""
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_html(experiments):
    """Generate the analysis report as a self-contained HTML page with embedded images."""

    # Collect statistics for the report
    wins = defaultdict(int)
    all_keys = sorted(experiments.keys())

    for (target, scaffold) in all_keys:
        strategies = experiments[(target, scaffold)]
        best_strat = min(strategies.keys(), key=lambda s: strategies[s]["min_energy"])
        wins[best_strat] += 1

    total = sum(wins.values())

    # Aggregate stats
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

    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    # Build HTML
    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ensemble Benchmark Analysis</title>
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent-color: #00bfff;
            --accent-green: #00e676;
            --accent-red: #ff6b6b;
            --accent-orange: #ffab40;
            --border-color: #333;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            color: var(--accent-color);
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            text-shadow: 0 0 20px rgba(0, 191, 255, 0.3);
        }
        h2 {
            color: var(--accent-green);
            border-bottom: 2px solid var(--accent-green);
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 { color: var(--accent-orange); margin-top: 30px; }
        .subtitle {
            text-align: center;
            color: #aaa;
            margin-bottom: 40px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: rgba(0, 191, 255, 0.2);
            color: var(--accent-color);
            font-weight: 600;
        }
        tr:hover { background: rgba(255, 255, 255, 0.05); }
        .best { color: var(--accent-green); font-weight: bold; }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-color);
        }
        .stat-label { color: #aaa; font-size: 0.9em; }
        .img-container {
            text-align: center;
            margin: 30px 0;
        }
        .img-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        .findings {
            background: linear-gradient(135deg, rgba(0, 191, 255, 0.1), rgba(0, 230, 118, 0.1));
            border-left: 4px solid var(--accent-color);
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        .findings li { margin: 10px 0; }
        .h2h-result {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 5px;
            font-weight: bold;
        }
        .h2h-win { background: rgba(0, 230, 118, 0.3); }
        .h2h-lose { background: rgba(255, 107, 107, 0.3); }
        .h2h-tie { background: rgba(255, 171, 64, 0.3); }
        .target-section {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible:after {
            content: ' ▼';
            font-size: 0.8em;
        }
        .img-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }
    </style>
</head>
<body>
<div class="container">
''')

    html_parts.append('<h1>🧬 Ensemble Benchmark Analysis</h1>')
    html_parts.append('<p class="subtitle">Comparative analysis of ensemble-scored protein design strategies across targets and scaffolds</p>')

    # Strategy descriptions
    html_parts.append('<h2>Strategies</h2>')
    html_parts.append('<div class="card">')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Strategy</th><th>Description</th></tr>')
    html_parts.append('<tr><td><strong>Bandit Greedy</strong></td><td>ProFam/random bandit with greedy (argmax) arm selection</td></tr>')
    html_parts.append('<tr><td><strong>Bandit Thompson</strong></td><td>ProFam/random bandit with Thompson sampling arm selection</td></tr>')
    html_parts.append('<tr><td><strong>Random Greedy</strong></td><td>Random mutations with greedy (argmax) selection</td></tr>')
    html_parts.append('<tr><td><strong>Random Thompson</strong></td><td>Random mutations with Thompson sampling selection</td></tr>')
    html_parts.append('<tr><td><strong>Bandit Thompson EB8</strong></td><td>ProFam/random bandit with Thompson sampling, ensemble budget=8</td></tr>')
    html_parts.append('</table>')
    html_parts.append('</div>')

    # Win count stats cards
    html_parts.append('<h2>Strategy Win Count</h2>')
    html_parts.append('<div class="stats-grid">')
    for strat in STRATEGY_ORDER:
        if strat in wins:
            pct = wins[strat] / total * 100 if total > 0 else 0
            html_parts.append(f'''
            <div class="stat-card">
                <div class="stat-value">{wins[strat]}</div>
                <div class="stat-label">{STRATEGY_LABELS[strat]}<br>({pct:.0f}% of {total})</div>
            </div>''')
    html_parts.append('</div>')

    # Summary table
    html_parts.append('<h2>Summary Table</h2>')
    html_parts.append('<div class="card" style="overflow-x: auto;">')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Target</th><th>Scaffold</th><th>Strategy</th><th>Cycles</th><th>Seed Energy</th><th>Min Energy</th><th>Improvements</th><th>Rate</th></tr>')

    for (target, scaffold) in all_keys:
        strategies = experiments[(target, scaffold)]
        best_strat = min(strategies.keys(), key=lambda s: strategies[s]["min_energy"])

        for strat in STRATEGY_ORDER:
            if strat not in strategies:
                continue
            d = strategies[strat]
            rate = f"{d['n_improvements']/d['total_cycles']:.1%}" if d["total_cycles"] > 0 else "N/A"
            best_class = ' class="best"' if strat == best_strat else ''
            marker = " ✓" if strat == best_strat else ""
            html_parts.append(
                f'<tr><td>{target}</td><td>{scaffold}</td><td>{STRATEGY_LABELS[strat]}</td>'
                f'<td>{d["total_cycles"]}</td><td>{d["seed_energy"]:.4f}</td>'
                f'<td{best_class}>{d["min_energy"]:.4f}{marker}</td>'
                f'<td>{d["n_improvements"]}</td><td>{rate}</td></tr>'
            )

    html_parts.append('</table>')
    html_parts.append('</div>')

    # Aggregate stats table
    html_parts.append('<h2>Aggregate Statistics</h2>')
    html_parts.append('<div class="card" style="overflow-x: auto;">')
    html_parts.append('<table>')
    html_parts.append('<tr><th>Strategy</th><th>N</th><th>Mean Min Energy</th><th>Median Min Energy</th><th>Mean Improvements</th><th>Mean Rate</th><th>Mean Energy Gain</th></tr>')

    for strat in STRATEGY_ORDER:
        if strat not in strategy_min_energies:
            continue
        me = strategy_min_energies[strat]
        imp = strategy_improvements[strat]
        rates = strategy_rates[strat]
        gains = strategy_energy_gain[strat]
        html_parts.append(
            f'<tr><td>{STRATEGY_LABELS[strat]}</td><td>{len(me)}</td>'
            f'<td>{np.mean(me):.4f}</td><td>{np.median(me):.4f}</td>'
            f'<td>{np.mean(imp):.1f}</td><td>{np.mean(rates):.1%}</td><td>{np.mean(gains):.4f}</td></tr>'
        )

    html_parts.append('</table>')
    html_parts.append('</div>')

    # Head-to-head comparisons
    html_parts.append('<h2>Head-to-Head Comparisons</h2>')
    html_parts.append('<div class="card">')

    html_parts.append('<h3>Greedy vs Thompson</h3>')
    for prefix, label in [("bandit", "Bandit"), ("random", "Random")]:
        greedy_key = f"{prefix}_greedy"
        thompson_key = f"{prefix}_thompson"
        greedy_wins_count, thompson_wins_count, ties = 0, 0, 0
        for (target, scaffold), strategies in experiments.items():
            if greedy_key in strategies and thompson_key in strategies:
                ge = strategies[greedy_key]["min_energy"]
                te = strategies[thompson_key]["min_energy"]
                if ge < te:
                    greedy_wins_count += 1
                elif te < ge:
                    thompson_wins_count += 1
                else:
                    ties += 1
        html_parts.append(f'<p><strong>{label}:</strong> ')
        html_parts.append(f'<span class="h2h-result h2h-win">Greedy {greedy_wins_count}</span> ')
        html_parts.append(f'<span class="h2h-result h2h-win">Thompson {thompson_wins_count}</span> ')
        html_parts.append(f'<span class="h2h-result h2h-tie">Ties {ties}</span></p>')

    html_parts.append('<h3>Bandit vs Random</h3>')
    for suffix, label in [("greedy", "Greedy"), ("thompson", "Thompson")]:
        bandit_key = f"bandit_{suffix}"
        random_key = f"random_{suffix}"
        bandit_wins_count, random_wins_count, ties = 0, 0, 0
        for (target, scaffold), strategies in experiments.items():
            if bandit_key in strategies and random_key in strategies:
                be = strategies[bandit_key]["min_energy"]
                re = strategies[random_key]["min_energy"]
                if be < re:
                    bandit_wins_count += 1
                elif re < be:
                    random_wins_count += 1
                else:
                    ties += 1
        html_parts.append(f'<p><strong>{label}:</strong> ')
        html_parts.append(f'<span class="h2h-result h2h-win">Bandit {bandit_wins_count}</span> ')
        html_parts.append(f'<span class="h2h-result h2h-win">Random {random_wins_count}</span> ')
        html_parts.append(f'<span class="h2h-result h2h-tie">Ties {ties}</span></p>')

    html_parts.append('</div>')

    # Key findings
    html_parts.append('<h2>Key Findings</h2>')
    html_parts.append('<div class="findings"><ol>')

    if wins:
        best_overall = max(wins.keys(), key=lambda s: wins[s])
        pct = wins[best_overall] / total * 100 if total > 0 else 0
        html_parts.append(f'<li><strong>{STRATEGY_LABELS[best_overall]}</strong> achieves the lowest energy '
                         f'in {wins[best_overall]}/{total} experiments ({pct:.0f}%).</li>')

    ranked = sorted(
        [(s, np.mean(strategy_min_energies[s])) for s in strategy_min_energies],
        key=lambda x: x[1]
    )
    ranking_str = " &gt; ".join(
        f"<strong>{STRATEGY_LABELS[s]}</strong> ({v:.4f})" for s, v in ranked
    )
    html_parts.append(f'<li>Mean min energy ranking: {ranking_str}.</li>')

    rate_strs = []
    for strat in STRATEGY_ORDER:
        if strat in strategy_rates:
            rate_strs.append(f"{STRATEGY_LABELS[strat]} {np.mean(strategy_rates[strat]):.1%}")
    if rate_strs:
        html_parts.append(f'<li>Improvement rates: {", ".join(rate_strs)}.</li>')

    gain_strs = []
    for strat in STRATEGY_ORDER:
        if strat in strategy_energy_gain:
            gain_strs.append(f"{STRATEGY_LABELS[strat]} ({np.mean(strategy_energy_gain[strat]):.4f})")
    if gain_strs:
        html_parts.append(f'<li>Mean energy gain from seed: {", ".join(gain_strs)}.</li>')

    html_parts.append('</ol></div>')

    # Comparative plots section
    html_parts.append('<h2>Comparative Plots</h2>')

    for title, fname in [
        ("Min Energy and Improvements Bar Chart", "bar_comparison.png"),
        ("Min Energy Heatmap", "heatmap_min_energy.png"),
        ("Aggregate Trajectories", "aggregate_trajectories.png"),
        ("Greedy vs Thompson Scatter", "greedy_vs_thompson_scatter.png"),
        ("Bandit vs Random Scatter", "bandit_vs_random_scatter.png"),
    ]:
        img_path = OUT_DIR / fname
        img_data = embed_image_base64(img_path)
        if img_data:
            html_parts.append(f'<h3>{title}</h3>')
            html_parts.append(f'<div class="img-container"><img src="{img_data}" alt="{title}"></div>')

    # Per-target analysis
    html_parts.append('<h2>Per-Target Analysis</h2>')

    for target in sorted(by_target.keys()):
        scaffolds_list = by_target[target]
        html_parts.append(f'<div class="target-section">')
        html_parts.append(f'<h3>{target}</h3>')
        html_parts.append(f'<p>Scaffolds tested: {", ".join(s for s, _ in sorted(scaffolds_list))}</p>')

        target_best = {}
        for scaffold, strategies in scaffolds_list:
            for strat, d in strategies.items():
                if strat not in target_best or d["min_energy"] < target_best[strat]["min_energy"]:
                    target_best[strat] = {"min_energy": d["min_energy"], "scaffold": scaffold, **d}

        html_parts.append('<table>')
        html_parts.append('<tr><th>Strategy</th><th>Best Scaffold</th><th>Min Energy</th><th>Improvements</th><th>Cycles</th></tr>')
        for strat in STRATEGY_ORDER:
            if strat not in target_best:
                continue
            tb = target_best[strat]
            html_parts.append(
                f'<tr><td>{STRATEGY_LABELS[strat]}</td><td>{tb["scaffold"]}</td>'
                f'<td>{tb["min_energy"]:.4f}</td><td>{tb["n_improvements"]}</td><td>{tb["total_cycles"]}</td></tr>'
            )
        html_parts.append('</table>')

        # Target-specific plots
        html_parts.append('<div class="img-grid">')
        for plot_type, plot_name in [
            ("Energy Trajectories", f"trajectories_{target}.png"),
            ("Ensemble Uncertainty", f"ensemble_std_{target}.png"),
            ("Bandit Arm Count", f"bandit_arms_{target}.png"),
            ("ProFam Selection Rate", f"proposal_method_{target}.png"),
        ]:
            img_path = OUT_DIR / plot_name
            img_data = embed_image_base64(img_path)
            if img_data:
                html_parts.append(f'<div class="img-container"><img src="{img_data}" alt="{plot_type}"></div>')
        html_parts.append('</div>')
        html_parts.append('</div>')

    html_parts.append('</div></body></html>')

    html_text = "\n".join(html_parts)
    html_path = OUT_DIR / "benchmark_analysis.html"
    with open(html_path, "w") as f:
        f.write(html_text)
    print(f"Saved {html_path}")


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
    print("\nGenerating HTML report...")
    generate_html(experiments)
    print("\nDone! Results in", OUT_DIR)


if __name__ == "__main__":
    main()
