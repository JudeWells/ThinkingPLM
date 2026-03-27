"""Comparative analysis of benchmark experiments across targets and strategies."""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path("outputs/bench")
OUT_DIR = Path("outputs/bench_analysis")
OUT_DIR.mkdir(exist_ok=True)

# Strategy display order and colors
STRATEGY_ORDER = ["profam_update", "profam_frozen", "random_update", "random_frozen",
                  "proposal_bandit", "proposal_bandit_eb5", "proposal_bandit_eb5_d1",
                  "proposal_bandit_eb10_d1", "greedy_proposal_bandit", "greedy_diverse"]
STRATEGY_COLORS = {
    "profam_update": "#ffffff",
    "profam_frozen": "#00bfff",
    "random_update": "#ff6b6b",
    "random_frozen": "#ffab40",
    "proposal_bandit": "#e040fb",
    "proposal_bandit_eb5": "#ff80ab",
    "proposal_bandit_eb5_d1": "#ffd740",
    "proposal_bandit_eb10_d1": "#18ffff",
    "greedy_proposal_bandit": "#76ff03",
    "greedy_diverse": "#ff1493",  # deep pink
}
STRATEGY_LABELS = {
    "profam_update": "ProFam Update",
    "profam_frozen": "ProFam Frozen",
    "random_update": "Random Update",
    "random_frozen": "Random Frozen",
    "proposal_bandit": "Bandit EB=2",
    "proposal_bandit_eb5": "Bandit EB=5 d=0.95",
    "proposal_bandit_eb5_d1": "Bandit EB=5 d=1.0",
    "proposal_bandit_eb10_d1": "Bandit EB=10 d=1.0",
    "greedy_proposal_bandit": "Greedy Bandit",
    "greedy_diverse": "Greedy Diverse K=10",
}


def load_experiment(exp_dir):
    """Load cycle_stats.json and all_sequences.csv for an experiment."""
    stats_path = exp_dir / "cycle_stats.json"
    csv_path = exp_dir / "all_sequences.csv"

    if not stats_path.exists() or not csv_path.exists():
        return None

    with open(stats_path) as f:
        cycle_stats = json.load(f)

    # Build per-cycle best energy from cycle_stats (includes cycle 0)
    cycle_energies = {}
    for key, val in cycle_stats.items():
        cycle_num = int(key)
        best_energy = val.get("best_sequence", {}).get("energy", None)
        if best_energy is not None:
            cycle_energies[cycle_num] = best_energy

    # Also read CSV for additional metrics
    csv_rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(row)

    # Count improvements: track running best, count when best improves
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

    # Count swap accepts from cycle_stats
    n_swaps_accepted = sum(
        1 for v in cycle_stats.values()
        if v.get("swap_accepted") is True
    )

    # Gather ipSAE and other metrics from CSV
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
    }


def discover_experiments():
    """Walk bench dir and find all target/scaffold/strategy combos."""
    experiments = {}  # (target, scaffold) -> {strategy: data}

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

                # Only include known strategies
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
    # Group by target for multi-panel figures
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target, scaffolds in sorted(by_target.items()):
        n = len(scaffolds)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows),
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
                ax.plot(cycles, trace, color=STRATEGY_COLORS.get(strat, "white"),
                        label=STRATEGY_LABELS.get(strat, strat), linewidth=2, alpha=0.9)
            ax.set_title(scaffold, color="white", fontsize=12)
            ax.set_xlabel("Cycle", color="white")
            ax.set_ylabel("Best Energy", color="white")
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(colors="white")

        # Hide unused subplots
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.style.use("dark_background")
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
    width = 0.18
    present_strategies = []
    for strat in STRATEGY_ORDER:
        if any(strat in experiments[k] for k in all_keys):
            present_strategies.append(strat)

    # Plot 1: Min energy
    for i, strat in enumerate(present_strategies):
        vals = []
        for k in all_keys:
            d = experiments[k].get(strat)
            vals.append(d["min_energy"] if d else 0)
        offset = (i - len(present_strategies) / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, "white"), alpha=0.85)

    ax1.set_ylabel("Min Energy (lower is better)", color="white")
    ax1.set_title("Minimum Energy Achieved", color="white", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="white")
    ax1.legend(fontsize=9)
    ax1.tick_params(colors="white")
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Plot 2: Number of improvements
    for i, strat in enumerate(present_strategies):
        vals = []
        for k in all_keys:
            d = experiments[k].get(strat)
            vals.append(d["n_improvements"] if d else 0)
        offset = (i - len(present_strategies) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, "white"), alpha=0.85)

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
    fig, ax = plt.subplots(figsize=(max(8, len(present_strategies) * 2),
                                     max(6, len(row_labels) * 0.4)))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Use reversed colormap so lower (better) energy is darker green
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xticks(range(len(present_strategies)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in present_strategies],
                       rotation=45, ha="right", color="white")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8, color="white")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(present_strategies)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="black" if val < -0.2 else "white", fontsize=7)

    ax.set_title("Min Energy Heatmap (lower = better)", color="white", fontsize=14)
    plt.tight_layout()
    fname = OUT_DIR / "heatmap_min_energy.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def make_improvement_rate_plot(experiments):
    """For each strategy, plot improvement rate (improvements / total_cycles) across experiments."""
    # Only use long-run experiments (>5 cycles) for meaningful rates
    strategy_rates = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        for strat, d in strategies.items():
            if d["total_cycles"] > 5:
                rate = d["n_improvements"] / d["total_cycles"]
                strategy_rates[strat].append(rate)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    strats = [s for s in STRATEGY_ORDER if s in strategy_rates]
    positions = range(len(strats))
    for i, strat in enumerate(strats):
        rates = strategy_rates[strat]
        ax.boxplot([rates], positions=[i], widths=0.5,
                   boxprops=dict(color=STRATEGY_COLORS.get(strat, "white")),
                   whiskerprops=dict(color=STRATEGY_COLORS.get(strat, "white")),
                   capprops=dict(color=STRATEGY_COLORS.get(strat, "white")),
                   medianprops=dict(color="white"),
                   flierprops=dict(markeredgecolor=STRATEGY_COLORS.get(strat, "white")))

    ax.set_xticks(list(positions))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strats], rotation=30, ha="right", color="white")
    ax.set_ylabel("Improvement Rate (improvements / cycles)", color="white")
    ax.set_title("Improvement Rate Distribution by Strategy", color="white", fontsize=14)
    ax.tick_params(colors="white")
    plt.tight_layout()
    fname = OUT_DIR / "improvement_rate_boxplot.png"
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
            # Normalize cycles to [0, 1]
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
        # Interpolate all traces to common x grid
        x_grid = np.linspace(0, 1, 100)
        interp_traces = []
        for norm_c, trace in traces:
            interp_traces.append(np.interp(x_grid, norm_c, trace))
        interp_traces = np.array(interp_traces)
        mean_trace = np.mean(interp_traces, axis=0)
        std_trace = np.std(interp_traces, axis=0)

        color = STRATEGY_COLORS.get(strat, "white")
        ax.plot(x_grid, mean_trace, color=color, linewidth=2,
                label=f"{STRATEGY_LABELS.get(strat, strat)} (n={len(traces)})")
        ax.fill_between(x_grid, mean_trace - std_trace, mean_trace + std_trace,
                        color=color, alpha=0.15)

    ax.set_xlabel("Normalized Cycle Progress", color="white")
    ax.set_ylabel("Best Energy", color="white")
    ax.set_title("Aggregate Energy Trajectories (mean +/- std)", color="white", fontsize=14)
    ax.legend(fontsize=9)
    ax.tick_params(colors="white")
    plt.tight_layout()
    fname = OUT_DIR / "aggregate_trajectories.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def generate_markdown(experiments):
    """Generate the analysis report as markdown."""
    lines = []
    lines.append("# Benchmark Analysis: Strategy Comparison\n")
    lines.append("Comparative analysis of protein design strategies across targets and scaffolds.\n")
    lines.append("## Strategies\n")
    lines.append("| Strategy | Description |")
    lines.append("|----------|-------------|")
    lines.append("| **ProFam Update** | ProFam generates sequences; injection set updated each cycle |")
    lines.append("| **ProFam Frozen** | ProFam generates sequences; injection set frozen (no update from best) |")
    lines.append("| **Random Update** | Random mutations; injection set updated each cycle |")
    lines.append("| **Random Frozen** | Random mutations; injection set frozen |")
    lines.append("| **Bandit EB=2** | Thompson bandit, exploit_bias=2, discount=0.95 |")
    lines.append("| **Bandit EB=5 d=0.95** | Thompson bandit, exploit_bias=5, discount=0.95 |")
    lines.append("| **Bandit EB=5 d=1.0** | Thompson bandit, exploit_bias=5, discount=1.0 (no decay) |")
    lines.append("| **Bandit EB=10 d=1.0** | Thompson bandit, exploit_bias=10, discount=1.0 (no decay) |")
    lines.append("| **Greedy Bandit** | Greedy proposal bandit (exploits best-known proposals) |")
    lines.append("| **Greedy Diverse K=10** | Greedy bandit with diverse arm pruning (max 10 arms, 95% identity threshold) |")
    lines.append("")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Target | Scaffold | Strategy | Cycles | Seed Energy | Min Energy | Improvements | Improvement Rate |")
    lines.append("|--------|----------|----------|--------|-------------|------------|--------------|------------------|")

    # Track strategy wins
    wins = defaultdict(int)  # strategy -> count of being best
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
                f"| {target} | {scaffold} | {STRATEGY_LABELS.get(strat, strat)} | "
                f"{d['total_cycles']} | {d['seed_energy']:.4f} | {d['min_energy']:.4f}{marker} | "
                f"{d['n_improvements']} | {rate} |"
            )

    # Win count
    lines.append("\n## Strategy Win Count (Lowest Min Energy)\n")
    lines.append("| Strategy | Wins |")
    lines.append("|----------|------|")
    for strat in STRATEGY_ORDER:
        if strat in wins:
            lines.append(f"| {STRATEGY_LABELS.get(strat, strat)} | {wins[strat]} |")

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

    lines.append("| Strategy | Mean Min Energy | Median Min Energy | Mean Improvements | Mean Rate | Mean Energy Gain |")
    lines.append("|----------|-----------------|-------------------|-------------------|-----------|------------------|")
    for strat in STRATEGY_ORDER:
        if strat not in strategy_min_energies:
            continue
        me = strategy_min_energies[strat]
        imp = strategy_improvements[strat]
        rates = strategy_rates[strat]
        gains = strategy_energy_gain[strat]
        lines.append(
            f"| {STRATEGY_LABELS.get(strat, strat)} | {np.mean(me):.4f} | {np.median(me):.4f} | "
            f"{np.mean(imp):.1f} | {np.mean(rates):.1%} | {np.mean(gains):.4f} |"
        )

    # Per-target breakdown
    lines.append("\n## Per-Target Analysis\n")
    by_target = defaultdict(list)
    for (target, scaffold), strategies in experiments.items():
        by_target[target].append((scaffold, strategies))

    for target in sorted(by_target.keys()):
        scaffolds = by_target[target]
        lines.append(f"### {target}\n")
        lines.append(f"Scaffolds tested: {', '.join(s for s, _ in sorted(scaffolds))}\n")

        # Best result per strategy across scaffolds for this target
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
                f"| {STRATEGY_LABELS.get(strat, strat)} | {tb['scaffold']} | "
                f"{tb['min_energy']:.4f} | {tb['n_improvements']} | {tb['total_cycles']} |"
            )

        lines.append(f"\n![Trajectories]({f'trajectories_{target}.png'})\n")

    # Figures
    lines.append("## Comparative Plots\n")
    lines.append("### Min Energy and Improvements Bar Chart\n")
    lines.append("![Bar Comparison](bar_comparison.png)\n")
    lines.append("### Min Energy Heatmap\n")
    lines.append("![Heatmap](heatmap_min_energy.png)\n")
    lines.append("### Aggregate Trajectories\n")
    lines.append("![Aggregate](aggregate_trajectories.png)\n")

    # Long-run only stats (>=10 cycles) — more meaningful
    lines.append("\n## Long-Run Experiments Only (>=10 cycles)\n")
    lines.append("Short 2-cycle experiments have high variance. Focusing on the 100-cycle runs:\n")

    long_strategy_min = defaultdict(list)
    long_strategy_imp = defaultdict(list)
    long_strategy_rate = defaultdict(list)
    long_wins = defaultdict(int)
    long_total = 0

    for (target, scaffold) in all_keys:
        strategies = experiments[(target, scaffold)]
        # Filter to long runs
        long_strats = {s: d for s, d in strategies.items() if d["total_cycles"] >= 10}
        if not long_strats:
            continue
        long_total += 1
        best_strat = min(long_strats.keys(), key=lambda s: long_strats[s]["min_energy"])
        long_wins[best_strat] += 1
        for strat, d in long_strats.items():
            long_strategy_min[strat].append(d["min_energy"])
            long_strategy_imp[strat].append(d["n_improvements"])
            long_strategy_rate[strat].append(d["n_improvements"] / d["total_cycles"])

    lines.append("| Strategy | N | Mean Min Energy | Median Min Energy | Mean Improvements | Mean Rate | Wins |")
    lines.append("|----------|---|-----------------|-------------------|-------------------|-----------|------|")
    for strat in STRATEGY_ORDER:
        if strat not in long_strategy_min:
            continue
        me = long_strategy_min[strat]
        imp = long_strategy_imp[strat]
        rates = long_strategy_rate[strat]
        w = long_wins.get(strat, 0)
        lines.append(
            f"| {STRATEGY_LABELS.get(strat, strat)} | {len(me)} | {np.mean(me):.4f} | {np.median(me):.4f} | "
            f"{np.mean(imp):.1f} | {np.mean(rates):.1%} | {w}/{long_total} |"
        )

    # Key findings
    lines.append("\n## Key Findings\n")

    finding_num = 1

    # Determine overall best strategy (long runs)
    if long_wins:
        best_overall = max(long_wins.keys(), key=lambda s: long_wins[s])
        lines.append(f"{finding_num}. **{STRATEGY_LABELS.get(best_overall, best_overall)}** achieves the lowest energy "
                     f"in {long_wins[best_overall]}/{long_total} long-run experiments "
                     f"({long_wins[best_overall]/long_total:.0%}).\n")
        finding_num += 1

    # Ranking of strategies by mean min energy
    ranked = sorted(
        [(s, np.mean(long_strategy_min[s])) for s in long_strategy_min],
        key=lambda x: x[1]
    )
    ranking_str = " > ".join(
        f"**{STRATEGY_LABELS.get(s, s)}** ({v:.4f})" for s, v in ranked
    )
    lines.append(f"{finding_num}. Mean min energy ranking: {ranking_str}.\n")
    finding_num += 1

    # Update vs frozen (long runs)
    profam_long = long_strategy_min.get("profam_update", [])
    profam_frozen_long = long_strategy_min.get("profam_frozen", [])
    if profam_long and profam_frozen_long:
        lines.append(f"{finding_num}. **Update vs Frozen** (ProFam): Frozen (mean {np.mean(profam_frozen_long):.4f}) "
                     f"slightly outperforms Update (mean {np.mean(profam_long):.4f}).\n")
        finding_num += 1

    # Proposal Bandit specific
    bandit_long = long_strategy_min.get("proposal_bandit", [])
    if bandit_long:
        bandit_wins = long_wins.get("proposal_bandit", 0)
        lines.append(f"{finding_num}. **Proposal Bandit** (n={len(bandit_long)}): mean min energy {np.mean(bandit_long):.4f}, "
                     f"wins {bandit_wins}/{long_total} experiments. "
                     f"Avg {np.mean(long_strategy_imp.get('proposal_bandit', [0])):.1f} improvements per run.\n")
        finding_num += 1

    # Improvement rates comparison
    rate_strs = []
    for strat in ["random_update", "proposal_bandit", "profam_update", "profam_frozen"]:
        if strat in long_strategy_rate:
            rate_strs.append(f"{STRATEGY_LABELS.get(strat)} {np.mean(long_strategy_rate[strat]):.1%}")
    if rate_strs:
        lines.append(f"{finding_num}. Improvement rates: {', '.join(rate_strs)}.\n")
        finding_num += 1

    # Energy gain from seed
    gain_strs = []
    for strat in STRATEGY_ORDER:
        if strat in long_strategy_min:
            gains = [long_strategy_min[strat][i] - 0 for i in range(len(long_strategy_min[strat]))]
            # Use actual energy_gain from aggregate stats
            eg = strategy_energy_gain.get(strat, [])
            if eg:
                gain_strs.append(f"{STRATEGY_LABELS.get(strat)} ({np.mean(eg):.4f})")
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
    # Skip improvement rate plot due to bug fix needed
    print("\nGenerating markdown report...")
    generate_markdown(experiments)
    print("\nDone! Results in", OUT_DIR)


if __name__ == "__main__":
    main()
