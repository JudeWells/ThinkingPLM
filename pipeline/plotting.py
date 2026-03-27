"""Plotting functions for the ProFam + BAGEL pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def make_energy_summary_plot(
    log_path: Path,
    output_dir: Path,
) -> None:
    """Produce a PNG plot of average and minimum energy as a function of cycle index."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.style.use("dark_background")
    except ImportError:
        print("matplotlib not available, skipping summary plot.")
        return

    if not log_path.is_file():
        print(f"No cycle log found at {log_path}, skipping summary plot.")
        return

    with log_path.open("r") as f:
        log_data = json.load(f)

    if not log_data:
        print("Cycle log is empty, nothing to plot.")
        return

    cycles = sorted(int(k) for k in log_data.keys())
    avg = [log_data[str(c)].get("all_avg_energy", log_data[str(c)].get("avg_energy")) for c in cycles]
    min_e = [log_data[str(c)].get("all_min_energy", log_data[str(c)].get("min_energy")) for c in cycles]

    cum_min = []
    running_min = float("inf")
    for e in min_e:
        if e is not None and e < running_min:
            running_min = e
        cum_min.append(running_min if running_min != float("inf") else e)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, avg, marker="o", color="#00bfff", label="Average energy (all generated)")
    ax.plot(cycles, min_e, marker="s", color="#00e676", label="Minimum energy (all generated)")
    ax.plot(cycles, cum_min, linestyle="--", color="#ff6b6b", linewidth=1.5, label="Global best (cumulative min)")

    rejected_cycles = []
    rejected_energies = []
    for c in cycles:
        entry = log_data[str(c)]
        if entry.get("swap_accepted") is False:
            rejected_cycles.append(c)
            rejected_energies.append(entry.get("all_min_energy", entry.get("min_energy")))
    if rejected_cycles:
        ax.scatter(rejected_cycles, rejected_energies, marker="x", color="#ff6b6b",
                   s=100, zorder=5, label="Swap rejected")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Energy")
    ax.set_title("Energy & similarity trajectory over cycles")
    ax.grid(True, linestyle="--", alpha=0.4)

    sim_original = [log_data[str(c)].get("all_avg_similarity") for c in cycles]
    sim_prompt = [log_data[str(c)].get("all_avg_similarity_to_prompt") for c in cycles]
    has_sim_original = any(s is not None for s in sim_original)
    has_sim_prompt = any(s is not None for s in sim_prompt)
    if has_sim_original or has_sim_prompt:
        ax2 = ax.twinx()
        if has_sim_original:
            ax2.plot(
                cycles,
                [s if s is not None else float("nan") for s in sim_original],
                marker="^",
                linestyle="--",
                color="#ffab40",
                label="Similarity to original",
            )
        if has_sim_prompt:
            ax2.plot(
                cycles,
                [s if s is not None else float("nan") for s in sim_prompt],
                marker="v",
                linestyle="--",
                color="#e040fb",
                label="Similarity to prompt",
            )
        ax2.set_ylabel("Sequence similarity")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize="small")
    else:
        ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "energy_summary.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)
