"""Plot energy terms across optimization cycles for all campaigns."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_campaign(output_dir: str) -> dict:
    """Load cycle_stats.json and extract per-cycle energy term stats."""
    stats_path = os.path.join(output_dir, "cycle_stats.json")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path) as f:
        data = json.load(f)

    cycles = []
    for key in sorted(data.keys(), key=int):
        entry = data[key]
        cycle_num = entry["cycle"]

        # Collect all energy_terms from best_sequence to get term names
        term_names = list(entry["best_sequence"]["energy_terms"].keys())

        # Gather per-term values across ALL generated sequences in this cycle
        # We only have best_sequence and selected_sequences in the stats,
        # but all_avg_energy and all_min_energy give us the totals.
        # For per-term breakdown, use best_sequence for min and selected for averages.

        # Actually, let's extract from selected_sequences for richer data
        selected = entry.get("selected_sequences", [])

        term_stats = {}
        for term in term_names:
            best_val = entry["best_sequence"]["energy_terms"].get(term)
            # Compute mean/min from selected sequences
            sel_vals = [s["energy_terms"][term] for s in selected if term in s.get("energy_terms", {})]
            if sel_vals:
                term_stats[term] = {
                    "best": best_val,
                    "selected_mean": np.mean(sel_vals),
                    "selected_min": np.min(sel_vals),
                }
            else:
                term_stats[term] = {
                    "best": best_val,
                    "selected_mean": best_val,
                    "selected_min": best_val,
                }

        cycles.append({
            "cycle": cycle_num,
            "all_avg_energy": entry["all_avg_energy"],
            "all_min_energy": entry["all_min_energy"],
            "term_names": term_names,
            "term_stats": term_stats,
        })

    return cycles


def plot_campaign(cycles: list, title: str, output_path: str):
    """Plot energy terms for a single campaign."""
    if not cycles:
        return

    term_names = cycles[0]["term_names"]
    x = [c["cycle"] for c in cycles]
    n_terms = len(term_names)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    if n_terms == 1:
        # Single energy term — simple plot
        term = term_names[0]
        means = [c["term_stats"][term]["selected_mean"] for c in cycles]
        mins = [c["term_stats"][term]["best"] for c in cycles]

        ax1.plot(x, means, color=colors[0], alpha=0.7, label=f"{term} (mean)")
        ax1.plot(x, mins, color=colors[0], linewidth=2, label=f"{term} (best)")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel(term)
        ax1.legend(loc="best")

    elif n_terms == 2:
        # Two energy terms — dual y-axes
        term1, term2 = term_names[0], term_names[1]

        means1 = [c["term_stats"][term1]["selected_mean"] for c in cycles]
        mins1 = [c["term_stats"][term1]["best"] for c in cycles]
        means2 = [c["term_stats"][term2]["selected_mean"] for c in cycles]
        mins2 = [c["term_stats"][term2]["best"] for c in cycles]

        ax1.plot(x, means1, color=colors[0], alpha=0.5, linestyle="--", label=f"{term1} (mean)")
        ax1.plot(x, mins1, color=colors[0], linewidth=2, label=f"{term1} (best)")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel(term1, color=colors[0])
        ax1.tick_params(axis="y", labelcolor=colors[0])

        ax2 = ax1.twinx()
        ax2.plot(x, means2, color=colors[1], alpha=0.5, linestyle="--", label=f"{term2} (mean)")
        ax2.plot(x, mins2, color=colors[1], linewidth=2, label=f"{term2} (best)")
        ax2.set_ylabel(term2, color=colors[1])
        ax2.tick_params(axis="y", labelcolor=colors[1])

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    else:
        # 3+ terms — all on ax1
        for i, term in enumerate(term_names):
            c = colors[i % len(colors)]
            means = [c_["term_stats"][term]["selected_mean"] for c_ in cycles]
            mins = [c_["term_stats"][term]["best"] for c_ in cycles]
            ax1.plot(x, means, color=c, alpha=0.5, linestyle="--", label=f"{term} (mean)")
            ax1.plot(x, mins, color=c, linewidth=2, label=f"{term} (best)")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Energy")
        ax1.legend(loc="best", fontsize=8)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Main ---
campaigns = {
    "Campaign 1: Short Helix (aggressive)": "outputs/campaign1_short_helix",
    "Campaign 2: 3-Helix Bundle (memory)": "outputs/campaign2_3helix_memory",
    "Campaign 3: Ankyrin Repeat (explore)": "outputs/campaign3_ankyrin_explore",
    "Campaign 4: Hairpin (pure ipSAE)": "outputs/campaign4_hairpin_pure_ipsae",
    "Campaign 5: Nanobody (creative)": "outputs/campaign5_nanobody_creative",
    "4D5 Antibody + MPNN": "outputs/2GDZ_boltz_ipsae_mpnn_4D5_modal",
    "Hairpin + MPNN (original)": "outputs/2GDZ_boltz_ipsae_mpnn_modal",
}

print("Plotting campaign results...\n")
for title, output_dir in campaigns.items():
    cycles = load_campaign(output_dir)
    if cycles is None:
        print(f"  {title}: no data yet")
        continue
    print(f"  {title}: {len(cycles)} cycles")
    safe_name = output_dir.replace("outputs/", "").replace("/", "_")
    plot_path = f"outputs/{safe_name}_energy_plot.png"
    plot_campaign(cycles, title, plot_path)

print("\nDone!")
