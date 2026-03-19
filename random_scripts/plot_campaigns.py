"""Plot energy terms across optimization cycles for all campaigns."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("dark_background")


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

    colors = ["#00bfff", "#ff6b6b", "#00e676", "#ffab40"]

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
    fig.savefig(output_path, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def load_campaign_raw(output_dir: str) -> dict | None:
    """Load the raw cycle_stats.json data."""
    stats_path = os.path.join(output_dir, "cycle_stats.json")
    if not os.path.exists(stats_path):
        return None
    with open(stats_path) as f:
        return json.load(f)


def plot_accepted_best(raw_data: dict, title: str, output_path: str):
    """Plot the energy of the best (last accepted) sequence over cycles.

    Shows: cycle min energy, global elite energy, and marks accepted/rejected swaps.
    """
    if not raw_data:
        return

    cycles = sorted(int(k) for k in raw_data.keys())
    cycle_min = [raw_data[str(c)].get("all_min_energy", raw_data[str(c)].get("min_energy")) for c in cycles]

    # Global elite energy (cumulative best accepted).
    elite_energy = []
    for c in cycles:
        ge = raw_data[str(c)].get("global_elite")
        if ge is not None:
            elite_energy.append(ge["energy"])
        else:
            elite_energy.append(None)

    # Accepted / rejected markers.
    accepted_x, accepted_y = [], []
    rejected_x, rejected_y = [], []
    for c, e in zip(cycles, cycle_min):
        swap = raw_data[str(c)].get("swap_accepted")
        if swap is True:
            accepted_x.append(c)
            accepted_y.append(e)
        elif swap is False:
            rejected_x.append(c)
            rejected_y.append(e)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(cycles, cycle_min, color="#00bfff", alpha=0.5, linewidth=1, label="Cycle min energy")

    if any(e is not None for e in elite_energy):
        ax.plot(cycles, elite_energy, color="#00e676", linewidth=2.5, label="Global elite (best accepted)")

    if accepted_x:
        ax.scatter(accepted_x, accepted_y, marker="o", color="#00e676", s=30, zorder=5, alpha=0.7, label="Swap accepted")
    if rejected_x:
        ax.scatter(rejected_x, rejected_y, marker="x", color="#ff6b6b", s=60, zorder=5, label="Swap rejected")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Energy")
    ax.set_title(f"{title} — accepted best")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Main ---
# campaigns = {
#     "Campaign 1: Short Helix (aggressive)": "outputs/campaign1_short_helix",
#     "Campaign 2: 3-Helix Bundle (memory)": "outputs/campaign2_3helix_memory",
#     "Campaign 3: Ankyrin Repeat (explore)": "outputs/campaign3_ankyrin_explore",
#     "Campaign 4: Hairpin (pure ipSAE)": "outputs/campaign4_hairpin_pure_ipsae",
#     "Campaign 5: Nanobody (creative)": "outputs/campaign5_nanobody_creative",
#     "Campaign 6: Hairpin (elite, ipSAE)": "outputs/campaign6_hairpin_elite",
#     "Campaign 7: RFd3 (elite, ipSAE)": "outputs/campaign7_rfd3_elite",
#     "Campaign 8: Nanobody (elite, ipSAE)": "outputs/campaign8_nanobody_elite",
#     "Campaign 9: Hairpin (elite, ipSAE+MPNN)": "outputs/campaign9_hairpin_elite_mpnn",
#     "Campaign 10: RFd3 (elite, ipSAE+MPNN)": "outputs/campaign10_rfd3_elite_mpnn",
#     "Campaign 11: Nanobody (elite, ipSAE+MPNN)": "outputs/campaign11_nanobody_elite_mpnn",
#     "Campaign 12: Tiny Barrel (elite, ipSAE)": "outputs/campaign12_tiny_barrel_elite",
#     "Campaign 13: Repebody (elite, ipSAE+size)": "outputs/campaign13_repebody_elite_size",
#     "4D5 Antibody + MPNN": "outputs/2GDZ_boltz_ipsae_mpnn_4D5_modal",
#     "Hairpin + MPNN (original)": "outputs/2GDZ_boltz_ipsae_mpnn_modal",
# }

campaigns = {
    "Campaign 16: bindcraft boltz mpnn": "outputs/campaign16_bindcraft_boltz_mpnn",
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

    # Accepted-best plot (requires swap_accepted / global_elite fields).
    raw_data = load_campaign_raw(output_dir)
    if raw_data is not None:
        first_entry = raw_data.get(next(iter(raw_data), ""), {})
        if first_entry.get("swap_accepted") is not None or first_entry.get("global_elite") is not None:
            accepted_path = f"outputs/{safe_name}_accepted_best.png"
            plot_accepted_best(raw_data, title, accepted_path)

print("\nDone!")
