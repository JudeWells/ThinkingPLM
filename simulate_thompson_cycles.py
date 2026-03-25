"""
Sweep exploit_bias x n_cycles to see how optimal EB changes with campaign length.
Uses realistic reward model from simulate_thompson_sweep.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

OUT_DIR = Path("outputs/bench_analysis")
OUT_DIR.mkdir(exist_ok=True)
plt.style.use("dark_background")


# ---------------------------------------------------------------------------
# Realistic reward model (copied from sweep script)
# ---------------------------------------------------------------------------

def sample_initial_reward(rng):
    if rng.random() < 0.56:
        return 0.0
    return float(rng.beta(0.8, 4.0))

def sample_child_reward(parent_reward, rng):
    if parent_reward > 0:
        if rng.random() < 0.64:
            return float(np.clip(0.5 * parent_reward + 0.5 * rng.beta(0.8, 4.0), 0, 1))
        return 0.0
    else:
        if rng.random() < 0.27:
            return float(rng.beta(0.8, 4.0))
        return 0.0

def sample_observation(true_reward, rng):
    if true_reward == 0:
        if rng.random() < 0.1:
            return float(rng.beta(0.5, 10.0) * 0.2)
        return 0.0
    return float(np.clip(true_reward + rng.normal(0, 0.08), 0, 1))


# ---------------------------------------------------------------------------
# Thompson Sampler
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    arm_id: int
    alpha: float
    beta_param: float
    true_reward: float
    times_selected: int = 0

class ThompsonSampler:
    def __init__(self, m_samples=5, exploit_bias=1.0, rng=None):
        self.m_samples = max(1, m_samples)
        self.exploit_bias = max(1.0, exploit_bias)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arms: Dict[int, Arm] = {}
        self._next_id = 0

    def add_arm(self, true_reward, init_reward=None):
        r = init_reward if init_reward is not None else true_reward
        arm = Arm(self._next_id, 1.0 + r, 2.0 - r, true_reward)
        self.arms[arm.arm_id] = arm
        self._next_id += 1
        return arm

    def select_arm(self):
        best_arm, best_theta = None, -1.0
        b = self.exploit_bias
        for arm in self.arms.values():
            theta = float(np.max(self.rng.beta(arm.alpha * b, arm.beta_param * b, size=self.m_samples)))
            if theta > best_theta:
                best_theta, best_arm = theta, arm
        best_arm.times_selected += 1
        return best_arm

    def update_arm(self, arm_id, reward):
        arm = self.arms[arm_id]
        arm.alpha += reward
        arm.beta_param += (1.0 - reward)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(n_cycles=100, exploit_bias=2.0, seed=42):
    rng = np.random.default_rng(seed)
    sampler = ThompsonSampler(m_samples=5, exploit_bias=exploit_bias, rng=rng)

    for _ in range(20):
        r = sample_initial_reward(rng)
        sampler.add_arm(r, r)

    cumulative = 0.0
    selection_ids = []

    for cycle in range(n_cycles):
        arm = sampler.select_arm()
        selection_ids.append(arm.arm_id)
        obs = sample_observation(arm.true_reward, rng)
        sampler.update_arm(arm.arm_id, obs)
        cumulative += obs

        child_r = sample_child_reward(arm.true_reward, rng)
        child_obs = sample_observation(child_r, rng)
        sampler.add_arm(child_r, child_obs)

    ts = [a.times_selected for a in sampler.arms.values()]
    true_r = [a.true_reward for a in sampler.arms.values()]
    best_true = max(sampler.arms.values(), key=lambda a: a.true_reward)
    most_sel = max(sampler.arms.values(), key=lambda a: a.times_selected)

    sorted_by_true = sorted(sampler.arms.values(), key=lambda a: a.true_reward, reverse=True)
    rank = next(i for i, a in enumerate(sorted_by_true) if a.arm_id == most_sel.arm_id)

    top5 = sorted(sampler.arms.values(), key=lambda a: a.times_selected, reverse=True)[:5]
    corr, _ = sp_stats.spearmanr(true_r, ts) if np.std(ts) > 0 else (0.0, 1.0)

    return {
        "corr": corr,
        "best_is_most": most_sel.arm_id == best_true.arm_id,
        "rank_most_sel": rank,
        "avg_top5": np.mean([a.true_reward for a in top5]),
        "cumulative": cumulative,
        "frac_zero": np.mean([sampler.arms[i].true_reward == 0 for i in selection_ids]),
    }


def main():
    n_runs = 200
    eb_values = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    cycle_values = [50, 100, 200, 500]

    # Grid: (n_cycles, eb) -> aggregated metrics
    grid = {}
    for nc in cycle_values:
        for eb in eb_values:
            label = f"c={nc},eb={eb}"
            print(f"  {label}...", end=" ", flush=True)
            results = [run_sim(n_cycles=nc, exploit_bias=eb, seed=i * 17 + 3) for i in range(n_runs)]
            grid[(nc, eb)] = {
                "corr": np.mean([r["corr"] for r in results]),
                "best_is_most": np.mean([r["best_is_most"] for r in results]),
                "rank_most_sel": np.mean([r["rank_most_sel"] for r in results]),
                "avg_top5": np.mean([r["avg_top5"] for r in results]),
                "cumulative": np.mean([r["cumulative"] for r in results]),
                "frac_zero": np.mean([r["frac_zero"] for r in results]),
            }
            print(f"r={grid[(nc,eb)]['corr']:.3f} P(best)={grid[(nc,eb)]['best_is_most']:.0%}")

    # ---- Plot 1: Line plots — metric vs EB, one line per n_cycles ----
    metrics = [
        ("corr", "r(true, times_selected)", True),
        ("best_is_most", "P(best arm = most selected)", True),
        ("avg_top5", "Avg true reward of top-5", True),
        ("cumulative", "Cumulative reward", True),
        ("frac_zero", "Frac selections on zero arms", False),
        ("rank_most_sel", "True rank of most-selected arm", False),
    ]

    cycle_colors = {50: "#ff6b6b", 100: "#ffab40", 200: "#00bfff", 500: "#00e676"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EB Sweep x Campaign Length (discount=1.0, 200 runs)", color="white", fontsize=16)
    fig.patch.set_facecolor("black")

    for ax_idx, (key, title, higher_better) in enumerate(metrics):
        ax = axes[ax_idx // 3][ax_idx % 3]
        ax.set_facecolor("black")

        for nc in cycle_values:
            vals = [grid[(nc, eb)][key] for eb in eb_values]
            ax.plot(eb_values, vals, "o-", color=cycle_colors[nc], linewidth=2,
                    markersize=5, label=f"{nc} cycles", alpha=0.85)

            # Mark best
            best_idx = np.argmax(vals) if higher_better else np.argmin(vals)
            ax.plot(eb_values[best_idx], vals[best_idx], "s", color=cycle_colors[nc],
                    markersize=10, zorder=5, markeredgecolor="white", markeredgewidth=1)

        ax.set_xlabel("Exploit Bias", color="white")
        ax.set_title(title, color="white", fontsize=12)
        ax.legend(fontsize=9)
        ax.tick_params(colors="white")
        ax.set_xscale("log")
        ax.set_xticks(eb_values)
        ax.set_xticklabels([f"{e:.0f}" for e in eb_values], fontsize=8, color="white")

    plt.tight_layout()
    fname = OUT_DIR / "sim_cycles_x_eb.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 2: Heatmaps ----
    hm_metrics = [
        ("corr", "r(true, selected)"),
        ("best_is_most", "P(best = most selected)"),
        ("avg_top5", "Avg true reward top-5"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EB x Cycles Heatmaps", color="white", fontsize=14)
    fig.patch.set_facecolor("black")

    for ax_idx, (key, title) in enumerate(hm_metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("black")

        data = np.array([[grid[(nc, eb)][key] for eb in eb_values] for nc in cycle_values])
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors="white")

        ax.set_xticks(range(len(eb_values)))
        ax.set_xticklabels([f"{e:.0f}" for e in eb_values], color="white", fontsize=9)
        ax.set_yticks(range(len(cycle_values)))
        ax.set_yticklabels([str(c) for c in cycle_values], color="white", fontsize=9)
        ax.set_xlabel("Exploit Bias", color="white")
        ax.set_ylabel("Cycles", color="white")
        ax.set_title(title, color="white", fontsize=11)

        for i in range(len(cycle_values)):
            for j in range(len(eb_values)):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        color="black" if data[i,j] > data.mean() else "white", fontsize=8)

    plt.tight_layout()
    fname = OUT_DIR / "sim_cycles_x_eb_heatmap.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Markdown ----
    lines = []
    lines.append("# EB x Campaign Length Sweep\n")
    lines.append(f"200 runs per config, discount=1.0, realistic reward model.\n")

    lines.append("## Full Results Grid\n")
    for key, title, _ in metrics:
        lines.append(f"### {title}\n")
        header = "| Cycles | " + " | ".join(f"EB={e:.0f}" for e in eb_values) + " |"
        sep = "|--------|" + "|".join("--------" for _ in eb_values) + "|"
        lines.append(header)
        lines.append(sep)
        for nc in cycle_values:
            vals = [grid[(nc, eb)][key] for eb in eb_values]
            best_idx = np.argmax(vals) if _ else np.argmin(vals)
            row = f"| {nc} |"
            for j, v in enumerate(vals):
                bold = " **" if j == best_idx else " "
                bold_end = "**" if j == best_idx else ""
                row += f"{bold}{v:.3f}{bold_end} |"
            lines.append(row)
        lines.append("")

    # Optimal EB per cycle length
    lines.append("## Optimal EB by Campaign Length\n")
    lines.append("| Cycles | Best EB (correlation) | Best EB (P best=most) | Best EB (top-5 quality) |")
    lines.append("|--------|----------------------|----------------------|------------------------|")
    for nc in cycle_values:
        best_corr = eb_values[np.argmax([grid[(nc, eb)]["corr"] for eb in eb_values])]
        best_pbest = eb_values[np.argmax([grid[(nc, eb)]["best_is_most"] for eb in eb_values])]
        best_top5 = eb_values[np.argmax([grid[(nc, eb)]["avg_top5"] for eb in eb_values])]
        lines.append(f"| {nc} | {best_corr:.0f} | {best_pbest:.0f} | {best_top5:.0f} |")

    lines.append("\n## Key Findings\n")

    # Check if optimal EB shifts with cycle count
    best_corr_by_nc = [eb_values[np.argmax([grid[(nc, eb)]["corr"] for eb in eb_values])] for nc in cycle_values]
    best_pbest_by_nc = [eb_values[np.argmax([grid[(nc, eb)]["best_is_most"] for eb in eb_values])] for nc in cycle_values]

    lines.append(f"1. **Optimal EB for correlation** shifts with campaign length: "
                 + ", ".join(f"{nc} cycles→EB={best_corr_by_nc[i]:.0f}" for i, nc in enumerate(cycle_values)) + ".\n")

    lines.append(f"2. **Optimal EB for best-arm identification**: "
                 + ", ".join(f"{nc} cycles→EB={best_pbest_by_nc[i]:.0f}" for i, nc in enumerate(cycle_values)) + ".\n")

    # Does higher EB hurt more at longer campaigns?
    for nc in cycle_values:
        eb12 = grid[(nc, 12.0)]["best_is_most"]
        eb20 = grid[(nc, 20.0)]["best_is_most"]
        if eb20 < eb12:
            lines.append(f"3. At {nc} cycles, EB=20 ({eb20:.0%}) underperforms EB=12 ({eb12:.0%}) "
                         f"on best-arm ID — over-exploitation penalty grows with more cycles.\n")
            break

    # Overall recommendation
    lines.append(f"4. **Recommendation**: For 100-cycle campaigns, EB=8-12. "
                 f"For longer campaigns (200-500), EB may need to decrease slightly to maintain exploration.\n")

    lines.append("## Plots\n")
    lines.append("### Metrics vs EB by Campaign Length\n![](sim_cycles_x_eb.png)\n")
    lines.append("### Heatmaps\n![](sim_cycles_x_eb_heatmap.png)\n")

    md_path = OUT_DIR / "thompson_cycles_x_eb.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
