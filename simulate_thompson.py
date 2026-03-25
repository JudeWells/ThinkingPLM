"""
Simulate Thompson sampling campaigns with synthetic bandit arms.

Reproduces the real ThompsonSampler logic to study:
1. Does discount=0.95 cause flat (uniform) arm selection?
2. Does discount=1.0 fix it?
3. Do posteriors converge to true parameters?
4. How often is the true best arm discovered?
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

OUT_DIR = Path("outputs/bench_analysis")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("dark_background")

# ---------------------------------------------------------------------------
# Copy of the real ThompsonSampler (simplified — no sequence fields)
# ---------------------------------------------------------------------------

@dataclass
class ThompsonArm:
    arm_id: int
    alpha: float
    beta_param: float
    true_mean: float          # hidden ground truth
    true_std: float           # hidden ground truth
    times_selected: int = 0
    total_reward_credited: float = 0.0


class ThompsonSampler:
    """Exact copy of the real sampler logic, with optional eb_on_update mode.

    eb_on_update=False (default / current):
        select: sample Beta(alpha * EB, beta * EB)  — EB concentrates sampling
        update: alpha += reward, beta += (1 - reward)

    eb_on_update=True (proposed):
        select: sample Beta(alpha, beta)             — standard sampling
        update: alpha += reward * EB, beta += (1 - reward) * EB  — EB amplifies evidence
    """

    def __init__(self, m_samples=1, exploit_bias=1.0, rng=None, eb_on_update=False):
        self.m_samples = max(1, m_samples)
        self.exploit_bias = max(1.0, exploit_bias)
        self.eb_on_update = eb_on_update
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arms: Dict[int, ThompsonArm] = {}
        self._next_arm_id = 0

    def add_arm(self, true_mean, true_std, init_reward=0.0):
        arm = ThompsonArm(
            arm_id=self._next_arm_id,
            alpha=1.0 + init_reward,
            beta_param=2.0 - init_reward,
            true_mean=true_mean,
            true_std=true_std,
        )
        self.arms[arm.arm_id] = arm
        self._next_arm_id += 1
        return arm

    def select_arm(self):
        best_arm = None
        best_theta = -1.0
        # EB on select only when NOT in eb_on_update mode
        b = self.exploit_bias if not self.eb_on_update else 1.0
        for arm in self.arms.values():
            thetas = self.rng.beta(arm.alpha * b, arm.beta_param * b, size=self.m_samples)
            theta = float(np.max(thetas))
            if theta > best_theta:
                best_theta = theta
                best_arm = arm
        best_arm.times_selected += 1
        return best_arm

    def update_arm(self, arm_id, reward):
        arm = self.arms[arm_id]
        # EB on update only when in eb_on_update mode
        b = self.exploit_bias if self.eb_on_update else 1.0
        arm.alpha += reward * b
        arm.beta_param += (1.0 - reward) * b
        arm.total_reward_credited += reward

    def decay_posteriors(self, discount):
        if discount >= 1.0:
            return
        for arm in self.arms.values():
            arm.alpha = 1.0 + discount * (arm.alpha - 1.0)
            arm.beta_param = 1.0 + discount * (arm.beta_param - 1.0)


def sample_reward(arm, rng):
    """Draw a noisy reward from the arm's true distribution, clipped to [0,1]."""
    return float(np.clip(rng.normal(arm.true_mean, arm.true_std), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    n_arms=20,
    n_cycles=100,
    m_samples=5,
    exploit_bias=2.0,
    discount=0.95,
    seed=42,
    add_new_arm_each_cycle=True,  # mirrors real pipeline: 1 new arm per cycle
    eb_on_update=False,
):
    """Run one campaign and return stats."""
    rng = np.random.default_rng(seed)

    # Create arms with varying true means (spread across [0, 1])
    # A few are good (high mean), most are mediocre
    true_means = np.concatenate([
        rng.uniform(0.05, 0.25, size=n_arms // 2),    # bad arms
        rng.uniform(0.25, 0.50, size=n_arms // 4),    # mediocre
        rng.uniform(0.50, 0.80, size=n_arms // 4),    # good arms
    ])
    rng.shuffle(true_means)
    true_stds = rng.uniform(0.05, 0.15, size=n_arms)

    sampler = ThompsonSampler(m_samples=m_samples, exploit_bias=exploit_bias, rng=rng,
                              eb_on_update=eb_on_update)

    # Add initial arms (like cycle 0 seed)
    for i in range(n_arms):
        init_reward = float(np.clip(rng.normal(true_means[i], true_stds[i]), 0, 1))
        sampler.add_arm(true_means[i], true_stds[i], init_reward)

    best_true_arm_id = max(sampler.arms.values(), key=lambda a: a.true_mean).arm_id

    selection_history = []  # which arm was selected each cycle
    best_found_history = []  # was the true best ever selected?

    for cycle in range(n_cycles):
        # Select
        arm = sampler.select_arm()
        selection_history.append(arm.arm_id)

        # Observe reward
        reward = sample_reward(arm, rng)
        sampler.update_arm(arm.arm_id, reward)

        # Optionally add a new arm (like the pipeline does)
        if add_new_arm_each_cycle and cycle < n_cycles - 1:
            new_mean = float(rng.uniform(0.05, 0.50))  # mostly mediocre
            new_std = float(rng.uniform(0.05, 0.15))
            new_init = float(np.clip(rng.normal(new_mean, new_std), 0, 1))
            new_arm = sampler.add_arm(new_mean, new_std, new_init)
            # Update best true arm if this new one is better
            if new_mean > sampler.arms[best_true_arm_id].true_mean:
                best_true_arm_id = new_arm.arm_id

        # Decay
        sampler.decay_posteriors(discount)

        best_found_history.append(best_true_arm_id in selection_history)

    # Final stats
    total_arms = len(sampler.arms)
    ts_counts = [a.times_selected for a in sampler.arms.values()]
    true_means_final = [a.true_mean for a in sampler.arms.values()]
    expected_rewards = [a.alpha / (a.alpha + a.beta_param) for a in sampler.arms.values()]

    # How concentrated is selection?
    frac_ge3 = sum(1 for t in ts_counts if t >= 3) / total_arms
    max_selected = max(ts_counts)
    gini = _gini(ts_counts)

    # Correlation between true_mean and times_selected
    corr, p = sp_stats.spearmanr(true_means_final, ts_counts)

    # Correlation between true_mean and estimated E[theta]
    corr_est, p_est = sp_stats.spearmanr(true_means_final, expected_rewards)

    # Was the best arm the most selected?
    most_selected_arm = max(sampler.arms.values(), key=lambda a: a.times_selected)
    best_is_most_selected = most_selected_arm.arm_id == best_true_arm_id

    # Was best arm ever selected?
    best_ever_found = best_true_arm_id in selection_history

    # Top-5 by times_selected: what are their true ranks?
    arms_by_ts = sorted(sampler.arms.values(), key=lambda a: a.times_selected, reverse=True)
    arms_by_true = sorted(sampler.arms.values(), key=lambda a: a.true_mean, reverse=True)
    true_rank_of_best_selected = next(
        i for i, a in enumerate(arms_by_true) if a.arm_id == most_selected_arm.arm_id
    )

    return {
        "total_arms": total_arms,
        "frac_ge3": frac_ge3,
        "max_selected": max_selected,
        "gini": gini,
        "corr_true_vs_selected": corr,
        "p_corr": p,
        "corr_true_vs_estimated": corr_est,
        "best_is_most_selected": best_is_most_selected,
        "best_ever_found": best_ever_found,
        "true_rank_of_most_selected": true_rank_of_best_selected,
        "selection_history": selection_history,
        "arms": sampler.arms,
        "best_true_arm_id": best_true_arm_id,
    }


def _gini(values):
    """Gini coefficient: 0=perfectly equal, 1=maximally concentrated."""
    v = np.array(values, dtype=float)
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def run_batch(n_runs=50, **kwargs):
    """Run many simulations and aggregate stats."""
    results = []
    for i in range(n_runs):
        r = run_simulation(seed=i * 17 + 3, **kwargs)
        results.append(r)
    return results


def main():
    n_runs = 100
    configs = {
        # Current production settings
        "d=0.95 EB=2 select (current)":  dict(discount=0.95, exploit_bias=2.0, eb_on_update=False),
        "d=0.95 EB=5 select":            dict(discount=0.95, exploit_bias=5.0, eb_on_update=False),
        # EB on update (proposed) — with decay
        "d=0.95 EB=2 update":            dict(discount=0.95, exploit_bias=2.0, eb_on_update=True),
        "d=0.95 EB=5 update":            dict(discount=0.95, exploit_bias=5.0, eb_on_update=True),
        # No decay baselines
        "d=1.0 EB=2 select":             dict(discount=1.0, exploit_bias=2.0, eb_on_update=False),
        "d=1.0 EB=5 select":             dict(discount=1.0, exploit_bias=5.0, eb_on_update=False),
        # EB on update — no decay
        "d=1.0 EB=2 update":             dict(discount=1.0, exploit_bias=2.0, eb_on_update=True),
        "d=1.0 EB=5 update":             dict(discount=1.0, exploit_bias=5.0, eb_on_update=True),
        # Vanilla baseline
        "d=1.0 EB=1 vanilla":            dict(discount=1.0, exploit_bias=1.0, eb_on_update=False),
    }

    colors = {
        "d=0.95 EB=2 select (current)":  "#ff6b6b",
        "d=0.95 EB=5 select":            "#ffab40",
        "d=0.95 EB=2 update":            "#ff6b6b",
        "d=0.95 EB=5 update":            "#ffab40",
        "d=1.0 EB=2 select":             "#00bfff",
        "d=1.0 EB=5 select":             "#00e676",
        "d=1.0 EB=2 update":             "#00bfff",
        "d=1.0 EB=5 update":             "#00e676",
        "d=1.0 EB=1 vanilla":            "#bb86fc",
    }

    all_results = {}
    for label, kw in configs.items():
        print(f"Running {label} ({n_runs} runs)...")
        all_results[label] = run_batch(n_runs=n_runs, n_cycles=100, m_samples=5,
                                        add_new_arm_each_cycle=True, **kw)

    # ---- Table: aggregate stats ----
    agg = {}
    for label, results in all_results.items():
        agg[label] = {
            "gini": np.mean([r["gini"] for r in results]),
            "frac_ge3": np.mean([r["frac_ge3"] for r in results]),
            "max_sel": np.mean([r["max_selected"] for r in results]),
            "corr_true_sel": np.mean([r["corr_true_vs_selected"] for r in results]),
            "corr_true_est": np.mean([r["corr_true_vs_estimated"] for r in results]),
            "best_most_sel": np.mean([r["best_is_most_selected"] for r in results]),
            "best_ever_found": np.mean([r["best_ever_found"] for r in results]),
            "true_rank_most_sel": np.mean([r["true_rank_of_most_selected"] for r in results]),
        }

    # ---- Plot 1: Selection distribution (histogram of times_selected for one run) ----
    fig, axes = plt.subplots(1, len(configs), figsize=(4.5 * len(configs), 4), squeeze=False)
    fig.suptitle("Distribution of times_selected (single run, seed=3)", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[0][idx]
        ax.set_facecolor("black")
        r = results[0]  # first run
        ts = [a.times_selected for a in r["arms"].values()]
        max_ts = max(ts)
        bins = np.arange(0, max_ts + 2) - 0.5
        ax.hist(ts, bins=bins, color=colors[label], alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Times Selected", color="white", fontsize=9)
        ax.set_ylabel("Count", color="white", fontsize=9)
        ax.set_title(label, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        # Add gini annotation
        ax.text(0.95, 0.95, f"Gini={r['gini']:.2f}", transform=ax.transAxes,
                ha="right", va="top", color="white", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    plt.tight_layout()
    fname = OUT_DIR / "sim_selection_distribution.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 2: True mean vs times_selected (scatter, one run each) ----
    fig, axes = plt.subplots(1, len(configs), figsize=(4.5 * len(configs), 4), squeeze=False)
    fig.suptitle("True Mean vs Times Selected (single run)", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[0][idx]
        ax.set_facecolor("black")
        r = results[0]
        true_m = [a.true_mean for a in r["arms"].values()]
        ts = [a.times_selected for a in r["arms"].values()]
        ax.scatter(true_m, ts, c=colors[label], alpha=0.5, s=15, edgecolors="none")
        ax.set_xlabel("True Mean Reward", color="white", fontsize=9)
        ax.set_ylabel("Times Selected", color="white", fontsize=9)
        ax.set_title(f"{label}\nr={r['corr_true_vs_selected']:.3f}", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
    plt.tight_layout()
    fname = OUT_DIR / "sim_true_vs_selected.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 3: True mean vs estimated E[theta] (posterior convergence) ----
    fig, axes = plt.subplots(1, len(configs), figsize=(4.5 * len(configs), 4), squeeze=False)
    fig.suptitle("True Mean vs Estimated E[θ] (posterior convergence)", color="white", fontsize=14)
    fig.patch.set_facecolor("black")
    for idx, (label, results) in enumerate(all_results.items()):
        ax = axes[0][idx]
        ax.set_facecolor("black")
        r = results[0]
        true_m = [a.true_mean for a in r["arms"].values()]
        est = [a.alpha / (a.alpha + a.beta_param) for a in r["arms"].values()]
        ax.scatter(true_m, est, c=colors[label], alpha=0.5, s=15, edgecolors="none")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.5)
        ax.set_xlabel("True Mean Reward", color="white", fontsize=9)
        ax.set_ylabel("Estimated E[θ]", color="white", fontsize=9)
        ax.set_title(f"{label}\nr={r['corr_true_vs_estimated']:.3f}", color="white", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors="white", labelsize=8)
    plt.tight_layout()
    fname = OUT_DIR / "sim_posterior_convergence.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 4: Aggregate bar chart across n_runs ----
    metrics = [
        ("gini", "Selection Concentration\n(Gini, higher=more concentrated)"),
        ("corr_true_sel", "Spearman r\n(true mean vs times_selected)"),
        ("corr_true_est", "Spearman r\n(true mean vs estimated E[θ])"),
        ("best_most_sel", "P(best arm = most selected)"),
        ("best_ever_found", "P(best arm ever selected)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 5))
    fig.suptitle(f"Aggregate Statistics ({n_runs} runs per config)", color="white", fontsize=14)
    fig.patch.set_facecolor("black")

    x = np.arange(len(configs))
    labels_short = [l.replace(" (current)", "").replace(" (no decay)", "\n(no decay)").replace(" (vanilla)", "\n(vanilla)")
                    for l in configs.keys()]

    for ax_idx, (metric_key, metric_title) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("black")
        vals = [agg[label][metric_key] for label in configs.keys()]
        bar_colors = [colors[label] for label in configs.keys()]
        ax.bar(x, vals, color=bar_colors, alpha=0.85, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=6.5, rotation=45, ha="right", color="white")
        ax.set_title(metric_title, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    fname = OUT_DIR / "sim_aggregate_bars.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Plot 5: Posterior alpha/beta trajectories for top arm over cycles ----
    # Re-run a single campaign with detailed tracking
    fig, axes = plt.subplots(2, len(configs), figsize=(4.5 * len(configs), 7), squeeze=False)
    fig.suptitle("Posterior Evolution of Best True Arm Over Cycles", color="white", fontsize=14)
    fig.patch.set_facecolor("black")

    for idx, (label, kw) in enumerate(configs.items()):
        rng = np.random.default_rng(3)
        # Reproduce same arms as run_simulation with seed=3
        n_arms = 20
        true_means = np.concatenate([
            rng.uniform(0.05, 0.25, size=n_arms // 2),
            rng.uniform(0.25, 0.50, size=n_arms // 4),
            rng.uniform(0.50, 0.80, size=n_arms // 4),
        ])
        rng.shuffle(true_means)
        true_stds = rng.uniform(0.05, 0.15, size=n_arms)

        sampler = ThompsonSampler(m_samples=5, exploit_bias=kw["exploit_bias"], rng=rng,
                                  eb_on_update=kw.get("eb_on_update", False))
        for i in range(n_arms):
            init_r = float(np.clip(rng.normal(true_means[i], true_stds[i]), 0, 1))
            sampler.add_arm(true_means[i], true_stds[i], init_r)

        best_id = max(sampler.arms.values(), key=lambda a: a.true_mean).arm_id
        alpha_trace = [sampler.arms[best_id].alpha]
        beta_trace = [sampler.arms[best_id].beta_param]
        expected_trace = [sampler.arms[best_id].alpha /
                          (sampler.arms[best_id].alpha + sampler.arms[best_id].beta_param)]

        for cycle in range(100):
            arm = sampler.select_arm()
            reward = sample_reward(arm, rng)
            sampler.update_arm(arm.arm_id, reward)
            if cycle < 99:
                nm = float(rng.uniform(0.05, 0.50))
                ns = float(rng.uniform(0.05, 0.15))
                nr = float(np.clip(rng.normal(nm, ns), 0, 1))
                sampler.add_arm(nm, ns, nr)
            sampler.decay_posteriors(kw["discount"])
            alpha_trace.append(sampler.arms[best_id].alpha)
            beta_trace.append(sampler.arms[best_id].beta_param)
            expected_trace.append(sampler.arms[best_id].alpha /
                                  (sampler.arms[best_id].alpha + sampler.arms[best_id].beta_param))

        ax_top = axes[0][idx]
        ax_bot = axes[1][idx]
        ax_top.set_facecolor("black")
        ax_bot.set_facecolor("black")

        cycles = range(len(alpha_trace))
        ax_top.plot(cycles, alpha_trace, color="#00e676", label="α", linewidth=1.5)
        ax_top.plot(cycles, beta_trace, color="#ff6b6b", label="β", linewidth=1.5)
        ax_top.set_title(label, color="white", fontsize=9)
        ax_top.set_ylabel("α / β", color="white", fontsize=9)
        ax_top.legend(fontsize=8)
        ax_top.tick_params(colors="white", labelsize=7)

        true_m = sampler.arms[best_id].true_mean
        ax_bot.plot(cycles, expected_trace, color=colors[label], linewidth=1.5, label="E[θ]")
        ax_bot.axhline(true_m, color="white", linestyle="--", linewidth=0.8, label=f"True mean={true_m:.2f}")
        ax_bot.set_xlabel("Cycle", color="white", fontsize=9)
        ax_bot.set_ylabel("E[θ]", color="white", fontsize=9)
        ax_bot.legend(fontsize=8)
        ax_bot.tick_params(colors="white", labelsize=7)

    plt.tight_layout()
    fname = OUT_DIR / "sim_posterior_evolution.png"
    fig.savefig(fname, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- Generate markdown report ----
    lines = []
    lines.append("# Thompson Sampling Simulation: Discount & Exploit Bias Effects\n")
    lines.append(f"Synthetic bandit simulation ({n_runs} runs per config, 100 cycles, 20 initial arms + 1 new arm/cycle = ~120 total arms).\n")
    lines.append("## Setup\n")
    lines.append("- Arms have hidden true mean rewards drawn from a mixture: 50% bad (0.05-0.25), 25% mediocre (0.25-0.50), 25% good (0.50-0.80)")
    lines.append("- Each cycle: select arm via Thompson sampling, observe noisy reward, update posterior, optionally decay, add 1 new arm")
    lines.append("- This mirrors the real pipeline: ~100 cycles, growing arm pool, m_samples=5\n")

    lines.append("## Aggregate Results\n")
    lines.append("| Config | Gini | r(true,selected) | r(true,E[θ]) | P(best=most sel) | P(best found) | Avg rank of most sel |")
    lines.append("|--------|------|-------------------|--------------|-------------------|---------------|----------------------|")
    for label in configs:
        a = agg[label]
        lines.append(f"| {label} | {a['gini']:.3f} | {a['corr_true_sel']:.3f} | "
                     f"{a['corr_true_est']:.3f} | {a['best_most_sel']:.0%} | "
                     f"{a['best_ever_found']:.0%} | {a['true_rank_most_sel']:.1f} |")

    lines.append("\n## Key Observations\n")

    # Helper to safely get agg values
    def g(label, metric):
        return agg[label][metric]

    # 1. Decay effect
    lines.append(f"1. **Discount=0.95 causes near-uniform selection** (Gini={g('d=0.95 EB=2 select (current)', 'gini'):.3f}) "
                 f"vs discount=1.0 (Gini={g('d=1.0 EB=2 select', 'gini'):.3f}). "
                 f"Decay erases posterior differences faster than they accumulate.\n")

    # 2. Quality tracking
    lines.append(f"2. **Removing decay improves quality-tracking**: r(true mean, times_selected) jumps from "
                 f"{g('d=0.95 EB=2 select (current)', 'corr_true_sel'):.3f} to "
                 f"{g('d=1.0 EB=2 select', 'corr_true_sel'):.3f}.\n")

    # 3. EB on select with decay: no effect
    lines.append(f"3. **EB on select + decay = no effect**: EB=2 Gini={g('d=0.95 EB=2 select (current)', 'gini'):.3f} "
                 f"vs EB=5 Gini={g('d=0.95 EB=5 select', 'gini'):.3f} (nearly identical).\n")

    # 4. EB on UPDATE with decay: the key test
    lines.append(f"4. **EB on update + decay partially rescues the signal**: "
                 f"d=0.95 EB=2 update: Gini={g('d=0.95 EB=2 update', 'gini'):.3f}, "
                 f"r={g('d=0.95 EB=2 update', 'corr_true_sel'):.3f}, "
                 f"P(best)={g('d=0.95 EB=2 update', 'best_most_sel'):.0%}. "
                 f"d=0.95 EB=5 update: Gini={g('d=0.95 EB=5 update', 'gini'):.3f}, "
                 f"r={g('d=0.95 EB=5 update', 'corr_true_sel'):.3f}, "
                 f"P(best)={g('d=0.95 EB=5 update', 'best_most_sel'):.0%}. "
                 f"By amplifying the evidence per observation, EB on update counteracts the decay.\n")

    # 5. No decay comparisons
    lines.append(f"5. **Without decay, EB on select vs update**: "
                 f"d=1.0 EB=5 select: Gini={g('d=1.0 EB=5 select', 'gini'):.3f}, "
                 f"r={g('d=1.0 EB=5 select', 'corr_true_sel'):.3f}. "
                 f"d=1.0 EB=5 update: Gini={g('d=1.0 EB=5 update', 'gini'):.3f}, "
                 f"r={g('d=1.0 EB=5 update', 'corr_true_sel'):.3f}.\n")

    # 6. Best arm identification
    lines.append(f"6. **Best arm identification rates**: "
                 f"d=0.95 EB=2 select={g('d=0.95 EB=2 select (current)', 'best_most_sel'):.0%}, "
                 f"d=0.95 EB=5 update={g('d=0.95 EB=5 update', 'best_most_sel'):.0%}, "
                 f"d=1.0 EB=2 select={g('d=1.0 EB=2 select', 'best_most_sel'):.0%}, "
                 f"d=1.0 EB=5 select={g('d=1.0 EB=5 select', 'best_most_sel'):.0%}, "
                 f"d=1.0 EB=5 update={g('d=1.0 EB=5 update', 'best_most_sel'):.0%}.\n")

    # 7. Posterior convergence
    lines.append(f"7. **Posterior convergence (r true vs E[θ])**: "
                 f"d=0.95 select={g('d=0.95 EB=2 select (current)', 'corr_true_est'):.3f}, "
                 f"d=0.95 update EB=5={g('d=0.95 EB=5 update', 'corr_true_est'):.3f}, "
                 f"d=1.0 select={g('d=1.0 EB=2 select', 'corr_true_est'):.3f}, "
                 f"d=1.0 update EB=5={g('d=1.0 EB=5 update', 'corr_true_est'):.3f}.\n")

    lines.append("## Recommendations\n")
    lines.append("**Option A (simplest fix):** Set `thompson_discount: 1.0`. This alone fixes the main issue — "
                 "posteriors accumulate evidence and EB-on-select works as intended.\n")
    lines.append("**Option B (if decay is needed):** Move EB from selection to update: "
                 "`alpha += reward * EB, beta += (1-reward) * EB`. This amplifies evidence per observation, "
                 "counteracting the decay. Particularly effective at higher EB values.\n")
    lines.append("**Option C (best of both):** Disable decay AND apply EB on update. "
                 "This gives the strongest quality-tracking and best arm identification.\n")

    lines.append("## Plots\n")
    lines.append("### Selection Distribution (single run)\n![](sim_selection_distribution.png)\n")
    lines.append("### True Mean vs Times Selected\n![](sim_true_vs_selected.png)\n")
    lines.append("### Posterior Convergence (True vs Estimated)\n![](sim_posterior_convergence.png)\n")
    lines.append("### Aggregate Statistics\n![](sim_aggregate_bars.png)\n")
    lines.append("### Posterior Evolution of Best Arm\n![](sim_posterior_evolution.png)\n")

    md_path = OUT_DIR / "thompson_simulation.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
