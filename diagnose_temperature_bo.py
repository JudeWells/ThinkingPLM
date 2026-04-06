#!/usr/bin/env python3
"""
Diagnostic script for TemperatureBO.

Tests the BO loop against synthetic reward functions with known optima.
For each function we run the full suggest-observe loop and track:
  - Whether samples converge to the true maximum
  - Whether EI peaks match where the BO actually samples
  - GP posterior accuracy vs the true function
  - Per-step EI landscapes and chosen points

Generates per-function diagnostic plots showing:
  1. Top panel: true function, GP posterior, and all sampled points (numbered)
  2. Middle panel: EI at each step overlaid, with chosen point marked
  3. Bottom panel: regret over time (best_possible - best_observed)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path
from scipy.stats import norm as sp_norm

from pipeline.temperature_bo import TemperatureBO


# ── Synthetic reward functions ──────────────────────────────────────────

def unimodal(t):
    """Single peak at T=0.8."""
    return 0.75 * np.exp(-((t - 0.8) ** 2) / (2 * 0.15**2))


def bimodal(t):
    """Two peaks: T=0.5 (global max) and T=1.3 (local max)."""
    return 0.8 * np.exp(-((t - 0.5) ** 2) / (2 * 0.1**2)) + \
           0.6 * np.exp(-((t - 1.3) ** 2) / (2 * 0.12**2))


def plateau_with_spike(t):
    """Flat plateau 0.4-1.0 with a narrow spike at T=1.4."""
    base = 0.5 * (1.0 / (1 + np.exp(-20 * (t - 0.4)))) * (1.0 / (1 + np.exp(20 * (t - 1.0))))
    spike = 0.9 * np.exp(-((t - 1.4) ** 2) / (2 * 0.05**2))
    return base + spike


def monotonic_increasing(t):
    """Reward increases with temperature — optimum at upper bound."""
    return 0.3 + 0.5 * (t - 0.4) / (1.6 - 0.4)


def noisy_multimodal(t):
    """Three peaks of similar height — tests exploration."""
    return (0.7 * np.exp(-((t - 0.55) ** 2) / (2 * 0.08**2)) +
            0.72 * np.exp(-((t - 0.95) ** 2) / (2 * 0.08**2)) +
            0.68 * np.exp(-((t - 1.35) ** 2) / (2 * 0.08**2)))


FUNCTIONS = {
    "unimodal": unimodal,
    "bimodal": bimodal,
    "plateau_with_spike": plateau_with_spike,
    "monotonic_increasing": monotonic_increasing,
    "noisy_multimodal": noisy_multimodal,
}


# ── Regime-shift reward functions ───────────────────────────────────────
# These are callables that take (t, cycle) and return reward.
# The optimum location changes at a known switch point.

def make_regime_shift_smooth(switch_cycle: int = 25, transition_width: int = 3):
    """Optimum smoothly moves from T=0.6 to T=1.3 around switch_cycle."""
    def func(t, cycle):
        # Sigmoid blend: 0 before switch, 1 after
        blend = 1.0 / (1 + np.exp(-(cycle - switch_cycle) / max(transition_width, 0.5)))
        r1 = 0.8 * np.exp(-((t - 0.6) ** 2) / (2 * 0.1**2))
        r2 = 0.8 * np.exp(-((t - 1.3) ** 2) / (2 * 0.1**2))
        return (1 - blend) * r1 + blend * r2
    func.switch_cycle = switch_cycle
    func.opt_before = 0.6
    func.opt_after = 1.3
    return func


def make_regime_shift_abrupt(switch_cycle: int = 25):
    """Optimum abruptly jumps from T=0.6 to T=1.3 at switch_cycle."""
    def func(t, cycle):
        if cycle < switch_cycle:
            return 0.8 * np.exp(-((t - 0.6) ** 2) / (2 * 0.1**2))
        else:
            return 0.8 * np.exp(-((t - 1.3) ** 2) / (2 * 0.1**2))
    func.switch_cycle = switch_cycle
    func.opt_before = 0.6
    func.opt_after = 1.3
    return func


def make_regime_shift_swap_heights(switch_cycle: int = 25):
    """Two peaks always present; the global max swaps at switch_cycle.
    Before: T=0.6 is higher (0.8 vs 0.5). After: T=1.3 is higher (0.8 vs 0.5)."""
    def func(t, cycle):
        if cycle < switch_cycle:
            return (0.8 * np.exp(-((t - 0.6) ** 2) / (2 * 0.1**2)) +
                    0.5 * np.exp(-((t - 1.3) ** 2) / (2 * 0.1**2)))
        else:
            return (0.5 * np.exp(-((t - 0.6) ** 2) / (2 * 0.1**2)) +
                    0.8 * np.exp(-((t - 1.3) ** 2) / (2 * 0.1**2)))
    func.switch_cycle = switch_cycle
    func.opt_before = 0.6
    func.opt_after = 1.3
    return func


REGIME_SHIFT_FUNCTIONS = {
    "regime_smooth": make_regime_shift_smooth(switch_cycle=25, transition_width=3),
    "regime_abrupt": make_regime_shift_abrupt(switch_cycle=25),
    "regime_swap_heights": make_regime_shift_swap_heights(switch_cycle=25),
}


def compute_ei(gp, T_grid, best_reward, xi=0.01):
    """Compute EI at each point in T_grid given a fitted GP."""
    ei_vals = []
    for t in T_grid:
        t_arr = np.array([[t]])
        mu, sigma = gp.predict(t_arr, return_std=True)
        if sigma[0] < 1e-8:
            ei_vals.append(0.0)
        else:
            z = (mu[0] - best_reward - xi) / sigma[0]
            ei = (mu[0] - best_reward - xi) * sp_norm.cdf(z) + sigma[0] * sp_norm.pdf(z)
            ei_vals.append(max(ei, 0.0))
    return np.array(ei_vals)


def run_diagnostic(
    func_name: str,
    func,
    noise_std: float = 0.03,
    n_cycles: int = 30,
    n_random: int = 5,
    seed: int = 42,
    temp_min: float = 0.4,
    temp_max: float = 1.6,
    save_dir: str = "outputs/bo_diagnostics",
):
    """Run BO loop on a synthetic function and produce diagnostic plots."""

    rng = np.random.default_rng(seed)
    T_grid = np.linspace(temp_min, temp_max, 500)
    true_rewards = np.array([func(t) for t in T_grid])
    true_best = T_grid[np.argmax(true_rewards)]
    true_best_reward = np.max(true_rewards)

    bo = TemperatureBO(
        temp_min=temp_min,
        temp_max=temp_max,
        initial_temp=0.8,
        n_random=n_random,
        xi=0.01,
        seed=seed,
    )

    # Storage for per-step diagnostics
    sampled_temps = []
    observed_rewards = []
    true_at_sampled = []
    ei_snapshots = []       # EI landscape right before each GP-based decision
    suggested_temps = []    # what suggest() returned (before observing)
    best_so_far = []
    regret = []

    for cycle in range(n_cycles):
        # Get suggestion
        t = bo.suggest()
        sampled_temps.append(t)
        suggested_temps.append(t)

        # Noisy observation
        true_r = func(t)
        noisy_r = true_r + rng.normal(0, noise_std)
        true_at_sampled.append(true_r)
        observed_rewards.append(noisy_r)

        # Snapshot the EI landscape BEFORE observing (i.e. the EI that led to this choice)
        # For cycles >= n_random, we can compute EI from the current GP state
        # (which was fitted inside suggest())
        if cycle >= n_random:
            best_reward = max(bo.rewards)  # best before this observation
            ei = compute_ei(bo.gp, T_grid, best_reward, bo.xi)
            ei_snapshots.append((cycle, ei.copy(), t))
        else:
            ei_snapshots.append((cycle, None, t))

        # Observe
        bo.observe(t, -noisy_r)  # observe expects energy (negative reward)

        # Track convergence
        best_observed = max(observed_rewards)
        best_so_far.append(best_observed)
        regret.append(true_best_reward - max(true_at_sampled))

    # ── Generate diagnostic plots ───────────────────────────────────────

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), height_ratios=[3, 2, 1.5])
    plt.style.use("dark_background")
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    # ── Panel 1: True function, GP posterior, samples ───────────────────
    ax = axes[0]
    ax.plot(T_grid, true_rewards, color="#ff6b6b", linewidth=2, linestyle="--",
            label="True function", zorder=2)
    ax.axvline(true_best, color="#ff6b6b", linestyle=":", alpha=0.5,
               label=f"True optimum T={true_best:.2f}")

    # Final GP posterior
    if len(bo.temps) >= n_random:
        X = np.array(bo.temps).reshape(-1, 1)
        y = np.array(bo.rewards)
        bo.gp.fit(X, y)
        mu, sigma = bo.gp.predict(T_grid.reshape(-1, 1), return_std=True)
        ax.plot(T_grid, mu, color="#00bfff", linewidth=2, label="Final GP mean")
        ax.fill_between(T_grid, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.15, color="#00bfff")

    # Samples colored by order
    color_norm = Normalize(vmin=0, vmax=max(n_cycles - 1, 1))
    cmap = cm.plasma
    for i, (t, r) in enumerate(zip(sampled_temps, bo.rewards)):
        marker = "s" if i < n_random else "o"
        edgecolor = "gray" if i < n_random else "white"
        ax.scatter(t, r, c=[cmap(color_norm(i))], s=80, marker=marker,
                   edgecolors=edgecolor, linewidths=0.8, zorder=5)
        ax.annotate(str(i), (t, r), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", color="white")

    # Mark random vs GP phases
    ax.scatter([], [], marker="s", color="gray", label=f"Random phase (0-{n_random-1})")
    ax.scatter([], [], marker="o", color="gray", label=f"GP-guided ({n_random}+)")

    ax.set_ylabel("Reward (negated energy)", fontsize=11)
    ax.set_title(f"BO Diagnostic: {func_name}  |  noise_std={noise_std}  |  "
                 f"best found T={sampled_temps[np.argmax(bo.rewards)]:.3f} "
                 f"(true opt T={true_best:.3f})",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    # ── Panel 2: EI at each GP-guided step ──────────────────────────────
    ax2 = axes[1]
    gp_steps = [(c, ei, t) for c, ei, t in ei_snapshots if ei is not None]

    if gp_steps:
        ei_norm = Normalize(vmin=gp_steps[0][0], vmax=gp_steps[-1][0])
        for cycle_idx, ei, chosen_t in gp_steps:
            color = cmap(ei_norm(cycle_idx))
            ax2.plot(T_grid, ei, color=color, alpha=0.5, linewidth=1)
            # Mark where it actually sampled
            ei_at_chosen = np.interp(chosen_t, T_grid, ei)
            ax2.scatter(chosen_t, ei_at_chosen, color=color, s=50,
                        edgecolors="white", linewidths=0.5, zorder=5)
            # Mark the EI maximum
            ei_max_t = T_grid[np.argmax(ei)]
            ei_max_val = np.max(ei)
            ax2.scatter(ei_max_t, ei_max_val, color=color, s=30,
                        marker="v", edgecolors="white", linewidths=0.3, zorder=4)

        ax2.scatter([], [], marker="o", color="gray", s=50, label="Chosen point")
        ax2.scatter([], [], marker="v", color="gray", s=30, label="EI maximum")

        # Check for mismatches: how often does chosen != argmax(EI)?
        mismatches = 0
        for cycle_idx, ei, chosen_t in gp_steps:
            ei_max_t = T_grid[np.argmax(ei)]
            if abs(chosen_t - ei_max_t) > 0.05:
                mismatches += 1
        ax2.set_title(f"EI per GP step (circles=chosen, triangles=EI max) | "
                      f"mismatches: {mismatches}/{len(gp_steps)}",
                      fontsize=11)
    else:
        ax2.text(0.5, 0.5, "No GP-guided steps yet", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")

    ax2.set_ylabel("Expected Improvement", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)

    # ── Panel 3: Regret over time ───────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(range(n_cycles), regret, color="#00e676", linewidth=2, marker="o",
             markersize=4)
    ax3.axhline(0, color="white", alpha=0.3, linestyle="--")
    ax3.axvline(n_random - 0.5, color="yellow", alpha=0.4, linestyle="--",
                label="Random -> GP transition")
    ax3.set_xlabel("Cycle", fontsize=11)
    ax3.set_ylabel("Simple Regret", fontsize=11)
    ax3.set_title("Convergence: true_best - best_found_so_far (lower=better)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / f"bo_diag_{func_name}.png", dpi=150, facecolor="black",
                edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_dir / f'bo_diag_{func_name}.png'}")

    # Also generate the BO's own plot for comparison
    next_t = bo.suggest()
    bo.plot(str(out_dir / f"bo_native_{func_name}.png"), next_suggestion=next_t)
    print(f"  Saved: {out_dir / f'bo_native_{func_name}.png'}")

    # ── Summary stats ───────────────────────────────────────────────────
    best_found_t = sampled_temps[np.argmax(bo.rewards)]
    best_found_r = max(bo.rewards)
    return {
        "func": func_name,
        "true_opt_T": true_best,
        "true_opt_reward": true_best_reward,
        "best_found_T": best_found_t,
        "best_found_reward": best_found_r,
        "final_regret": regret[-1],
        "converged_by": next((i for i, r in enumerate(regret) if r < 0.01), None),
        "n_mismatches": sum(
            1 for c, ei, t in ei_snapshots
            if ei is not None and abs(t - T_grid[np.argmax(ei)]) > 0.05
        ),
    }


def run_regime_shift_diagnostic(
    func_name: str,
    func,
    noise_std: float = 0.03,
    n_cycles: int = 50,
    n_random: int = 5,
    seed: int = 42,
    temp_min: float = 0.4,
    temp_max: float = 1.6,
    save_dir: str = "outputs/bo_diagnostics",
):
    """Run BO on a time-varying reward function and measure adaptation speed.

    The reward function takes (temperature, cycle) — its optimum shifts at
    func.switch_cycle.  We measure how many cycles after the shift before
    the BO starts sampling near the new optimum.
    """
    rng = np.random.default_rng(seed)
    T_grid = np.linspace(temp_min, temp_max, 500)
    switch = func.switch_cycle
    opt_before = func.opt_before
    opt_after = func.opt_after

    bo = TemperatureBO(
        temp_min=temp_min,
        temp_max=temp_max,
        initial_temp=0.8,
        n_random=n_random,
        xi=0.01,
        seed=seed,
    )

    sampled_temps = []
    observed_rewards = []
    true_opt_per_cycle = []      # where the true optimum is each cycle
    instantaneous_regret = []    # regret vs current (not historical) optimum
    near_new_opt = []            # bool: is the sample within 0.1 of new optimum?

    for cycle in range(n_cycles):
        t = bo.suggest()
        sampled_temps.append(t)

        # Current true function and its optimum
        true_rewards_now = np.array([func(tt, cycle) for tt in T_grid])
        true_opt_t = T_grid[np.argmax(true_rewards_now)]
        true_opt_r = np.max(true_rewards_now)
        true_opt_per_cycle.append(true_opt_t)

        # Noisy observation
        true_r = func(t, cycle)
        noisy_r = true_r + rng.normal(0, noise_std)
        observed_rewards.append(noisy_r)

        # Instantaneous regret: how far is this sample from the current optimum?
        instantaneous_regret.append(true_opt_r - true_r)

        # Is the sample near the new optimum?
        near_new_opt.append(abs(t - opt_after) < 0.15)

        bo.observe(t, -noisy_r)

    # ── Adaptation metrics ──────────────────────────────────────────────
    # After the switch, how many cycles until we first sample near the new opt?
    post_switch_near = [i for i in range(switch, n_cycles) if near_new_opt[i]]
    cycles_to_first_near = (post_switch_near[0] - switch) if post_switch_near else None

    # How many cycles until we consistently sample near new opt (3 of 5)?
    cycles_to_consistent = None
    for i in range(switch, n_cycles - 4):
        window = near_new_opt[i:i+5]
        if sum(window) >= 3:
            cycles_to_consistent = i - switch
            break

    # Average instantaneous regret in windows
    pre_regret = np.mean(instantaneous_regret[n_random:switch]) if switch > n_random else float("nan")
    post_regret_early = np.mean(instantaneous_regret[switch:min(switch+10, n_cycles)])
    post_regret_late = np.mean(instantaneous_regret[max(switch+10, switch):n_cycles]) if n_cycles > switch + 10 else float("nan")

    # ── Plot ────────────────────────────────────────────────────────────
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16),
                             height_ratios=[3, 1.5, 1.5, 1.2])
    plt.style.use("dark_background")
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    # ── Panel 1: Sampled temperatures over time ─────────────────────────
    ax = axes[0]

    # Background: true reward heatmap over (cycle, temperature)
    reward_map = np.zeros((len(T_grid), n_cycles))
    for c in range(n_cycles):
        for j, tt in enumerate(T_grid):
            reward_map[j, c] = func(tt, c)
    im = ax.imshow(reward_map, aspect="auto", origin="lower",
                   extent=[0, n_cycles, temp_min, temp_max],
                   cmap="inferno", alpha=0.7)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("True reward", fontsize=9)

    # Sampled points
    color_norm = Normalize(vmin=0, vmax=n_cycles - 1)
    cmap_pts = cm.cool
    for i, (t, r) in enumerate(zip(sampled_temps, observed_rewards)):
        marker = "s" if i < n_random else "o"
        ax.scatter(i, t, c=[cmap_pts(color_norm(i))], s=60, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=5)

    # True optimum trajectory
    ax.plot(range(n_cycles), true_opt_per_cycle, color="#00e676", linewidth=2,
            linestyle="--", label="True optimum", zorder=4)
    ax.axvline(switch, color="#ff6b6b", linewidth=2, linestyle="--",
               alpha=0.8, label=f"Regime shift (cycle {switch})")

    ax.set_ylabel("Temperature", fontsize=11)
    title_adapt = (f"first near: {cycles_to_first_near}" if cycles_to_first_near is not None
                   else "first near: never")
    title_consist = (f"consistent: {cycles_to_consistent}" if cycles_to_consistent is not None
                     else "consistent: never")
    ax.set_title(f"Regime Shift: {func_name}  |  {title_adapt}  |  {title_consist}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.15)

    # ── Panel 2: True function before & after, with GP posterior ────────
    ax2 = axes[1]
    rewards_before = np.array([func(t, switch - 1) for t in T_grid])
    rewards_after = np.array([func(t, n_cycles - 1) for t in T_grid])
    ax2.plot(T_grid, rewards_before, color="#ffab40", linewidth=2,
             linestyle="--", label=f"Before shift (cycle {switch-1})")
    ax2.plot(T_grid, rewards_after, color="#00e676", linewidth=2,
             linestyle="--", label=f"After shift (cycle {n_cycles-1})")

    # Final GP posterior
    if len(bo.temps) >= n_random:
        X = np.array(bo.temps).reshape(-1, 1)
        y = np.array(bo.rewards)
        bo.gp.fit(X, y)
        mu, sigma = bo.gp.predict(T_grid.reshape(-1, 1), return_std=True)
        ax2.plot(T_grid, mu, color="#00bfff", linewidth=2, label="Final GP mean")
        ax2.fill_between(T_grid, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.15, color="#00bfff")

    ax2.set_ylabel("Reward", fontsize=10)
    ax2.set_xlabel("Temperature", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # ── Panel 3: Distance from current optimum per cycle ────────────────
    ax3 = axes[2]
    dist_from_opt = [abs(sampled_temps[i] - true_opt_per_cycle[i]) for i in range(n_cycles)]
    colors = ["#ffab40" if i < switch else "#00e676" for i in range(n_cycles)]
    ax3.bar(range(n_cycles), dist_from_opt, color=colors, alpha=0.7, width=1.0)
    ax3.axvline(switch, color="#ff6b6b", linewidth=2, linestyle="--", alpha=0.8)
    ax3.axhline(0.15, color="white", linewidth=1, linestyle=":", alpha=0.4,
                label="Near-optimum threshold (0.15)")
    ax3.set_ylabel("|T_sampled - T_opt|", fontsize=10)
    ax3.set_title("Distance from current optimum each cycle", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    # ── Panel 4: Instantaneous regret ───────────────────────────────────
    ax4 = axes[3]
    ax4.plot(range(n_cycles), instantaneous_regret, color="#00bfff",
             linewidth=2, marker="o", markersize=3)
    ax4.axvline(switch, color="#ff6b6b", linewidth=2, linestyle="--", alpha=0.8,
                label=f"Shift at cycle {switch}")
    ax4.axhline(0, color="white", alpha=0.3, linestyle="--")
    ax4.set_xlabel("Cycle", fontsize=11)
    ax4.set_ylabel("Instantaneous Regret", fontsize=10)
    ax4.set_title(f"Regret: pre={pre_regret:.4f}, post(0-10)={post_regret_early:.4f}, "
                  f"post(10+)={post_regret_late:.4f}", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / f"bo_regime_{func_name}.png", dpi=150,
                facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_dir / f'bo_regime_{func_name}.png'}")

    return {
        "func": func_name,
        "switch_cycle": switch,
        "opt_before": opt_before,
        "opt_after": opt_after,
        "cycles_to_first_near": cycles_to_first_near,
        "cycles_to_consistent": cycles_to_consistent,
        "pre_regret": pre_regret,
        "post_regret_early": post_regret_early,
        "post_regret_late": post_regret_late,
    }


def main():
    save_dir = "outputs/bo_diagnostics"
    print("=" * 70)
    print("  TemperatureBO Diagnostic Suite")
    print("=" * 70)

    results = []
    for name, func in FUNCTIONS.items():
        print(f"\nRunning: {name}")
        r = run_diagnostic(name, func, noise_std=0.03, n_cycles=30, save_dir=save_dir)
        results.append(r)

    # Also test with higher noise
    print(f"\nRunning: bimodal (high noise)")
    r = run_diagnostic("bimodal_noisy", bimodal, noise_std=0.1, n_cycles=30,
                       save_dir=save_dir)
    results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("  STATIC FUNCTIONS")
    print("=" * 70)
    print(f"{'Function':<25} {'True T':>7} {'Found T':>8} {'Regret':>8} "
          f"{'Conv@':>6} {'Mismatch':>9}")
    print("-" * 70)
    for r in results:
        conv = str(r["converged_by"]) if r["converged_by"] is not None else "never"
        print(f"{r['func']:<25} {r['true_opt_T']:>7.3f} {r['best_found_T']:>8.3f} "
              f"{r['final_regret']:>8.4f} {conv:>6} "
              f"{r['n_mismatches']:>5}/{30 - 5}")

    # ── Regime shift tests ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  REGIME SHIFT TESTS")
    print("=" * 70)

    regime_results = []
    for name, func in REGIME_SHIFT_FUNCTIONS.items():
        print(f"\nRunning: {name}")
        r = run_regime_shift_diagnostic(name, func, noise_std=0.03, n_cycles=50,
                                        save_dir=save_dir)
        regime_results.append(r)

    # Also test with more cycles post-shift
    print(f"\nRunning: regime_abrupt (long, 80 cycles)")
    r = run_regime_shift_diagnostic(
        "regime_abrupt_long",
        make_regime_shift_abrupt(switch_cycle=25),
        noise_std=0.03, n_cycles=80, save_dir=save_dir,
    )
    regime_results.append(r)

    print("\n" + "-" * 80)
    print(f"{'Function':<25} {'Switch':>6} {'1st Near':>9} {'Consistent':>11} "
          f"{'Pre Regret':>11} {'Post(0-10)':>11} {'Post(10+)':>11}")
    print("-" * 80)
    for r in regime_results:
        first = str(r["cycles_to_first_near"]) if r["cycles_to_first_near"] is not None else "never"
        consist = str(r["cycles_to_consistent"]) if r["cycles_to_consistent"] is not None else "never"
        print(f"{r['func']:<25} {r['switch_cycle']:>6} {first:>9} {consist:>11} "
              f"{r['pre_regret']:>11.4f} {r['post_regret_early']:>11.4f} "
              f"{r['post_regret_late']:>11.4f}")


if __name__ == "__main__":
    main()
