"""
Bayesian optimization of ProFam sampling temperature using a Gaussian Process.

Each cycle, the GP models the relationship between temperature and the best
energy achieved in that cycle.  It proposes the next temperature by maximising
expected improvement (EI) over the current best observation.

This replaces the simple adaptive temperature heuristic with a principled
exploration-exploitation strategy.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

logger = logging.getLogger(__name__)


class TemperatureBO:
    """Bayesian optimisation over sampling temperature.

    Parameters
    ----------
    temp_min : float
        Lower bound of temperature search range.
    temp_max : float
        Upper bound of temperature search range.
    initial_temp : float
        Temperature for the first cycle (before any observations).
    n_random : int
        Number of initial random samples before switching to EI.
    xi : float
        Exploration parameter for expected improvement.  Higher = more
        exploration of uncertain temperatures.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        temp_min: float = 0.4,
        temp_max: float = 1.6,
        initial_temp: float = 0.8,
        n_random: int = 5,
        xi: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.initial_temp = initial_temp
        self.n_random = n_random
        self.xi = xi
        self.rng = np.random.default_rng(seed)

        # Observations: (temperature, reward) pairs
        self.temps: List[float] = []
        self.rewards: List[float] = []  # higher = better (negated energy)

        # GP with Matérn kernel (smooth but flexible)
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5) + WhiteKernel(0.01)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=seed,
        )

    def _windowed_data(self) -> Tuple[List[float], List[float]]:
        """Return the most recent observations within the sliding window.

        Window size is one third of total observations, with a minimum of
        ``n_random`` so the GP always has enough points after the random phase.
        """
        n = len(self.temps)
        window = max(n // 3, self.n_random)
        return self.temps[-window:], self.rewards[-window:]

    def observe(
        self,
        temperature: float,
        best_energy: float,
        unique_fraction: float = 1.0,
    ) -> None:
        """Record an observation: temperature used and best energy achieved.

        When ``unique_fraction`` drops below 50%, the reward is scaled
        down so the GP learns to avoid temperatures that cause severe
        duplicate generation.  Above 50% there is no penalty — some
        duplicates are acceptable, especially at lower temperatures
        that tend to produce the best individual sequences.
        """
        reward = -best_energy

        if unique_fraction < 0.5:
            # Linear penalty: at 0.5 → no penalty, at 0.0 → reward zeroed
            penalty_scale = unique_fraction / 0.5  # 0.0 at 0%, 1.0 at 50%
            reward *= penalty_scale
            logger.info(
                f"TemperatureBO: penalising reward for low diversity "
                f"({unique_fraction:.0%}): {-best_energy:.4f} → {reward:.4f}"
            )

        self.temps.append(temperature)
        self.rewards.append(reward)

    def suggest(self) -> float:
        """Suggest the next temperature to try."""
        n = len(self.temps)

        # First cycle: use initial temperature
        if n == 0:
            return self.initial_temp

        # Random exploration phase
        if n < self.n_random:
            t = self.rng.uniform(self.temp_min, self.temp_max)
            logger.info(f"TemperatureBO: random sample T={t:.3f} (phase {n}/{self.n_random})")
            return float(t)

        # Fit GP to windowed observations (recent history only)
        w_temps, w_rewards = self._windowed_data()
        X = np.array(w_temps).reshape(-1, 1)
        y = np.array(w_rewards)
        self.gp.fit(X, y)

        # Maximise expected improvement
        best_reward = max(w_rewards)

        def neg_ei(t):
            t_arr = np.array([[t]])
            mu, sigma = self.gp.predict(t_arr, return_std=True)
            if sigma < 1e-8:
                return 0.0
            z = (mu[0] - best_reward - self.xi) / sigma[0]
            ei = (mu[0] - best_reward - self.xi) * norm.cdf(z) + sigma[0] * norm.pdf(z)
            return -ei  # minimize negative EI

        # Dense grid search + local refinement around each grid peak
        grid = np.linspace(self.temp_min, self.temp_max, 200)
        grid_ei = [-neg_ei(t) for t in grid]
        best_grid_idx = int(np.argmax(grid_ei))
        best_grid_val = grid_ei[best_grid_idx]

        # Refine around the best grid point with bounded local search
        grid_step = grid[1] - grid[0]
        refine_lo = max(self.temp_min, grid[best_grid_idx] - 3 * grid_step)
        refine_hi = min(self.temp_max, grid[best_grid_idx] + 3 * grid_step)
        result = minimize_scalar(
            neg_ei,
            bounds=(refine_lo, refine_hi),
            method="bounded",
        )
        # Use refined result only if it's actually better than the grid
        if result.success and result.fun < -best_grid_val:
            suggested = result.x
        else:
            suggested = grid[best_grid_idx]
        suggested = float(np.clip(suggested, self.temp_min, self.temp_max))

        # Log GP state
        mu_at_suggested, sigma_at_suggested = self.gp.predict(
            np.array([[suggested]]), return_std=True
        )
        logger.info(
            f"TemperatureBO: suggesting T={suggested:.3f} "
            f"(GP μ={mu_at_suggested[0]:.4f}, σ={sigma_at_suggested[0]:.4f}, "
            f"best_observed={best_reward:.4f} at T={self.temps[np.argmax(self.rewards)]:.3f})"
        )

        return suggested

    def plot(self, save_path: str, next_suggestion: float | None = None) -> None:
        """Plot the GP posterior, observations, and next suggestion.

        Parameters
        ----------
        save_path : str or Path
            Where to save the PNG.
        next_suggestion : float or None
            If given, mark the next temperature to try.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm
        from pathlib import Path

        n = len(self.temps)
        if n < 2:
            return  # not enough data to plot

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1],
                                 sharex=True)
        plt.style.use("dark_background")
        fig.patch.set_facecolor("black")
        for ax in axes:
            ax.set_facecolor("black")

        T_grid = np.linspace(self.temp_min, self.temp_max, 200).reshape(-1, 1)

        # --- Top panel: GP posterior + observations ---
        ax = axes[0]

        # GP prediction (only if fitted, i.e. past random phase)
        if n >= self.n_random:
            w_temps, w_rewards = self._windowed_data()
            X = np.array(w_temps).reshape(-1, 1)
            y = np.array(w_rewards)
            self.gp.fit(X, y)
            mu, sigma = self.gp.predict(T_grid, return_std=True)

            ax.plot(T_grid.ravel(), mu, color="#00bfff", linewidth=2, label="GP mean")
            ax.fill_between(
                T_grid.ravel(), mu - 2 * sigma, mu + 2 * sigma,
                alpha=0.2, color="#00bfff", label="±2σ",
            )

        # Observations coloured by cycle index
        color_norm = Normalize(vmin=0, vmax=max(n - 1, 1))
        cmap = cm.plasma
        colors = [cmap(color_norm(i)) for i in range(n)]
        scatter = ax.scatter(
            self.temps, self.rewards, c=list(range(n)), cmap="plasma",
            s=60, edgecolors="white", linewidths=0.5, zorder=5,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Cycle", fontsize=10)

        # Next suggestion
        if next_suggestion is not None:
            ax.axvline(next_suggestion, color="#ff6b6b", linestyle="--",
                       linewidth=1.5, alpha=0.8, label=f"next T={next_suggestion:.3f}")

        # Best observed
        best_idx = int(np.argmax(self.rewards))
        ax.scatter(
            [self.temps[best_idx]], [self.rewards[best_idx]],
            s=150, marker="*", color="#00e676", edgecolors="white",
            linewidths=1, zorder=6, label=f"best (T={self.temps[best_idx]:.2f})",
        )

        ax.set_ylabel("Reward (−energy)", fontsize=11)
        ax.set_title("Temperature Bayesian Optimisation", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.2)

        # --- Bottom panel: EI acquisition function ---
        ax2 = axes[1]
        if n >= self.n_random:
            best_reward = max(w_rewards)
            ei_vals = []
            for t in T_grid.ravel():
                t_arr = np.array([[t]])
                mu_t, sigma_t = self.gp.predict(t_arr, return_std=True)
                if sigma_t[0] < 1e-8:
                    ei_vals.append(0.0)
                else:
                    z = (mu_t[0] - best_reward - self.xi) / sigma_t[0]
                    ei = (mu_t[0] - best_reward - self.xi) * norm.cdf(z) + sigma_t[0] * norm.pdf(z)
                    ei_vals.append(max(ei, 0.0))
            ax2.fill_between(T_grid.ravel(), ei_vals, alpha=0.4, color="#ffab40")
            ax2.plot(T_grid.ravel(), ei_vals, color="#ffab40", linewidth=1.5)
            if next_suggestion is not None:
                ax2.axvline(next_suggestion, color="#ff6b6b", linestyle="--",
                            linewidth=1.5, alpha=0.8)
        else:
            ax2.text(0.5, 0.5, f"Random phase ({n}/{self.n_random})",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=11, color="gray")

        ax2.set_xlabel("Temperature", fontsize=11)
        ax2.set_ylabel("Expected\nImprovement", fontsize=10)
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, facecolor="black", edgecolor="none")
        plt.close(fig)

    def get_summary(self) -> dict:
        """Return summary of BO state for logging."""
        if not self.temps:
            return {}
        best_idx = int(np.argmax(self.rewards))
        return {
            "n_observations": len(self.temps),
            "best_temp": self.temps[best_idx],
            "best_reward": self.rewards[best_idx],
            "last_temp": self.temps[-1],
            "last_reward": self.rewards[-1],
            "temp_mean": float(np.mean(self.temps)),
            "temp_std": float(np.std(self.temps)),
        }
