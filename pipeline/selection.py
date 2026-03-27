"""Selection management for the ProFam + BAGEL pipeline.

Contains:
- SelectionResult: Result of prompt selection
- PromptSelector: Strategy pattern for selecting conditioning sequences
- SelectionManager: Orchestrator for arm lifecycle management
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from pipeline.bandits import ThompsonArm, ThompsonSampler
from pipeline.utils import extract_reward_term


@dataclass
class SelectionResult:
    """Result of prompt selection."""
    selected_arm: ThompsonArm | None
    selected_sequence: str
    selected_name: str
    selected_ipsae: float
    method: str  # "thompson" or "greedy"


class PromptSelector(ABC):
    """Abstract base class for prompt selection strategies."""

    @abstractmethod
    def select(self, thompson_sampler: ThompsonSampler) -> SelectionResult:
        """Select a prompt sequence from the Thompson sampler's arm pool.

        Args:
            thompson_sampler: The ThompsonSampler containing all registered arms.

        Returns:
            SelectionResult with the selected arm and its metadata.
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this selection method."""
        pass


class GreedyPromptSelector(PromptSelector):
    """Picks the arm with the lowest (best) ipSAE_raw value."""

    def select(self, thompson_sampler: ThompsonSampler) -> SelectionResult:
        if not thompson_sampler.arms:
            raise RuntimeError("No arms registered in ThompsonSampler.")

        best_arm = min(
            thompson_sampler.arms.values(),
            key=lambda a: a.ipsae_raw,
        )
        best_arm.times_selected += 1
        thompson_sampler._last_selected_arm_id = best_arm.arm_id

        return SelectionResult(
            selected_arm=best_arm,
            selected_sequence=best_arm.sequence,
            selected_name=best_arm.name,
            selected_ipsae=best_arm.ipsae_raw,
            method="greedy",
        )

    @property
    def method_name(self) -> str:
        return "greedy"


class ThompsonPromptSelector(PromptSelector):
    """Uses Thompson sampling to select the next arm."""

    def select(self, thompson_sampler: ThompsonSampler) -> SelectionResult:
        if not thompson_sampler.arms:
            raise RuntimeError("No arms registered in ThompsonSampler.")

        # Use the sampler's select_arm method for true Thompson sampling
        selected_arm = thompson_sampler.select_arm()
        thompson_sampler._last_selected_arm_id = selected_arm.arm_id

        return SelectionResult(
            selected_arm=selected_arm,
            selected_sequence=selected_arm.sequence,
            selected_name=selected_arm.name,
            selected_ipsae=selected_arm.ipsae_raw,
            method="thompson",
        )

    @property
    def method_name(self) -> str:
        return "thompson"


class SelectionManager:
    """Orchestrates the arm lifecycle for Thompson sampling.

    Handles:
    - Registering progeny sequences as new arms
    - Updating parent arm posteriors
    - Pruning arms for diversity
    - Selecting the next prompt via the configured selector
    - Building the injection set from selection results
    """

    def __init__(
        self,
        thompson_sampler: ThompsonSampler,
        prompt_selector: PromptSelector,
        reward_term: str = "ipSAE",
        max_arms: int = 0,
        max_identity: float = 0.95,
    ):
        """Initialize the SelectionManager.

        Args:
            thompson_sampler: The ThompsonSampler to manage.
            prompt_selector: Strategy for selecting prompts (greedy or thompson).
            reward_term: Name of the energy term to use as reward (default: "ipSAE").
            max_arms: Maximum number of arms to retain (0 = unlimited).
            max_identity: Maximum sequence identity between retained arms.
        """
        self.thompson_sampler = thompson_sampler
        self.prompt_selector = prompt_selector
        self.reward_term = reward_term
        self.max_arms = max_arms
        self.max_identity = max_identity

    def register_progeny(
        self,
        names: Sequence[str],
        sequences: Sequence[str],
        details: Sequence[Dict[str, Any]],
        cycle: int,
    ) -> Dict[str, Any]:
        """Register all progeny with finite reward as new arms.

        Args:
            names: Names of generated sequences.
            sequences: Generated sequences.
            details: Evaluation details for each sequence.
            cycle: Current cycle number.

        Returns:
            Dict with registration statistics.
        """
        reward_values = extract_reward_term(details, self.reward_term)
        parent_arm_id = getattr(self.thompson_sampler, '_last_selected_arm_id', None)

        n_registered = 0
        n_skipped_inf = 0

        for name, seq, ipsae_val in zip(names, sequences, reward_values):
            if math.isfinite(ipsae_val):
                self.thompson_sampler.add_arm(
                    sequence=seq,
                    name=name,
                    ipsae_raw=ipsae_val,
                    parent_arm_id=parent_arm_id,
                    cycle=cycle,
                )
                n_registered += 1
            else:
                n_skipped_inf += 1

        return {
            "n_registered": n_registered,
            "n_skipped_inf": n_skipped_inf,
            "n_total_arms": len(self.thompson_sampler.arms),
            "parent_arm_id": parent_arm_id,
        }

    def update_parent_posterior(
        self,
        details: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Update the parent arm's Beta distribution with the best progeny reward.

        Args:
            details: Evaluation details for all progeny.

        Returns:
            Dict with update statistics, or None if no update was made.
        """
        if not hasattr(self.thompson_sampler, '_last_selected_arm_id'):
            return None

        parent_id = self.thompson_sampler._last_selected_arm_id
        if parent_id not in self.thompson_sampler.arms:
            return None

        parent_arm = self.thompson_sampler.arms[parent_id]
        reward_values = extract_reward_term(details, self.reward_term)

        # Find the best (most negative) finite reward value
        finite_rewards = [(i, v) for i, v in enumerate(reward_values) if math.isfinite(v)]

        if not finite_rewards:
            return {
                "parent_arm_id": parent_id,
                "parent_name": parent_arm.name,
                "updated": False,
                "reason": "no_finite_progeny",
            }

        best_progeny_idx, best_progeny_ipsae = min(finite_rewards, key=lambda x: x[1])

        # Capture state before update
        alpha_before = parent_arm.alpha
        beta_before = parent_arm.beta_param

        # Update the arm
        self.thompson_sampler.update_arm(parent_id, best_progeny_ipsae)
        progeny_reward = ThompsonSampler._ipsae_to_reward(best_progeny_ipsae)

        return {
            "parent_arm_id": parent_id,
            "parent_name": parent_arm.name,
            "updated": True,
            "best_progeny_idx": best_progeny_idx,
            "best_progeny_ipsae": best_progeny_ipsae,
            "progeny_reward": progeny_reward,
            "alpha_before": alpha_before,
            "alpha_after": parent_arm.alpha,
            "beta_before": beta_before,
            "beta_after": parent_arm.beta_param,
            "expected_reward_after": parent_arm.alpha / (parent_arm.alpha + parent_arm.beta_param),
        }

    def prune_arms(self) -> Dict[str, Any]:
        """Prune to top-K diverse arms if configured.

        Returns:
            Dict with pruning statistics.
        """
        if self.max_arms <= 0:
            return {
                "pruned": False,
                "reason": "max_arms_disabled",
                "arms_count": len(self.thompson_sampler.arms),
            }

        stats = self.thompson_sampler.prune_to_top_k_diverse(
            k=self.max_arms,
            max_identity=self.max_identity,
        )

        return {
            "pruned": stats["arms_before"] > stats["arms_after"],
            "arms_before": stats["arms_before"],
            "arms_after": stats["arms_after"],
            "retained_ids": stats["retained_ids"],
            "pruned_ids": stats["pruned_ids"],
        }

    def select_prompt(self) -> SelectionResult:
        """Select the next prompt sequence using the configured selector.

        Returns:
            SelectionResult with the selected arm and metadata.
        """
        return self.prompt_selector.select(self.thompson_sampler)

    def build_injection_set(
        self,
        result: SelectionResult,
    ) -> Tuple[List[str], List[str]]:
        """Build the injection set from the selection result.

        Args:
            result: The SelectionResult from select_prompt().

        Returns:
            Tuple of (names, sequences) for injection.
        """
        return ([result.selected_name], [result.selected_sequence])

    def get_top_arms_summary(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get a summary of the top N arms by expected reward.

        Args:
            n: Number of top arms to include.

        Returns:
            List of arm summary dicts.
        """
        arms_by_expected = sorted(
            self.thompson_sampler.arms.values(),
            key=lambda a: a.alpha / (a.alpha + a.beta_param),
            reverse=True,
        )

        summary = []
        for arm in arms_by_expected[:n]:
            summary.append({
                "arm_id": arm.arm_id,
                "name": arm.name,
                "alpha": arm.alpha,
                "beta_param": arm.beta_param,
                "expected_reward": arm.alpha / (arm.alpha + arm.beta_param),
                "ipsae_raw": arm.ipsae_raw,
                "times_selected": arm.times_selected,
                "total_reward_credited": arm.total_reward_credited,
                "parent_arm_id": arm.parent_arm_id,
                "created_at_cycle": arm.created_at_cycle,
            })

        return summary

    def get_reward_stats(
        self,
        details: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get statistics about reward values from evaluation details.

        Args:
            details: Evaluation details for all sequences.

        Returns:
            Dict with reward statistics.
        """
        import numpy as np

        reward_values = extract_reward_term(details, self.reward_term)
        n_finite = sum(1 for v in reward_values if math.isfinite(v))
        n_inf = len(reward_values) - n_finite

        stats = {
            "reward_term": self.reward_term,
            "n_total": len(reward_values),
            "n_finite": n_finite,
            "n_inf": n_inf,
        }

        if n_finite > 0:
            finite_vals = [v for v in reward_values if math.isfinite(v)]
            stats["min"] = min(finite_vals)
            stats["max"] = max(finite_vals)
            stats["mean"] = float(np.mean(finite_vals))

        return stats
