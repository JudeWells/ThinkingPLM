"""Thompson sampling bandits for the ProFam + BAGEL pipeline.

Contains:
- ThompsonArm / ThompsonSampler: multi-armed bandit over conditioning sequences
- TemperatureBandit: Thompson sampler over discrete temperature bins
- ProposalBandit: Thompson sampler over proposal methods (profam vs random_mutation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from pipeline.utils import compute_sequence_identity


@dataclass
class ThompsonArm:
    arm_id: int
    sequence: str
    name: str
    alpha: float              # Beta posterior α
    beta_param: float         # Beta posterior β
    ipsae_raw: float          # Raw ipSAE at creation
    reward: float             # clamp(-ipsae_raw, 0, 1)
    parent_arm_id: int | None
    created_at_cycle: int
    times_selected: int = 0
    total_reward_credited: float = 0.0


class ThompsonSampler:
    """Manages a pool of arms for Thompson sampling over conditioning sequences."""

    def __init__(
        self,
        m_samples: int = 1,
        exploit_bias: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.m_samples = max(1, m_samples)
        self.exploit_bias = max(1.0, exploit_bias)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.arms: Dict[int, ThompsonArm] = {}
        self._next_arm_id = 0

    @staticmethod
    def _ipsae_to_reward(ipsae_raw: float) -> float:
        """Convert raw ipSAE (negative = better) to reward in [0, 1]."""
        return float(np.clip(-ipsae_raw, 0.0, 1.0))

    def add_arm(
        self,
        sequence: str,
        name: str,
        ipsae_raw: float,
        parent_arm_id: int | None,
        cycle: int,
    ) -> ThompsonArm:
        """Register a new arm with Beta(1 + r, 2 - r) prior from its own ipSAE."""
        reward = self._ipsae_to_reward(ipsae_raw)
        arm = ThompsonArm(
            arm_id=self._next_arm_id,
            sequence=sequence,
            name=name,
            alpha=1.0 + reward,
            beta_param=2.0 - reward,
            ipsae_raw=ipsae_raw,
            reward=reward,
            parent_arm_id=parent_arm_id,
            created_at_cycle=cycle,
        )
        self.arms[arm.arm_id] = arm
        self._next_arm_id += 1
        return arm

    def select_arm(self) -> ThompsonArm:
        """Thompson sampling: sample θ ~ Beta(α*b, β*b) per arm, pick max.

        When exploit_bias > 1, both α and β are scaled up before sampling,
        which concentrates the Beta distribution around its mean — making
        selection more greedy (exploitative).  exploit_bias=1.0 is standard
        Thompson sampling.
        """
        if not self.arms:
            raise RuntimeError("No arms registered in ThompsonSampler.")
        best_arm = None
        best_theta = -1.0
        b = self.exploit_bias
        for arm in self.arms.values():
            thetas = self.rng.beta(arm.alpha * b, arm.beta_param * b, size=self.m_samples)
            theta = float(np.max(thetas))
            if theta > best_theta:
                best_theta = theta
                best_arm = arm
        assert best_arm is not None
        best_arm.times_selected += 1
        return best_arm

    def update_arm(self, arm_id: int, progeny_ipsae: float) -> None:
        """Update the chosen arm's posterior with the progeny's reward."""
        arm = self.arms[arm_id]
        reward = self._ipsae_to_reward(progeny_ipsae)
        arm.alpha += reward
        arm.beta_param += (1.0 - reward)
        arm.total_reward_credited += reward

    def get_state_dict(self) -> List[Dict[str, Any]]:
        """Serialize all arm states for JSON logging."""
        result = []
        for arm in self.arms.values():
            result.append({
                "arm_id": arm.arm_id,
                "sequence": arm.sequence,
                "name": arm.name,
                "alpha": arm.alpha,
                "beta_param": arm.beta_param,
                "ipsae_raw": arm.ipsae_raw,
                "reward": arm.reward,
                "parent_arm_id": arm.parent_arm_id,
                "created_at_cycle": arm.created_at_cycle,
                "times_selected": arm.times_selected,
                "total_reward_credited": arm.total_reward_credited,
            })
        return result

    def prune_to_top_k_diverse(
        self,
        k: int,
        max_identity: float = 0.95,
    ) -> Dict[str, Any]:
        """Keep top-K arms ensuring diversity via sequence identity threshold.

        Uses greedy selection: iterates through arms sorted by posterior mean
        (expected reward = alpha / (alpha + beta)), keeping an arm only if its
        sequence identity to all already-retained arms is below max_identity.

        Args:
            k: Maximum number of arms to retain.
            max_identity: Maximum allowed sequence identity between any two retained
                          arms. Arms more similar than this are considered redundant.
                          Default 0.95 (95% identity) allows closely related variants
                          while filtering near-duplicates.

        Returns:
            Dict with pruning statistics: arms_before, arms_after, retained_ids,
            pruned_ids, and pairwise_identities of retained arms.
        """
        arms_before = len(self.arms)
        if arms_before <= k:
            return {
                "arms_before": arms_before,
                "arms_after": arms_before,
                "retained_ids": list(self.arms.keys()),
                "pruned_ids": [],
                "pairwise_identities": [],
            }

        sorted_arms = sorted(
            self.arms.values(),
            key=lambda a: a.alpha / (a.alpha + a.beta_param),
            reverse=True,
        )

        retained: List[ThompsonArm] = []
        for arm in sorted_arms:
            if len(retained) >= k:
                break
            is_diverse = all(
                compute_sequence_identity(arm.sequence, r.sequence) < max_identity
                for r in retained
            )
            if is_diverse:
                retained.append(arm)

        pairwise_identities = []
        for i, arm_i in enumerate(retained):
            for j, arm_j in enumerate(retained):
                if i < j:
                    identity = compute_sequence_identity(arm_i.sequence, arm_j.sequence)
                    pairwise_identities.append({
                        "arm_i": arm_i.arm_id,
                        "arm_j": arm_j.arm_id,
                        "identity": identity,
                    })

        retained_ids = {a.arm_id for a in retained}
        pruned_ids = [aid for aid in self.arms.keys() if aid not in retained_ids]
        self.arms = {aid: arm for aid, arm in self.arms.items() if aid in retained_ids}

        return {
            "arms_before": arms_before,
            "arms_after": len(self.arms),
            "retained_ids": list(retained_ids),
            "pruned_ids": pruned_ids,
            "pairwise_identities": pairwise_identities,
        }


class TemperatureBandit:
    """Thompson sampler over a discrete set of temperature bins."""

    def __init__(
        self,
        bins: List[float],
        exploit_bias: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.bins = sorted(bins)
        self.exploit_bias = max(1.0, exploit_bias)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.alphas = {t: 1.0 for t in self.bins}
        self.betas = {t: 1.0 for t in self.bins}
        self.times_selected: Dict[float, int] = {t: 0 for t in self.bins}
        self.total_reward: Dict[float, float] = {t: 0.0 for t in self.bins}

    def select(self) -> float:
        """Thompson-sample a temperature bin."""
        best_temp = self.bins[0]
        best_theta = -1.0
        b = self.exploit_bias
        for t in self.bins:
            theta = float(self.rng.beta(self.alphas[t] * b, self.betas[t] * b))
            if theta > best_theta:
                best_theta = theta
                best_temp = t
        self.times_selected[best_temp] += 1
        return best_temp

    def update(self, temperature: float, reward: float) -> None:
        """Update the chosen bin's posterior with the observed reward."""
        self.alphas[temperature] += reward
        self.betas[temperature] += (1.0 - reward)
        self.total_reward[temperature] += reward

    def get_state_dict(self) -> List[Dict[str, Any]]:
        """Serialize state for JSON logging."""
        result = []
        for t in self.bins:
            a, b = self.alphas[t], self.betas[t]
            result.append({
                "temperature": t,
                "alpha": a,
                "beta_param": b,
                "expected_reward": a / (a + b),
                "times_selected": self.times_selected[t],
                "total_reward": self.total_reward[t],
            })
        return result


class ProposalBandit:
    """Thompson sampler over proposal methods (profam vs random_mutation).

    Uses a Beta-Bernoulli model: each method's reward is the probability of
    improving over the current best energy.  On each cycle, the selected
    method gets α += 1 if improvement occurred, β += 1 otherwise.
    """

    METHODS = ["profam", "random_mutation"]

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.rng = rng if rng is not None else np.random.default_rng()
        self.alphas = {m: prior_alpha for m in self.METHODS}
        self.betas = {m: prior_beta for m in self.METHODS}
        self.times_selected: Dict[str, int] = {m: 0 for m in self.METHODS}

    def select(self) -> str:
        """Thompson-sample a proposal method."""
        best_method = self.METHODS[0]
        best_theta = -1.0
        for m in self.METHODS:
            theta = float(self.rng.beta(self.alphas[m], self.betas[m]))
            if theta > best_theta:
                best_theta = theta
                best_method = m
        self.times_selected[best_method] += 1
        return best_method

    def update(self, method: str, improved: bool) -> None:
        """Update posterior: α += 1 if improved, β += 1 otherwise."""
        if improved:
            self.alphas[method] += 1
        else:
            self.betas[method] += 1

    def get_state_dict(self) -> List[Dict[str, Any]]:
        """Serialize state for JSON logging."""
        result = []
        for m in self.METHODS:
            a, b = self.alphas[m], self.betas[m]
            result.append({
                "method": m,
                "alpha": a,
                "beta_param": b,
                "expected_reward": a / (a + b),
                "times_selected": self.times_selected[m],
            })
        return result
