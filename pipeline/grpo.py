"""
Pipeline-integrated GRPO (Group Relative Policy Optimization) configuration.

The actual GRPO computation is done by ``model.grpo_step_from_rewards()``
in ProFam's ``BaseFamilyLitModule``.  This module provides the config
dataclass and the ``PipelineGRPOStep`` wrapper that manages the optimizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for pipeline-integrated GRPO training.

    There is no explicit "group size" knob: the effective GRPO batch for
    each step is the full replay buffer assembled by the pipeline, whose
    size is ``profam_num_samples * (grpo_replay_cycles + 1)``.
    """

    enabled: bool = False
    grpo_beta: float = 0.05          # KL penalty coefficient
    grpo_clip_ratio: float = 0.2     # PPO-style clipping epsilon
    grpo_lr: float = 1e-5            # Learning rate for AdamW
    grpo_weight_decay: float = 0.01  # Weight decay
    grpo_temperature: float = 1.0    # Sampling temperature during RL generation
    grpo_top_p: float = 0.95         # Top-p for RL generation
    grpo_max_tokens: int = 8000      # Max tokens per batch (for gradient memory)
    grpo_normalize_rewards: bool = True   # Z-score normalize rewards within group
    grpo_reward_baseline: str = "mean"    # "mean" | "min" | "none"
    grpo_use_reference_model: bool = False  # KL to frozen initial weights
    rl_every_n_cycles: int = 1       # Run RL every N pipeline cycles
    rl_steps_per_cycle: int = 1      # Gradient steps per RL invocation


class PipelineGRPOStep:
    """
    Manages the optimizer for GRPO weight updates in the pipeline.

    The actual GRPO loss computation (PPO-clipped importance ratios,
    advantage normalization, KL regularization) is delegated to
    ``model.grpo_step_from_rewards()``.  This class owns the optimizer
    and provides ``should_run()`` scheduling logic.

    Parameters
    ----------
    model : LlamaLitModule
        The ProFam model.
    config : GRPOConfig
        GRPO hyperparameters.
    device : str
        Device for model computation.
    """

    def __init__(
        self,
        model: Any,  # LlamaLitModule
        config: GRPOConfig,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.grpo_lr,
            weight_decay=config.grpo_weight_decay,
        )

    def should_run(self, cycle: int) -> bool:
        """Check if GRPO should run on this cycle."""
        return self.config.enabled and (cycle % self.config.rl_every_n_cycles == 0)
