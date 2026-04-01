"""
Hyperparameter search space definitions for Optuna.

Defines the full search space over both BO/selection parameters
and GRPO/RL parameters, with conditional logic for strategy-
dependent parameters.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna


def sample_bo_selection_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample Bayesian optimization / selection hyperparameters."""
    params: Dict[str, Any] = {}

    # Selection strategy
    params["selection_strategy"] = trial.suggest_categorical(
        "selection_strategy", ["greedy", "thompson"]
    )

    # Generation parameters (always relevant)
    params["profam_temperature"] = trial.suggest_float(
        "profam_temperature", 0.3, 2.0
    )
    params["profam_top_p"] = trial.suggest_float(
        "profam_top_p", 0.7, 1.0
    )
    params["profam_num_samples"] = trial.suggest_int(
        "profam_num_samples", 4, 64, log=True
    )

    # Selection parameters
    params["softmax_temperature"] = trial.suggest_float(
        "softmax_temperature", 0.01, 5.0, log=True
    )
    params["f_inject"] = trial.suggest_float(
        "f_inject", 0.05, 1.0
    )
    params["elitism"] = trial.suggest_categorical(
        "elitism", [True, False]
    )
    params["n_memory"] = trial.suggest_int(
        "n_memory", 0, 5
    )

    # Conditional Thompson sampling parameters
    if params["selection_strategy"] == "thompson":
        params["thompson_exploit_bias"] = trial.suggest_float(
            "thompson_exploit_bias", 1.0, 10.0
        )
        params["thompson_m_samples"] = trial.suggest_int(
            "thompson_m_samples", 1, 10
        )
        params["thompson_max_arms"] = trial.suggest_int(
            "thompson_max_arms", 5, 50
        )
        params["thompson_max_identity"] = trial.suggest_float(
            "thompson_max_identity", 0.8, 0.99
        )

    return params


def sample_grpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample GRPO/RL hyperparameters."""
    params: Dict[str, Any] = {}

    params["grpo_enabled"] = trial.suggest_categorical(
        "grpo_enabled", [True, False]
    )

    if params["grpo_enabled"]:
        params["grpo_lr"] = trial.suggest_float(
            "grpo_lr", 1e-6, 1e-3, log=True
        )
        params["grpo_beta"] = trial.suggest_float(
            "grpo_beta", 0.001, 0.2, log=True
        )
        params["grpo_group_size"] = trial.suggest_int(
            "grpo_group_size", 4, 32, log=True
        )
        params["grpo_clip_ratio"] = trial.suggest_float(
            "grpo_clip_ratio", 0.1, 0.5
        )
        params["grpo_temperature"] = trial.suggest_float(
            "grpo_temperature", 0.5, 2.0
        )
        params["rl_every_n_cycles"] = trial.suggest_int(
            "rl_every_n_cycles", 1, 5
        )

    return params


def sample_all_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample the full hyperparameter space (BO + RL)."""
    params = sample_bo_selection_params(trial)
    params.update(sample_grpo_params(trial))
    return params
