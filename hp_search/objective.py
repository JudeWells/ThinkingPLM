"""
Optuna objective function for hyperparameter search.

Wraps a truncated pipeline run as an Optuna objective. Each trial:
1. Samples hyperparameters from the search space
2. Builds a pipeline config with those overrides
3. Runs the pipeline for max_cycles cycles
4. Returns the best energy achieved (to minimize)

Supports model reuse across trials and Optuna pruning via
intermediate per-cycle energy reports.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna

logger = logging.getLogger(__name__)


def build_config_from_overrides(
    base_config_path: str,
    energy_config_path: str,
    overrides: Dict[str, Any],
    max_cycles: int,
    output_dir: str,
) -> Any:
    """Build a PipelineConfig by loading base YAML and applying overrides."""
    import argparse
    import sys

    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from run_profam_bagel_pipeline import (
        PipelineConfig,
        load_yaml_config,
        merge_config,
    )

    yaml_cfg = load_yaml_config(Path(base_config_path))

    # Apply overrides to yaml_cfg
    for key, value in overrides.items():
        yaml_cfg[key] = value

    # Force settings for HP search
    yaml_cfg["max_cycles"] = max_cycles
    yaml_cfg["output_dir"] = output_dir
    yaml_cfg["energy_config"] = energy_config_path

    # Build a minimal argparse.Namespace with None values
    # (so merge_config picks everything from yaml_cfg)
    dummy_args = argparse.Namespace(**{k: None for k in yaml_cfg})

    cfg = merge_config(yaml_cfg, dummy_args)
    return cfg


def create_objective(
    base_config_path: str,
    energy_config_path: str,
    max_cycles: int = 10,
    shared_model: Any = None,
    shared_device: str = "cuda",
    initial_state_dict: Optional[Dict[str, Any]] = None,
    colabfold_override: Optional[Dict[str, Any]] = None,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function.

    Parameters
    ----------
    base_config_path : str
        Path to the base pipeline YAML config.
    energy_config_path : str
        Path to the energy YAML config (should use ColabFold).
    max_cycles : int
        Number of pipeline cycles per trial.
    shared_model : Any
        Pre-loaded ProFam model (shared across trials).
    shared_device : str
        Device for the shared model.
    initial_state_dict : dict
        Initial model state_dict for restoring between GRPO trials.
    colabfold_override : dict
        Optional override for ColabFold oracle settings.

    Returns
    -------
    Callable
        Optuna objective function.
    """
    from hp_search.search_space import sample_all_params

    def objective(trial: optuna.Trial) -> float:
        # 1. Restore model to initial weights (critical for GRPO trials)
        if shared_model is not None and initial_state_dict is not None:
            shared_model.load_state_dict(copy.deepcopy(initial_state_dict))
            shared_model.eval()

        # 2. Sample all hyperparameters
        overrides = sample_all_params(trial)
        logger.info(f"Trial {trial.number}: {overrides}")

        # 3. Build config
        trial_output_dir = f"outputs/hp_search/trial_{trial.number:04d}"
        try:
            cfg = build_config_from_overrides(
                base_config_path=base_config_path,
                energy_config_path=energy_config_path,
                overrides=overrides,
                max_cycles=max_cycles,
                output_dir=trial_output_dir,
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number}: config build failed: {e}")
            raise optuna.TrialPruned(f"Config build failed: {e}")

        # 4. Run pipeline
        from run_profam_bagel_pipeline import run_pipeline

        try:
            result = run_pipeline(
                cfg,
                shared_model=shared_model,
                shared_device=shared_device,
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number}: pipeline failed: {e}")
            raise optuna.TrialPruned(f"Pipeline failed: {e}")

        # 5. Report intermediate values for pruning
        per_cycle_best = result.get("per_cycle_best", [])
        for step, energy in enumerate(per_cycle_best):
            trial.report(energy, step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at cycle {step} with energy {energy:.4f}"
                )

        # 6. Return best energy
        best_energy = result.get("best_energy", float("inf"))
        logger.info(
            f"Trial {trial.number}: best_energy={best_energy:.4f}, "
            f"params={trial.params}"
        )
        return best_energy

    return objective
