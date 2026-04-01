#!/usr/bin/env python3
"""
Hyperparameter search over BO/selection + GRPO/RL parameters.

Uses Optuna with TPE sampler and MedianPruner. Results are stored in
a SQLite database for resumability.

Usage:
    python run_hp_search.py \
        --pipeline-config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml \
        --energy-config configs/energy/energy_colabfold_ipsae_2GDZ_15PGDH.yaml \
        --n-trials 100 \
        --max-cycles 10

Resume a previous study:
    python run_hp_search.py \
        --pipeline-config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml \
        --energy-config configs/energy/energy_colabfold_ipsae_2GDZ_15PGDH.yaml \
        --n-trials 50 \
        --study-name thinkingplm_hp_search \
        --storage sqlite:///hp_search.db
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

# Ensure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import optuna
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for ProFam+BAGEL pipeline."
    )
    parser.add_argument(
        "--pipeline-config", required=True,
        help="Base pipeline YAML config.",
    )
    parser.add_argument(
        "--energy-config", required=True,
        help="Energy YAML config (should use ColabFold for speed).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=10,
        help="Pipeline cycles per trial.",
    )
    parser.add_argument(
        "--study-name", default="thinkingplm_hp_search",
        help="Optuna study name (for resumability).",
    )
    parser.add_argument(
        "--storage", default="sqlite:///hp_search.db",
        help="Optuna storage URL (default: sqlite:///hp_search.db).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for TPE sampler.",
    )
    parser.add_argument(
        "--n-startup-trials", type=int, default=10,
        help="Number of random trials before TPE kicks in.",
    )
    parser.add_argument(
        "--n-warmup-steps", type=int, default=3,
        help="Warmup cycles before pruner starts (per trial).",
    )
    parser.add_argument(
        "--skip-model-load", action="store_true",
        help="Skip loading ProFam model (for testing search space only).",
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.pipeline_config).is_file():
        parser.error(f"Pipeline config not found: {args.pipeline_config}")
    if not Path(args.energy_config).is_file():
        parser.error(f"Energy config not found: {args.energy_config}")

    # Load ProFam model once (shared across all trials)
    shared_model = None
    shared_device = "cpu"
    initial_state_dict = None

    if not args.skip_model_load:
        from run_profam_bagel_pipeline import (
            PipelineConfig,
            load_profam_model,
            load_yaml_config,
        )

        yaml_cfg = load_yaml_config(Path(args.pipeline_config))
        # Build a minimal config just for model loading
        profam_checkpoint_dir = Path(yaml_cfg.get(
            "profam_checkpoint_dir",
            ".profam_repo/model_checkpoints/profam-1"
        ))
        minimal_cfg = PipelineConfig(
            initial_fasta=None,
            profam_checkpoint_dir=profam_checkpoint_dir,
            energy_config=Path(args.energy_config),
            random_init=True,  # avoid needing initial_fasta for loading
        )

        logger.info("Loading ProFam model (shared across all trials)...")
        shared_model, shared_device = load_profam_model(minimal_cfg)
        initial_state_dict = copy.deepcopy(shared_model.state_dict())
        logger.info(f"ProFam model loaded on {shared_device}")

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=args.n_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        ),
    )

    # Create objective function
    from hp_search.objective import create_objective

    objective = create_objective(
        base_config_path=args.pipeline_config,
        energy_config_path=args.energy_config,
        max_cycles=args.max_cycles,
        shared_model=shared_model,
        shared_device=shared_device,
        initial_state_dict=initial_state_dict,
    )

    # Run optimization
    logger.info(
        f"Starting Optuna study '{args.study_name}': "
        f"{args.n_trials} trials, {args.max_cycles} cycles/trial"
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Report results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 70)
    print(f"Study: {args.study_name}")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best energy: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params to JSON
    output_path = Path("outputs/hp_search/best_params.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, "w") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_energy": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    # Print parameter importance (if enough trials)
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 10:
        try:
            importances = optuna.importance.get_param_importances(study)
            print("\nParameter importance:")
            for param, importance in sorted(
                importances.items(), key=lambda x: -x[1]
            ):
                print(f"  {param}: {importance:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
