#!/usr/bin/env python
"""
Generate config files for scaffold × method comparison on 2GDZ target.

4 scaffolds × 4 methods = 16 configs

Scaffolds (best LIS scores from prior campaigns):
  - affibody_2B87      (0.68)
  - hairpin             (0.75)
  - beta_sheet_1E0L     (0.75)
  - bindcraft_15PGDH    (0.80)

Methods:
  1. random_greedy       — random mutation, greedy selection, 1 sample/cycle
  2. proposal_bandit     — proposal bandit (profam vs random_mutation), greedy, 1 sample/cycle
  3. bandit_grpo         — proposal bandit + GRPO preference optimization, 12 samples/cycle
  4. bandit_bt           — proposal bandit + BT preference optimization, 12 samples/cycle

Total evaluations kept constant across methods:
  - Methods 1 & 2: 5400 cycles × 1 sample  = 5400 evaluations
  - Methods 3 & 4:  450 cycles × 12 samples = 5400 evaluations
"""

import hashlib
from pathlib import Path

CONFIG_DIR = Path("configs/pipelines/scaffold_comparison")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_EVALUATIONS = 5400

# (short_name, fasta_path)
SCAFFOLDS = [
    ("affibody_2B87", "configs/sequences/initial_3helix_affibody_2B87.fasta"),
    ("hairpin", "configs/sequences/initial_sequence_hairpin.fasta"),
    ("beta_sheet_1E0L", "configs/sequences/initial_single_beta_sheet_1E0L.fasta"),
    ("bindcraft_15PGDH", "configs/sequences/initial_rank2_bindcraft_15PGDH_l107_s438837_mpnn2.fasta"),
]

# ── Shared base config ──────────────────────────────────────────────────────

BASE = dict(
    profam_checkpoint_dir=".profam_repo/model_checkpoints/profam-1",
    profam_sampler="single",
    profam_max_tokens=8192,
    profam_top_p=0.95,
    energy_config="configs/energy/energy_lis_2GDZ_local.yaml",
    f_inject=0.25,
    softmax_temperature=0.01,
    run_on_modal=False,
    output_frequency=1,
    enforce_template=True,
    sample_with_reinsertion=False,
    reinject_initial=False,
    n_memory=0,
    elitism=True,
    accept_only_improvement=True,
    deduplicate_sequences=True,
    wandb_enabled=True,
    wandb_project="profam-bagel-pipeline",
)

# ── Method-specific overrides ────────────────────────────────────────────────

METHODS = {
    "random_greedy": dict(
        profam_num_samples=1,
        profam_temperature=0.8,
        max_cycles=TOTAL_EVALUATIONS,
        # Proposal
        proposal_method="random_mutation",
        max_mutations=1,
        freeze_prompt=False,
        # Selection
        selection_strategy="greedy",
        # No bandit
        thompson_proposal_bandit=False,
        # No RL
        grpo_enabled=False,
        bt_enabled=False,
    ),
    "proposal_bandit": dict(
        profam_num_samples=1,
        profam_temperature=0.8,
        max_cycles=TOTAL_EVALUATIONS,
        # Proposal
        proposal_method="profam",
        max_mutations=1,
        freeze_prompt=False,
        # Selection
        selection_strategy="greedy",
        thompson_m_samples=1,
        thompson_reward_term="LIS",
        thompson_exploit_bias=8.0,
        thompson_max_arms=10,
        thompson_max_identity=0.95,
        # Proposal bandit
        thompson_proposal_bandit=True,
        proposal_bandit_prior_alpha=2.0,
        proposal_bandit_prior_beta=2.0,
        proposal_bandit_relative_reward=False,
        # No RL
        grpo_enabled=False,
        bt_enabled=False,
    ),
    "bandit_grpo": dict(
        profam_num_samples=12,
        profam_temperature=0.7,
        max_cycles=TOTAL_EVALUATIONS // 12,
        # Proposal
        proposal_method="profam",
        max_mutations=1,
        freeze_prompt=False,
        # Selection
        selection_strategy="greedy",
        thompson_m_samples=1,
        thompson_reward_term="LIS",
        thompson_exploit_bias=8.0,
        thompson_max_arms=10,
        thompson_max_identity=0.95,
        # Proposal bandit
        thompson_proposal_bandit=True,
        proposal_bandit_prior_alpha=2.0,
        proposal_bandit_prior_beta=2.0,
        proposal_bandit_relative_reward=False,
        # GRPO
        grpo_enabled=True,
        grpo_beta=0.0,
        grpo_clip_ratio=0.2,
        grpo_lr=2.0e-05,
        grpo_replay_cycles=7,
        grpo_temperature=1.0,
        grpo_use_reference_model=False,
        grpo_group_size=16,
        rl_every_n_cycles=1,
        rl_steps_per_cycle=1,
        likelihood_eval_every=5,
        likelihood_track_n=10,
        # No BT
        bt_enabled=False,
    ),
    "bandit_bt": dict(
        profam_num_samples=12,
        profam_temperature=0.7,
        max_cycles=TOTAL_EVALUATIONS // 12,
        # Proposal
        proposal_method="profam",
        max_mutations=1,
        freeze_prompt=False,
        # Selection
        selection_strategy="greedy",
        thompson_m_samples=1,
        thompson_reward_term="LIS",
        thompson_exploit_bias=8.0,
        thompson_max_arms=10,
        thompson_max_identity=0.95,
        # Proposal bandit
        thompson_proposal_bandit=True,
        proposal_bandit_prior_alpha=2.0,
        proposal_bandit_prior_beta=2.0,
        proposal_bandit_relative_reward=False,
        # No GRPO
        grpo_enabled=False,
        # BT
        bt_enabled=True,
        bt_batch_size=32,
        bt_every_n_cycles=1,
        bt_lr=2.0e-05,
        bt_pool_size=64,
        bt_steps_per_cycle=1,
        bt_sub_batch_size=4,
        likelihood_eval_every=5,
        likelihood_track_n=10,
    ),
}


def _yaml_val(v):
    """Format a Python value as YAML."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v}"
    if v is None:
        return "null"
    return str(v)


def generate_configs(replicate: int = 1, energy_config: str | None = None):
    configs_generated = []

    rep_suffix = f"_rep{replicate}" if replicate > 1 else ""

    for scaffold_name, fasta_path in SCAFFOLDS:
        for method_name, method_overrides in METHODS.items():
            # Deterministic seed — replicate changes the seed
            seed_str = f"{scaffold_name}_{method_name}_scaffold_comparison_rep{replicate}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % 1000000

            # Merge base + method overrides
            cfg = {**BASE}
            cfg.update(method_overrides)
            if energy_config is not None:
                cfg["energy_config"] = energy_config
            cfg["initial_fasta"] = fasta_path
            cfg["random_seed"] = seed
            cfg["output_dir"] = f"outputs/scaffold_comparison{rep_suffix}/{scaffold_name}/{method_name}"
            cfg["wandb_run_name"] = f"sc_{scaffold_name}_{method_name}{rep_suffix}"
            cfg["wandb_tags"] = [
                "scaffold_comparison",
                f"rep{replicate}",
                scaffold_name,
                method_name,
            ]

            # Build YAML string
            lines = [
                f"## Scaffold Comparison: {scaffold_name} / {method_name} (replicate {replicate})",
                f"## Target: 2GDZ (15-PGDH)",
                f"## Total evaluations: {TOTAL_EVALUATIONS}",
                "",
            ]
            # Write tags separately (list syntax)
            tags = cfg.pop("wandb_tags")
            for key in sorted(cfg.keys()):
                lines.append(f"{key}: {_yaml_val(cfg[key])}")
            lines.append("wandb_tags:")
            for tag in tags:
                lines.append(f"  - {tag}")
            lines.append("")

            config_path = CONFIG_DIR / f"{scaffold_name}_{method_name}{rep_suffix}.yaml"
            config_path.write_text("\n".join(lines))
            configs_generated.append(config_path)

    return configs_generated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, default=1,
                        help="Replicate number (changes seeds and output dirs)")
    parser.add_argument("--energy_config", type=str, default=None,
                        help="Override energy config path (default: from BASE)")
    args = parser.parse_args()

    configs = generate_configs(replicate=args.replicate, energy_config=args.energy_config)
    print(f"Generated {len(configs)} config files in {CONFIG_DIR}/")

    print(f"\nReplicate: {args.replicate}")
    print(f"\nScaffolds ({len(SCAFFOLDS)}):")
    for name, path in SCAFFOLDS:
        print(f"  {name}: {path}")

    print(f"\nMethods ({len(METHODS)}):")
    for name, overrides in METHODS.items():
        n = overrides["profam_num_samples"]
        c = overrides["max_cycles"]
        t = overrides["profam_temperature"]
        print(f"  {name}: {n} samples/cycle x {c} cycles = {n*c} evals, temp={t}")

    print(f"\nTotal configs: {len(SCAFFOLDS)} x {len(METHODS)} = {len(configs)}")
