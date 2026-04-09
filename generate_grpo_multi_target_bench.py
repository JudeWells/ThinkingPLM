#!/usr/bin/env python
"""
Generate benchmark configs for multi-target × scaffold × method comparison.

5 targets × 4 scaffolds × 3 methods = 60 configs

Targets (ESMFold + LIS energy):
  - 2VSM_nipah
  - 4OYD_epstein_barr
  - 4ZQK_PD-L1
  - 1TNF_TNF_alpha
  - 1YCR_MDM2

Scaffolds:
  - 4D5 (human single domain antibody)
  - ankyrin (ankyrin repeat)
  - nanobody (nanobody-like)
  - random_init (no initial sequence)

Methods:
  1. random_greedy       — random mutation, greedy selection, 1 sample/cycle
  2. proposal_bandit     — proposal bandit, greedy selection, 1 sample/cycle
  3. bandit_grpo         — proposal bandit + GRPO, 12 samples/cycle

Total evaluations kept constant: 5400
  - Methods 1 & 2: 5400 cycles × 1 sample
  - Method 3:       450 cycles × 12 samples
"""

import argparse
import hashlib
from pathlib import Path

TOTAL_EVALUATIONS = 5400

# Energy config variants
ENERGY_VARIANTS = {
    "mt": {  # original: LIS only
        "2VSM_nipah": "configs/energy/energy_lis_2VSM_nipah.yaml",
        "4OYD_epstein_barr": "configs/energy/energy_lis_4OYD_epstein_barr.yaml",
        "4ZQK_PD-L1": "configs/energy/energy_lis_4ZQK_PD-L1.yaml",
        "1TNF_TNF_alpha": "configs/energy/energy_lis_1TNF_TNF_alpha.yaml",
        "1YCR_MDM2": "configs/energy/energy_lis_1YCR_MDM2.yaml",
    },
    "mt2": {  # LIS + PLDDT (weight 0.1)
        "2VSM_nipah": "configs/energy/energy_lis_plddt_2VSM_nipah.yaml",
        "4OYD_epstein_barr": "configs/energy/energy_lis_plddt_4OYD_epstein_barr.yaml",
        "4ZQK_PD-L1": "configs/energy/energy_lis_plddt_4ZQK_PD-L1.yaml",
        "1TNF_TNF_alpha": "configs/energy/energy_lis_plddt_1TNF_TNF_alpha.yaml",
        "1YCR_MDM2": "configs/energy/energy_lis_plddt_1YCR_MDM2.yaml",
    },
}
TARGET_NAMES = list(ENERGY_VARIANTS["mt"].keys())

# (scaffold_name, fasta_path or None for random_init)
SCAFFOLDS = [
    ("4D5", "configs/sequences/initial_sequence_human_single_domain_antibody_4D5.fasta"),
    ("ankyrin", "configs/sequences/initial_sequence_ankyrin_repeat.fasta"),
    ("nanobody", "configs/sequences/initial_sequence_nanobody_like.fasta"),
    ("random_init", None),
]

RANDOM_INIT_MAX_RESIDUES = 80

# ── Shared base config ──────────────────────────────────────────────────────

BASE = dict(
    profam_checkpoint_dir=".profam_repo/model_checkpoints/profam-1",
    profam_sampler="single",
    profam_max_tokens=8192,
    profam_top_p=0.95,
    f_inject=0.25,
    softmax_temperature=0.01,
    run_on_modal=False,
    output_frequency=1,
    enforce_template=False,
    sample_with_reinsertion=False,
    reinject_initial=True,
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


def generate_configs(replicate: int = 1, variant: str = "mt"):
    energy_map = ENERGY_VARIANTS[variant]
    config_dir = Path(f"configs/pipelines/multi_target_bench_{variant}" if variant != "mt"
                      else "configs/pipelines/multi_target_bench")
    config_dir.mkdir(parents=True, exist_ok=True)

    configs_generated = []
    rep_suffix = f"_rep{replicate}" if replicate > 1 else ""

    for target_name in TARGET_NAMES:
        energy_config = energy_map[target_name]
        for scaffold_name, fasta_path in SCAFFOLDS:
            for method_name, method_overrides in METHODS.items():
                # Deterministic seed — variant changes seeds so mt2 != mt
                seed_str = f"{target_name}_{scaffold_name}_{method_name}_{variant}_rep{replicate}"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % 1000000

                # Merge base + method overrides
                cfg = {**BASE}
                cfg.update(method_overrides)
                cfg["energy_config"] = energy_config
                cfg["random_seed"] = seed

                # Handle random_init scaffold
                if fasta_path is None:
                    cfg["random_init"] = True
                    cfg["random_init_max_residues"] = RANDOM_INIT_MAX_RESIDUES
                else:
                    cfg["initial_fasta"] = fasta_path

                cfg["output_dir"] = f"outputs/{variant}_bench{rep_suffix}/{target_name}/{scaffold_name}/{method_name}"
                cfg["wandb_run_name"] = f"{variant}_{target_name}_{scaffold_name}_{method_name}{rep_suffix}"
                cfg["wandb_tags"] = [
                    f"{variant}_bench",
                    f"rep{replicate}",
                    target_name,
                    scaffold_name,
                    method_name,
                ]

                # Build YAML string
                lines = [
                    f"## {variant.upper()} Bench: {target_name} / {scaffold_name} / {method_name} (rep {replicate})",
                    f"## Total evaluations: {TOTAL_EVALUATIONS}",
                    f"## Energy: {energy_config}",
                    "",
                ]
                tags = cfg.pop("wandb_tags")
                for key in sorted(cfg.keys()):
                    lines.append(f"{key}: {_yaml_val(cfg[key])}")
                lines.append("wandb_tags:")
                for tag in tags:
                    lines.append(f"  - {tag}")
                lines.append("")

                config_path = config_dir / f"{target_name}_{scaffold_name}_{method_name}{rep_suffix}.yaml"
                config_path.write_text("\n".join(lines))
                configs_generated.append(config_path)

    return configs_generated, config_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, default=1,
                        help="Replicate number (changes seeds and output dirs)")
    parser.add_argument("--variant", type=str, default="mt",
                        choices=list(ENERGY_VARIANTS.keys()),
                        help="Energy variant: 'mt' (LIS only) or 'mt2' (LIS + PLDDT)")
    args = parser.parse_args()

    configs, config_dir = generate_configs(replicate=args.replicate, variant=args.variant)
    print(f"Generated {len(configs)} config files in {config_dir}/")

    print(f"\nVariant: {args.variant}, Replicate: {args.replicate}")
    print(f"Energy configs: {list(ENERGY_VARIANTS[args.variant].values())[0]} (etc.)")

    print(f"\nTargets ({len(TARGET_NAMES)}):")
    for name in TARGET_NAMES:
        print(f"  {name}: {ENERGY_VARIANTS[args.variant][name]}")

    print(f"\nScaffolds ({len(SCAFFOLDS)}):")
    for name, path in SCAFFOLDS:
        print(f"  {name}: {path or 'random_init'}")

    print(f"\nMethods ({len(METHODS)}):")
    for name, overrides in METHODS.items():
        n = overrides["profam_num_samples"]
        c = overrides["max_cycles"]
        t = overrides["profam_temperature"]
        print(f"  {name}: {n} samples/cycle x {c} cycles = {n*c} evals, temp={t}")

    print(f"\nTotal configs: {len(TARGET_NAMES)} x {len(SCAFFOLDS)} x {len(METHODS)} = {len(configs)}")
