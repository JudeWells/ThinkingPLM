#!/usr/bin/env python
"""
Generate config files for 2GDZ_15PGDH campaign.

Scaffolds: 14 different starting sequences
Strategies: 8 (random_greedy, random_thompson, bandit_greedy_eb8/16, bandit_thompson_eb8/16, thompson_eb8/16_bandit_rel)
Total: 112 configs
"""

from pathlib import Path

# Output directory for configs
CONFIG_DIR = Path("configs/pipelines/2gdz_campaign")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Scaffolds: (short_name, fasta_path)
SCAFFOLDS = [
    # Original 4 from benchmark
    ("3helix", "configs/sequences/initial_3helix_scaffold_pdb_1LQZ"),
    ("4D5", "configs/sequences/initial_sequence_human_single_domain_antibody_4D5.fasta"),
    ("ankyrin", "configs/sequences/initial_sequence_ankyrin_repeat.fasta"),
    ("nanobody", "configs/sequences/initial_sequence_nanobody_like.fasta"),
    # New scaffolds
    ("affibody_2B87", "configs/sequences/initial_3helix_affibody_2B87.fasta"),
    ("ankyrin_1MJ0", "configs/sequences/initial_ankyrin_repeat_1MJ0.fasta"),
    ("fibronectin_1TTG", "configs/sequences/initial_fibronectin_monoobody_1TTG.fasta"),
    ("pdz_1BE9", "configs/sequences/initial_pdz_domain_1BE9.fasta"),
    ("beta_sheet_1E0L", "configs/sequences/initial_single_beta_sheet_1E0L.fasta"),
    ("hairpin", "configs/sequences/initial_sequence_hairpin.fasta"),
    ("rfd3_inpaint", "configs/sequences/initial_sequence_rfd3_inpaint.fasta"),
    ("boltz2_denovo", "configs/sequences/initial_sequence_boltz2_de_novo.fasta"),
    ("bindcraft_denovo", "configs/sequences/initial_sequence_bindcraft_de_novo.fasta"),
    ("bindcraft_15PGDH", "configs/sequences/initial_rank2_bindcraft_15PGDH_l107_s438837_mpnn2.fasta"),
]

# Strategy templates
# (strategy_name, selection_strategy, proposal_method, thompson_proposal_bandit, exploit_bias)
STRATEGIES = [
    # Random mutation strategies (no bandit, no exploit_bias relevant)
    ("random_greedy", "greedy", "random_mutation", False, None),
    ("random_thompson", "thompson", "random_mutation", False, 5.0),
    # Bandit strategies with EB=8
    ("bandit_greedy_eb8", "greedy", "profam", True, 8.0),
    ("bandit_thompson_eb8", "thompson", "profam", True, 8.0),
    # Bandit strategies with EB=16
    ("bandit_greedy_eb16", "greedy", "profam", True, 16.0),
    ("bandit_thompson_eb16", "thompson", "profam", True, 16.0),
    # Thompson + bandit with relative reward
    ("thompson_eb8_bandit_rel", "thompson", "profam", True, 8.0),
    ("thompson_eb16_bandit_rel", "thompson", "profam", True, 16.0),
]

CONFIG_TEMPLATE = """## 2GDZ Campaign: {scaffold} / {strategy}
##
## Target: 2GDZ_15PGDH (15-Hydroxyprostaglandin Dehydrogenase)
## Scaffold: {scaffold}
## Strategy: {strategy}
## boltz_ensemble_n: 6, max_cycles: 1000

initial_fasta: {fasta_path}

profam_checkpoint_dir: ".profam_repo/model_checkpoints/profam-1"
profam_sampler: single
profam_num_samples: 1
profam_max_tokens: 8192
profam_max_generated_length: null
profam_temperature: 0.8
profam_top_p: 0.95

energy_config: configs/energy/example_energy_boltz_ipsae_2GDZ_15PGDH.yaml

f_inject: 0.25
max_cycles: 1000
output_dir: outputs/2gdz_campaign/{scaffold}/{strategy}
softmax_temperature: 0.01
random_seed: {seed}
run_on_modal: false
output_frequency: 1
enforce_template: false
sample_with_reinsertion: false
reinject_initial: true
n_memory: 0
elitism: true
accept_only_improvement: true

# Proposal method
proposal_method: {proposal_method}
max_mutations: 5
freeze_prompt: false

# Selection strategy
selection_strategy: {selection_strategy}
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: {exploit_bias}
thompson_temperature_bins: null

# Proposal bandit
thompson_proposal_bandit: {thompson_proposal_bandit}
proposal_bandit_prior_alpha: 2.0
proposal_bandit_prior_beta: 2.0
proposal_bandit_relative_reward: {relative_reward}

# Diverse arm pruning
thompson_max_arms: 10
thompson_max_identity: 0.95

deduplicate_sequences: true
boltz_ensemble_n: 6
"""

def generate_configs():
    """Generate all config files."""
    import hashlib

    configs_generated = []

    for scaffold_name, fasta_path in SCAFFOLDS:
        for strategy_name, selection_strategy, proposal_method, use_bandit, exploit_bias in STRATEGIES:
            # Generate deterministic seed from scaffold + strategy
            seed_str = f"{scaffold_name}_{strategy_name}_2gdz"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % 1000000

            # Determine relative_reward (only for thompson_ebX_bandit_rel strategies)
            relative_reward = "true" if "bandit_rel" in strategy_name else "false"

            # Format exploit_bias
            eb_str = f"{exploit_bias}" if exploit_bias else "5.0"

            config_content = CONFIG_TEMPLATE.format(
                scaffold=scaffold_name,
                strategy=strategy_name,
                fasta_path=fasta_path,
                seed=seed,
                proposal_method=proposal_method,
                selection_strategy=selection_strategy,
                exploit_bias=eb_str,
                thompson_proposal_bandit=str(use_bandit).lower(),
                relative_reward=relative_reward,
            )

            config_path = CONFIG_DIR / f"2gdz_{scaffold_name}_{strategy_name}.yaml"
            config_path.write_text(config_content)
            configs_generated.append(config_path)

    return configs_generated

if __name__ == "__main__":
    configs = generate_configs()
    print(f"Generated {len(configs)} config files in {CONFIG_DIR}")

    # Print summary
    print(f"\nScaffolds: {len(SCAFFOLDS)}")
    for name, path in SCAFFOLDS:
        print(f"  - {name}: {path}")

    print(f"\nStrategies: {len(STRATEGIES)}")
    for name, sel, prop, bandit, eb in STRATEGIES:
        print(f"  - {name}: selection={sel}, proposal={prop}, bandit={bandit}, eb={eb}")

    print(f"\nTotal configs: {len(SCAFFOLDS)} × {len(STRATEGIES)} = {len(configs)}")
