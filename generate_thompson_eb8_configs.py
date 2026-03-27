#!/usr/bin/env python3
"""
Generate configs for Thompson prompt selection with EB=8 + relative reward proposal bandit.

Like greedy_diverse_rel but with Thompson prompt selection instead of greedy.
"""

from pathlib import Path

TARGETS = {
    "2GDZ_15PGDH": "configs/energy/example_energy_boltz_ipsae_2GDZ_15PGDH.yaml",
    "4OYD_epstein_barr": "configs/energy/energy_boltz_ipsae_4OYD_epstein_barr.yaml",
    "2VSM_nipah": "configs/energy/energy_boltz_ipsae_2VSM_nipah.yaml",
    "4ZQK_PD-L1": "configs/energy/energy_boltz_ipsae_4ZQK_PD-L1.yaml",
}

SCAFFOLDS = {
    "3helix": "configs/sequences/initial_3helix_scaffold_pdb_1LQZ",
    "4D5": "configs/sequences/initial_sequence_human_single_domain_antibody_4D5.fasta",
    "ankyrin": "configs/sequences/initial_sequence_ankyrin_repeat.fasta",
    "nanobody": "configs/sequences/initial_sequence_nanobody_like.fasta",
}

CONFIG_TEMPLATE = """## Benchmark: Thompson prompt (EB=8) + ProFam/Random bandit — {target} / {scaffold}
##
## Key features:
## - selection_strategy: thompson (not greedy) for prompt selection
## - thompson_exploit_bias: 8.0 (higher exploitation than default)
## - thompson_max_arms: 10 (diverse arm pruning)
## - proposal_bandit with relative reward (profam vs random)
## - boltz_ensemble_n: 3 for noise reduction

initial_fasta: {initial_fasta}

profam_checkpoint_dir: ".profam_repo/model_checkpoints/profam-1"
profam_sampler: single
profam_num_samples: 1
profam_max_tokens: 8192
profam_max_generated_length: null
profam_temperature: 0.8
profam_top_p: 0.95

energy_config: {energy_config}

f_inject: 0.25
max_cycles: 100
output_dir: outputs/bench_ensemble/{target}/{scaffold}/thompson_eb8_bandit_rel
softmax_temperature: 0.01
random_seed: 223724
run_on_modal: false
output_frequency: 1
enforce_template: false
sample_with_reinsertion: false
reinject_initial: true
n_memory: 0
elitism: true
accept_only_improvement: true

# ProFam as default, but bandit can switch to random_mutation
proposal_method: profam
max_mutations: 5
freeze_prompt: false

# Thompson sampling for prompt selection with high exploit bias
selection_strategy: thompson
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: 8.0
thompson_temperature_bins: null

# Proposal bandit: Thompson over profam vs random_mutation with relative reward
thompson_proposal_bandit: true
proposal_bandit_prior_alpha: 2.0
proposal_bandit_prior_beta: 2.0
proposal_bandit_relative_reward: true

# Diverse arm pruning
thompson_max_arms: 10
thompson_max_identity: 0.95

deduplicate_sequences: true
boltz_ensemble_n: 3
"""

def main():
    config_dir = Path("configs/pipelines")
    config_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for target, energy_config in TARGETS.items():
        for scaffold, initial_fasta in SCAFFOLDS.items():
            config_content = CONFIG_TEMPLATE.format(
                target=target,
                scaffold=scaffold,
                initial_fasta=initial_fasta,
                energy_config=energy_config,
            )

            config_name = f"ens_{target}_{scaffold}_thompson_eb8_bandit_rel.yaml"
            config_path = config_dir / config_name

            with open(config_path, "w") as f:
                f.write(config_content)

            print(f"Created: {config_path}")
            count += 1

    print(f"\nGenerated {count} config files.")

if __name__ == "__main__":
    main()
