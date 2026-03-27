#!/usr/bin/env python3
"""
Generate benchmark configs for 4 strategies with Boltz ensemble (n=3).

Strategies:
1. random_greedy: Pure random mutations + greedy prompt selection (elitism)
2. random_thompson: Pure random mutations + Thompson prompt selection (10 arms)
3. bandit_greedy: ProFam+random bandit + greedy prompt selection
4. bandit_thompson: ProFam+random bandit + Thompson prompt selection (10 arms)

All use boltz_ensemble_n=3 for noise reduction.
Results saved to outputs/bench_ensemble/
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

# Strategy 1: Pure random mutations + greedy (elitism)
RANDOM_GREEDY = """## Benchmark: Random mutations + Greedy prompt (elitism) — {target} / {scaffold}
##
## Pure random mutations, always use the best sequence as prompt.
## boltz_ensemble_n=3 for noise reduction.

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
output_dir: outputs/bench_ensemble/{target}/{scaffold}/random_greedy
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

# Pure random mutations
proposal_method: random_mutation
max_mutations: 5
freeze_prompt: false

# Greedy selection (no Thompson sampling of prompt)
selection_strategy: greedy
thompson_proposal_bandit: false

deduplicate_sequences: true
boltz_ensemble_n: 3
"""

# Strategy 2: Pure random mutations + Thompson prompt selection
RANDOM_THOMPSON = """## Benchmark: Random mutations + Thompson prompt selection — {target} / {scaffold}
##
## Pure random mutations with Thompson sampling to select which sequence to mutate.
## thompson_max_arms=10 for diverse arm pruning.
## boltz_ensemble_n=3 for noise reduction.

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
output_dir: outputs/bench_ensemble/{target}/{scaffold}/random_thompson
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

# Pure random mutations
proposal_method: random_mutation
max_mutations: 5
freeze_prompt: false

# Thompson sampling for prompt selection
selection_strategy: thompson
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: 5.0
thompson_temperature_bins: null
thompson_proposal_bandit: false

# Diverse arm pruning (keep top 10 diverse arms)
thompson_max_arms: 10
thompson_max_identity: 0.95

deduplicate_sequences: true
boltz_ensemble_n: 3
"""

# Strategy 3: ProFam+random bandit + greedy prompt selection
BANDIT_GREEDY = """## Benchmark: ProFam+Random bandit + Greedy prompt — {target} / {scaffold}
##
## Thompson bandit chooses between ProFam and random mutations.
## Greedy prompt selection (always use best sequence).
## Relative reward for proposal bandit.
## boltz_ensemble_n=3 for noise reduction.

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
output_dir: outputs/bench_ensemble/{target}/{scaffold}/bandit_greedy
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

# Greedy prompt selection
selection_strategy: greedy
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: 5.0
thompson_temperature_bins: null

# Proposal bandit: Thompson over profam vs random_mutation
thompson_proposal_bandit: true
proposal_bandit_prior_alpha: 2.0
proposal_bandit_prior_beta: 2.0
proposal_bandit_relative_reward: true

# Diverse arm pruning (for proposal bandit arms)
thompson_max_arms: 10
thompson_max_identity: 0.95

deduplicate_sequences: true
boltz_ensemble_n: 3
"""

# Strategy 4: ProFam+random bandit + Thompson prompt selection
BANDIT_THOMPSON = """## Benchmark: ProFam+Random bandit + Thompson prompt — {target} / {scaffold}
##
## Thompson bandit chooses between ProFam and random mutations.
## Thompson sampling to select which sequence to use as prompt.
## Relative reward for proposal bandit.
## thompson_max_arms=10 for diverse arm pruning.
## boltz_ensemble_n=3 for noise reduction.

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
output_dir: outputs/bench_ensemble/{target}/{scaffold}/bandit_thompson
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

# Thompson sampling for prompt selection
selection_strategy: thompson
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: 5.0
thompson_temperature_bins: null

# Proposal bandit: Thompson over profam vs random_mutation
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

STRATEGIES = {
    "random_greedy": RANDOM_GREEDY,
    "random_thompson": RANDOM_THOMPSON,
    "bandit_greedy": BANDIT_GREEDY,
    "bandit_thompson": BANDIT_THOMPSON,
}

def main():
    config_dir = Path("configs/pipelines")
    config_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for target, energy_config in TARGETS.items():
        for scaffold, initial_fasta in SCAFFOLDS.items():
            for strategy_name, template in STRATEGIES.items():
                config_content = template.format(
                    target=target,
                    scaffold=scaffold,
                    initial_fasta=initial_fasta,
                    energy_config=energy_config,
                )

                config_name = f"ens_{target}_{scaffold}_{strategy_name}.yaml"
                config_path = config_dir / config_name

                with open(config_path, "w") as f:
                    f.write(config_content)

                print(f"Created: {config_path}")
                count += 1

    print(f"\nGenerated {count} config files (4 strategies x 4 targets x 4 scaffolds).")

if __name__ == "__main__":
    main()
