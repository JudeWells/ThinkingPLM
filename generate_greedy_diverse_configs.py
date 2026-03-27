#!/usr/bin/env python3
"""Generate pipeline configs for greedy_diverse benchmark runs."""

import random
from pathlib import Path

# Target -> energy config mapping
TARGETS = {
    "2GDZ_15PGDH": "configs/energy/example_energy_boltz_ipsae_2GDZ_15PGDH.yaml",
    "2VSM_nipah": "configs/energy/energy_boltz_ipsae_2VSM_nipah.yaml",
    "4OYD_epstein_barr": "configs/energy/energy_boltz_ipsae_4OYD_epstein_barr.yaml",
    "4ZQK_PD-L1": "configs/energy/energy_boltz_ipsae_4ZQK_PD-L1.yaml",
}

# Scaffold -> initial FASTA mapping
SCAFFOLDS = {
    "3helix": "configs/sequences/initial_3helix_scaffold_pdb_1LQZ",
    "4D5": "configs/sequences/initial_sequence_human_single_domain_antibody_4D5.fasta",
    "ankyrin": "configs/sequences/initial_sequence_ankyrin_repeat.fasta",
    "nanobody": "configs/sequences/initial_sequence_nanobody_like.fasta",
}

# Extra scaffold for 2GDZ only
EXTRA_2GDZ_SCAFFOLDS = {
    "rank2_bindcraft": "configs/sequences/initial_rank2_bindcraft_15PGDH_l107_s438837_mpnn2.fasta",
}

CONFIG_TEMPLATE = """\
## Benchmark: Greedy selection with diverse arm pruning — {target} / {scaffold}
##
## Uses thompson_max_arms to limit the number of arms, keeping only the
## top-K most diverse sequences (by sequence identity threshold).
## This reduces exploration overhead while maintaining diversity.

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
output_dir: outputs/bench/{target}/{scaffold}/greedy_diverse
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

proposal_method: profam
max_mutations: 5
freeze_prompt: false

# Selection: greedy with proposal bandit
selection_strategy: greedy
thompson_m_samples: 1
thompson_reward_term: ipSAE
thompson_exploit_bias: 5.0
thompson_temperature_bins: null
thompson_proposal_bandit: true

# NEW: Diverse arm pruning
thompson_max_arms: 10
thompson_max_identity: 0.95

deduplicate_sequences: true
"""


def main():
    config_dir = Path("configs/pipelines")
    config_dir.mkdir(parents=True, exist_ok=True)

    configs_created = []

    for target, energy_config in TARGETS.items():
        # Use standard scaffolds for all targets
        scaffolds = dict(SCAFFOLDS)

        # Add extra scaffold for 2GDZ
        if target == "2GDZ_15PGDH":
            scaffolds.update(EXTRA_2GDZ_SCAFFOLDS)

        for scaffold, initial_fasta in scaffolds.items():
            seed = random.randint(100000, 999999)

            config_content = CONFIG_TEMPLATE.format(
                target=target,
                scaffold=scaffold,
                initial_fasta=initial_fasta,
                energy_config=energy_config,
                seed=seed,
            )

            config_name = f"bench_{target}_{scaffold}_greedy_diverse.yaml"
            config_path = config_dir / config_name

            config_path.write_text(config_content)
            configs_created.append(config_path)
            print(f"Created: {config_path}")

    print(f"\nTotal configs created: {len(configs_created)}")
    return configs_created


if __name__ == "__main__":
    main()
