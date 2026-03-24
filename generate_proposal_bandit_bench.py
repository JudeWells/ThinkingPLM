#!/usr/bin/env python3
"""Generate benchmark configs and SLURM scripts for proposal bandit experiments."""

import os
import random

# Define targets with their energy configs
TARGETS = {
    "2GDZ_15PGDH": "example_energy_boltz_ipsae_2GDZ_15PGDH.yaml",
    "2VSM_nipah": "energy_boltz_ipsae_2VSM_nipah.yaml",
    "4OYD_epstein_barr": "energy_boltz_ipsae_4OYD_epstein_barr.yaml",
    "4ZQK_PD-L1": "energy_boltz_ipsae_4ZQK_PD-L1.yaml",
}

# Define initial sequences (scaffolds) with their fasta files
SCAFFOLDS = {
    "3helix": "initial_3helix_scaffold_pdb_1LQZ",
    "4D5": "initial_sequence_human_single_domain_antibody_4D5.fasta",
    "ankyrin": "initial_sequence_ankyrin_repeat.fasta",
    "nanobody": "initial_sequence_nanobody_like.fasta",
}

CONFIG_TEMPLATE = """## Benchmark: {target} / {scaffold} / proposal_bandit
##
## Target: {target}, Init: {scaffold}, Condition: proposal_bandit
## Thompson sampling with proposal bandit: learns to choose between profam and random_mutation

profam_checkpoint_dir: .profam_repo/model_checkpoints/profam-1
profam_sampler: single
profam_num_samples: 1
profam_max_tokens: 8192
profam_max_generated_length: null
profam_temperature: 0.8
profam_top_p: 0.95

f_inject: 0.25
max_cycles: 100
softmax_temperature: 0.01
run_on_modal: false
output_frequency: 1
enforce_template: false
sample_with_reinsertion: false
reinject_initial: true
n_memory: 0
elitism: true
accept_only_improvement: true
max_mutations: 5

initial_fasta: configs/sequences/{initial_fasta}
energy_config: configs/energy/{energy_config}
output_dir: outputs/bench/{target}/{scaffold}/proposal_bandit_eb5
random_seed: {random_seed}

proposal_method: profam
freeze_prompt: false

# Thompson sampling settings
selection_strategy: thompson
thompson_m_samples: 5
thompson_reward_term: ipSAE
thompson_exploit_bias: 5.0
thompson_temperature_bins: [0.6, 0.8, 1.0]
thompson_discount: 0.95
thompson_proposal_bandit: true
deduplicate_sequences: true
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=bench_{target}_{scaffold}_bandit_eb5
#SBATCH --output=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.out
#SBATCH --error=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00

WORKDIR=/projects/u6bz/jude/ThinkingPLM

# Create logs directory if it doesn't exist
mkdir -p "$WORKDIR/logs"

# Initialize conda and activate environment
source ~/.bashrc
conda activate profam_bagel

# Run the pipeline
cd "$WORKDIR"
python "$WORKDIR/run_profam_bagel_pipeline.py" --config "$WORKDIR/configs/pipelines/{config_name}"

echo "Job completed at $(date)"
"""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "configs/pipelines")
    slurm_dir = os.path.join(script_dir, "slurm_scripts")

    os.makedirs(slurm_dir, exist_ok=True)

    all_slurm_scripts = []

    for target, energy_config in TARGETS.items():
        for scaffold, initial_fasta in SCAFFOLDS.items():
            # Generate unique random seed
            random_seed = random.randint(100000, 999999)

            # Config filename
            config_name = f"bench_{target}_{scaffold}_proposal_bandit_eb5.yaml"
            config_path = os.path.join(config_dir, config_name)

            # Generate config content
            config_content = CONFIG_TEMPLATE.format(
                target=target,
                scaffold=scaffold,
                initial_fasta=initial_fasta,
                energy_config=energy_config,
                random_seed=random_seed,
            )

            # Write config file
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"Created config: {config_name}")

            # SLURM script filename
            slurm_name = f"run_bench_{target}_{scaffold}_proposal_bandit_eb5.sh"
            slurm_path = os.path.join(slurm_dir, slurm_name)

            # Generate SLURM content
            slurm_content = SLURM_TEMPLATE.format(
                target=target,
                scaffold=scaffold,
                config_name=config_name,
            )

            # Write SLURM script
            with open(slurm_path, "w") as f:
                f.write(slurm_content)
            os.chmod(slurm_path, 0o755)
            print(f"Created SLURM script: {slurm_name}")

            all_slurm_scripts.append(slurm_name)

    # Create a master submission script
    submit_all_path = os.path.join(slurm_dir, "submit_all_proposal_bandit_eb5.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all proposal bandit benchmark jobs\n\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n")
        for script in all_slurm_scripts:
            f.write(f'sbatch "$SCRIPT_DIR/{script}"\n')
        f.write('\necho "Submitted all proposal bandit benchmark jobs"\n')
    os.chmod(submit_all_path, 0o755)

    print(f"\nCreated {len(all_slurm_scripts)} configs and SLURM scripts")
    print(f"Run 'slurm_scripts/submit_all_proposal_bandit_eb5.sh' to submit all jobs")

if __name__ == "__main__":
    main()
