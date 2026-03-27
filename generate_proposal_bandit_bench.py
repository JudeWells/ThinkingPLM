#!/usr/bin/env python3
"""Generate benchmark configs and SLURM scripts for proposal bandit experiments.

Generates configs for all (target × scaffold × condition) combinations,
including a random_init scaffold that uses no initial FASTA.
"""

import os
import random

TARGETS = {
    "2GDZ_15PGDH": "example_energy_boltz_ipsae_2GDZ_15PGDH.yaml",
    "2VSM_nipah": "energy_boltz_ipsae_2VSM_nipah.yaml",
    "4OYD_epstein_barr": "energy_boltz_ipsae_4OYD_epstein_barr.yaml",
    "4ZQK_PD-L1": "energy_boltz_ipsae_4ZQK_PD-L1.yaml",
    "1TNF_TNF_alpha": "energy_boltz_ipsae_1TNF_TNF_alpha.yaml",
    "1YCR_MDM2": "energy_boltz_ipsae_1YCR_MDM2.yaml",
}

SCAFFOLDS = {
    "3helix": "initial_3helix_scaffold_pdb_1LQZ",
    "4D5": "initial_sequence_human_single_domain_antibody_4D5.fasta",
    "ankyrin": "initial_sequence_ankyrin_repeat.fasta",
    "nanobody": "initial_sequence_nanobody_like.fasta",
    "random_init": None,
}

# Condition definitions with Thompson sampling / proposal bandit params.
# Each condition maps to the fields that differ from the SHARED base.
CONDITIONS = {
    "proposal_bandit": {
        "selection_strategy": "thompson",
        "thompson_m_samples": 5,
        "thompson_exploit_bias": 2.0,
        "thompson_temperature_bins": [0.6, 0.8, 1.0],
        "thompson_discount": 0.95,
        "thompson_proposal_bandit": True,
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "proposal_bandit_eb5": {
        "selection_strategy": "thompson",
        "thompson_m_samples": 5,
        "thompson_exploit_bias": 5.0,
        "thompson_temperature_bins": [0.6, 0.8, 1.0],
        "thompson_discount": 0.95,
        "thompson_proposal_bandit": True,
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "proposal_bandit_eb5_d1": {
        "selection_strategy": "thompson",
        "thompson_m_samples": 5,
        "thompson_exploit_bias": 5.0,
        "thompson_temperature_bins": [0.6, 0.8, 1.0],
        "thompson_discount": 1.0,
        "thompson_proposal_bandit": True,
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "proposal_bandit_eb10_d1": {
        "selection_strategy": "thompson",
        "thompson_m_samples": 5,
        "thompson_exploit_bias": 10.0,
        "thompson_temperature_bins": [0.6, 0.8, 1.0],
        "thompson_discount": 1.0,
        "thompson_proposal_bandit": True,
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "greedy_proposal_bandit": {
        "selection_strategy": "greedy",
        "thompson_m_samples": 1,
        "thompson_exploit_bias": 5.0,
        "thompson_temperature_bins": None,
        "thompson_discount": 1.0,
        "thompson_proposal_bandit": True,
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "profam_update": {
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "profam_frozen": {
        "proposal_method": "profam",
        "freeze_prompt": True,
    },
    "random_update": {
        "proposal_method": "random_mutation",
        "freeze_prompt": False,
    },
}

SHARED = {
    "profam_checkpoint_dir": ".profam_repo/model_checkpoints/profam-1",
    "profam_sampler": "single",
    "profam_num_samples": 1,
    "profam_max_tokens": 8192,
    "profam_max_generated_length": None,
    "profam_temperature": 0.8,
    "profam_top_p": 0.95,
    "f_inject": 0.25,
    "max_cycles": 100,
    "softmax_temperature": 0.01,
    "run_on_modal": False,
    "output_frequency": 1,
    "enforce_template": False,
    "sample_with_reinsertion": False,
    "reinject_initial": True,
    "n_memory": 0,
    "elitism": True,
    "accept_only_improvement": True,
    "max_mutations": 5,
    "thompson_reward_term": "ipSAE",
    "deduplicate_sequences": True,
}

RANDOM_INIT_MAX_RESIDUES = 80

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=bench_{job_tag}
#SBATCH --output=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.out
#SBATCH --error=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00

WORKDIR=/projects/u6bz/jude/ThinkingPLM

mkdir -p "$WORKDIR/logs"

source ~/.bashrc
conda activate profam_bagel

cd "$WORKDIR"
python "$WORKDIR/run_profam_bagel_pipeline.py" --config "$WORKDIR/configs/pipelines/{config_name}"

echo "Job completed at $(date)"
"""


def _yaml_val(v):
    """Format a Python value for YAML output."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def _write_config(path, header_lines, fields):
    """Write a YAML config file with a comment header."""
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("\n")
        for key, val in fields.items():
            f.write(f"{key}: {_yaml_val(val)}\n")


def main():
    random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "configs/pipelines")
    slurm_dir = os.path.join(script_dir, "slurm_scripts")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(slurm_dir, exist_ok=True)

    all_slurm_scripts = []
    generated_configs = []

    for target, energy_config in TARGETS.items():
        for scaffold, initial_fasta in SCAFFOLDS.items():
            for cond_name, cond_params in CONDITIONS.items():
                random_seed = random.randint(100000, 999999)

                config_name = f"bench_{target}_{scaffold}_{cond_name}.yaml"
                config_path = os.path.join(config_dir, config_name)

                fields = dict(SHARED)
                fields["energy_config"] = f"configs/energy/{energy_config}"
                fields["output_dir"] = f"outputs/bench/{target}/{scaffold}/{cond_name}"
                fields["random_seed"] = random_seed

                if initial_fasta is None:
                    fields["random_init"] = True
                    fields["random_init_max_residues"] = RANDOM_INIT_MAX_RESIDUES
                else:
                    fields["initial_fasta"] = f"configs/sequences/{initial_fasta}"

                fields.update(cond_params)

                is_thompson = cond_params.get("selection_strategy") == "thompson"
                header_lines = [
                    f"## Benchmark: {target} / {scaffold} / {cond_name}",
                    f"##",
                    f"## Target: {target}, Init: {scaffold}, Condition: {cond_name}",
                ]
                if is_thompson:
                    header_lines.append(
                        "## Thompson sampling with proposal bandit: learns to choose between profam and random_mutation"
                    )
                else:
                    pm = cond_params.get("proposal_method", "profam")
                    fp = cond_params.get("freeze_prompt", False)
                    header_lines.append(f"## proposal_method: {pm}, freeze_prompt: {fp}")

                _write_config(config_path, header_lines, fields)
                generated_configs.append(config_name)

                job_tag = f"{target}_{scaffold}_{cond_name}"
                slurm_name = f"run_bench_{job_tag}.sh"
                slurm_path = os.path.join(slurm_dir, slurm_name)

                slurm_content = SLURM_TEMPLATE.format(
                    job_tag=job_tag,
                    config_name=config_name,
                )
                with open(slurm_path, "w") as f:
                    f.write(slurm_content)
                os.chmod(slurm_path, 0o755)

                all_slurm_scripts.append(slurm_name)

    submit_all_path = os.path.join(slurm_dir, "submit_all_bench.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all benchmark jobs\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        for script in all_slurm_scripts:
            f.write(f'sbatch "$SCRIPT_DIR/{script}"\n')
        f.write(f'\necho "Submitted {len(all_slurm_scripts)} benchmark jobs"\n')
    os.chmod(submit_all_path, 0o755)

    print(f"Generated {len(generated_configs)} benchmark configs and SLURM scripts")
    for c in generated_configs:
        print(f"  {c}")
    print(f"\nRun 'slurm_scripts/submit_all_bench.sh' to submit all jobs")


if __name__ == "__main__":
    main()
