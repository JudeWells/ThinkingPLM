#!/usr/bin/env python3
"""Create SLURM scripts for greedy_proposal_bandit configs."""

import os
import glob

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
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
    script_dir = "/lus/lfs1aip2/projects/u6bz/jude/ThinkingPLM"
    config_dir = os.path.join(script_dir, "configs/pipelines")
    slurm_dir = os.path.join(script_dir, "slurm_scripts")
    
    # Find all greedy_proposal_bandit configs
    pattern = os.path.join(config_dir, "bench_*_greedy_proposal_bandit.yaml")
    configs = sorted(glob.glob(pattern))
    
    all_slurm_scripts = []
    
    for config_path in configs:
        config_name = os.path.basename(config_path)
        # Extract job name from config name (remove bench_ prefix and .yaml suffix)
        base_name = config_name.replace(".yaml", "")
        job_name = base_name.replace("bench_", "")
        
        # SLURM script filename
        slurm_name = f"run_{base_name}.sh"
        slurm_path = os.path.join(slurm_dir, slurm_name)
        
        # Generate SLURM content
        slurm_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            config_name=config_name,
        )
        
        # Write SLURM script
        with open(slurm_path, "w") as f:
            f.write(slurm_content)
        os.chmod(slurm_path, 0o755)
        print(f"Created: {slurm_name}")
        
        all_slurm_scripts.append(slurm_name)
    
    # Create master submission script
    submit_all_path = os.path.join(slurm_dir, "submit_all_greedy_proposal_bandit.sh")
    with open(submit_all_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all greedy proposal bandit benchmark jobs\n\n")
        f.write("SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n")
        for script in all_slurm_scripts:
            f.write(f'sbatch "$SCRIPT_DIR/{script}"\n')
        f.write('\necho "Submitted all greedy proposal bandit benchmark jobs"\n')
    os.chmod(submit_all_path, 0o755)
    
    print(f"\nCreated {len(all_slurm_scripts)} SLURM scripts")
    print(f"Run 'slurm_scripts/submit_all_greedy_proposal_bandit.sh' to submit all jobs")

if __name__ == "__main__":
    main()
