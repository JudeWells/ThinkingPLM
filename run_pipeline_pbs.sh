#!/bin/bash
# PBS batch script to run the ProFam + BAGEL generative pipeline.
#
# Usage:
#   qsub run_pipeline_pbs.sh -v CONFIG=path/to/config.yaml
#
# Or edit the CONFIG variable below directly.
#
# You should customise the resource requests (#PBS -l ...) and any module
# or Conda environment activation commands for your cluster.

#PBS -N profam_bagel_pipeline
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o profam_bagel_pipeline.$PBS_JOBID.out

set -euo pipefail

# ----------------------- user configuration -----------------------

# Path to the YAML config describing the pipeline run.
: "${CONFIG:=pipeline_config.yaml}"

# Optional: path to your conda env or module setup.
# Example (uncomment and adapt to your system):
# module load anaconda/3
# source activate my_protein_design_env
# or:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate profam_bagel

# -------------------------- job execution -------------------------

echo "Job started on host $(hostname) at $(date)"

cd "${PBS_O_WORKDIR:-$PWD}"

echo "Running pipeline with CONFIG=${CONFIG}"
python run_profam_bagel_pipeline.py --config "${CONFIG}"

echo "Job finished at $(date)"

