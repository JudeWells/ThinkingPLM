# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Cluster State (2026-04-07)

### Nodes

| Node | IP | Instance | GPUs | Disk Free | Env |
|------|-----|----------|------|-----------|-----|
| Node 1 | 3.14.255.102 | p4d.24xlarge | 8x A100-40GB | ~83GB | profam_bagel |
| Node 2 | 18.191.140.159 | p4d.24xlarge | 8x A100-40GB | ~336GB | profam_bagel |
| Node 3 | 3.147.78.97 | p4d.24xlarge | 8x A100-40GB | ~112GB | profam_bagel |
| Node 4 | 3.147.71.187 | p5.4xlarge | 1x H100-80GB | ~82GB | profam_bagel |

SSH: `ssh -i ~/.ssh/gpu-ml-key.pem ubuntu@<IP>`

### Active Experiments

**1. Scaffold Comparison rep2 (15PGDH / 2GDZ target, ESMFold+LIS)**
- Config generator: `generate_scaffold_comparison_configs.py --replicate 2`
- Configs: `configs/pipelines/scaffold_comparison/*_rep2.yaml`
- 4 scaffolds (affibody_2B87, hairpin, beta_sheet_1E0L, bindcraft_15PGDH) × 4 methods (random_greedy, proposal_bandit, bandit_grpo, bandit_bt)
- Total evaluations per run: 5400 (1-sample methods: 5400 cycles, 12-sample methods: 450 cycles)
- Node 1: affibody_2B87 (4 methods) + hairpin (4 methods) — all 8 GPUs
- Node 2: beta_sheet_1E0L (4 methods) + bindcraft_15PGDH (4 methods) — GPUs 1-3, 5-7 (GPUs 0,4 freed after random_greedy completed)

**2. Multi-target Benchmark (5 targets, ESMFold+LIS)**
- Config generator: `generate_grpo_multi_target_bench.py`
- Configs: `configs/pipelines/multi_target_bench/*.yaml`
- 5 targets × 4 scaffolds × 3 methods = 60 configs total
- Targets: 2VSM_nipah, 4OYD_epstein_barr, 4ZQK_PD-L1, 1TNF_TNF_alpha, 1YCR_MDM2
- Scaffolds: 4D5, ankyrin, nanobody, random_init
- Methods: random_greedy, proposal_bandit, bandit_grpo
- Node 2 GPUs 0,4: TNF_alpha random_init (bandit_grpo, proposal_bandit)
- Node 3: all 8 GPUs — TNF_alpha (4D5, ankyrin, nanobody configs)
- Node 4: 1 GPU — TNF_alpha random_init random_greedy
- 13/60 configs launched so far; remaining 47 to backfill as GPUs free up

### Disk Maintenance
- `.pae` files are the main disk consumer (~1MB each, 65GB+ at scale). Delete periodically.
- `.cif` files are secondary (~250KB each). Safe to delete.
- Old sweep dirs (`sweep_v5/`, `sweep_v6/`, `bt_*/`) can be deleted after results are captured.

## Quick Start for Agents (AWS/Cloud)

**Goal:** Get the pipeline running as fast as possible on a fresh cloud instance.

### Recommended AWS Instance Types

| Use Case | Instance Type | GPU | vCPU | Memory | Notes |
|----------|---------------|-----|------|--------|-------|
| **Local GPU runs** | `g5.xlarge` | A10G (24GB) | 4 | 16 GB | Best cost/performance for Boltz/ESMFold |
| **Large batches** | `g5.2xlarge` | A10G (24GB) | 8 | 32 GB | More CPU for parallel preprocessing |
| **Modal offload** | `t3.medium` | None | 2 | 4 GB | Cheapest; all GPU work on Modal cloud |
| **Development** | `g4dn.xlarge` | T4 (16GB) | 4 | 16 GB | Budget option, slower folding |

**AMI:** Use "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" or similar with CUDA pre-installed.

### One-Command Setup (Fresh Instance)

```bash
# 1. Clone and enter repo
git clone https://github.com/JudeWells/ThinkingPLM.git && cd ThinkingPLM

# 2. Run cloud setup (installs miniconda if needed, creates env, downloads model)
chmod +x setup_cloud.sh && ./setup_cloud.sh

# 3. Activate and run
source ~/.bashrc && conda activate profam_bagel
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml
```

### Step-by-Step Setup (If One-Command Fails)

```bash
# 1. Ensure conda is available
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
fi

# 2. Clone repo
git clone https://github.com/JudeWells/ThinkingPLM.git
cd ThinkingPLM

# 3. Create environment
chmod +x setup_environment.sh && ./setup_environment.sh

# 4. Activate environment
conda activate profam_bagel

# 5. Download ProFam model checkpoint (~3GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='.profam_repo/model_checkpoints/profam-1')"

# 6. (Optional) Setup Modal for cloud GPU
modal token new
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx

# 7. Verify installation
python -c "import bagel; import profam; from src.models.inference import ProFamSampler; print('OK')"
```

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### Common Cloud Issues

| Problem | Solution |
|---------|----------|
| `conda: command not found` | Install miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b` |
| `CUDA out of memory` | Reduce `profam_num_samples` in config, or use Modal (`run_on_modal: true`) |
| `No CUDA GPUs available` | Check `nvidia-smi`; may need `sudo nvidia-smi -pm 1` or driver install |
| `libcudnn.so not found` | Install cuDNN: `conda install cudnn -c conda-forge` |
| `Permission denied` on model download | Set `HF_TOKEN` env var or run `huggingface-cli login` |
| Slow first run | Normal—ProFam/Boltz models download on first use (~10-15 min) |

### Running Modes Quick Reference

```bash
# Modal cloud (recommended for most users) - GPU work happens on Modal's servers
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml

# Local GPU (requires g5/g4dn instance)
# Set run_on_modal: false in config YAML first
export MODEL_DIR=~/.cache/bagel/models
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_berlin_hairpin_start.yaml

# Check a run's progress
tail -f outputs/campaign_name/cycle_stats.json
```

---

## Project Overview

ProFam + BAGEL generative protein design pipeline. Iteratively generates protein sequences with a language model (ProFam), evaluates them via structure prediction and energy scoring (BAGEL/Boltz), and selects promising candidates for the next cycle. Current target: designing protein binders against **15-PGDH** (15-Hydroxyprostaglandin Dehydrogenase, PDB: 2GDZ) for the Berlin Bio x AI Hackathon.

## Key Commands

### Environment Setup
```bash
chmod +x setup_environment.sh && ./setup_environment.sh
# Creates conda env "profam_bagel" with Python 3.11, installs BAGEL from GitHub, ProFam from cloned repo
conda activate profam_bagel
```

### Running the Pipeline
```bash
# Modal cloud (set run_on_modal: true in config YAML) — primary mode
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml

# Local (requires GPU)
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_berlin_hairpin_start.yaml

# CLI flags override YAML values
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml --max_cycles 5
```

### Visualization
```bash
# Plot energy curves for all campaigns (dark mode)
python plot_campaigns.py

# Animate best structure per cycle using PyMOL
python animate_campaign.py outputs/campaign12_tiny_barrel_elite
# Options: --width 1200 --height 900 --delay 50
```

### Modal Setup
```bash
modal token new
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

### PyMOL
Binary at `/home/judewells/miniconda3/bin/pymol`. Used headless (`-cq`) for rendering animations.

## Architecture

### Core Files

**`run_profam_bagel_pipeline.py`** (~2400 lines) — the entire pipeline in one file:
- `PipelineConfig` dataclass + `build_arg_parser()` + `merge_config()` — config loading (YAML + CLI merge)
- `load_profam_model()` / `run_profam_generation()` — ProFam model loading and sequence generation
- `build_folding_oracle()` / `build_energy_terms_for_chain()` / `evaluate_sequences_with_bagel()` — BAGEL folding and energy evaluation
- `softmax_from_energies()` / `sample_subset_indices()` / `update_cycle_log()` — sampling, statistics, logging
- `make_energy_summary_plot()` — per-run energy plot (dark mode, cumulative min trace)
- `run_pipeline()` — main loop (generate → fold+score → probabilities → select → elitism/swap → inject)

**`run_profam_bagel_modal_app.py`** — Modal cloud wrapper. Builds a container image, runs `run_pipeline()` remotely, syncs results via Modal Volume.

**`modal_proteinmpnn_score.py`** — Standalone Modal app for SolubleMPNN scoring. Runs in isolated container with numpy 1.x / torch 2.2.1 to avoid dependency conflicts.

**`plot_campaigns.py`** (in `random_scripts/`) — Plots energy term curves for all campaigns. Generates per-term breakdowns and accepted-best plots for campaigns with elitism.

**`animate_campaign.py`** (in `random_scripts/`) — Creates animated GIFs of best structure per cycle using PyMOL. Colors by chain (cyan=binder, orange=target), aligns with `super`, renders both black and white backgrounds.

### Configuration System

All configs live under `configs/` with three subdirectories:
- `configs/pipelines/` — pipeline YAML configs (ProFam settings, cycle count, injection fraction, anti-regression)
- `configs/energy/` — energy YAML configs (folding oracle type and energy terms with weights)
- `configs/sequences/` — initial FASTA sequences for each scaffold

Two YAML files drive each run:
1. **Pipeline config** (e.g., `configs/pipelines/pipeline_campaign6_hairpin_elite.yaml`)
2. **Energy config** (referenced by `energy_config` key, e.g., `configs/energy/example_energy_boltz_ipsae_2GDZ.yaml`)

Energy config structure:
```yaml
folding_oracle:
  type: Boltz           # or ESMFold
  kwargs: {}
energies:
  - type: ipSAEEnergy   # or LISEnergy, PTMEnergy, ChemicalPotentialEnergy, etc.
    kwargs:
      weight: 1.0
      residues:
        GEN: "all"
        B: "all"
```

The pipeline saves a `pipeline_config.json` snapshot to the output folder at the start of each run for reproducibility.

### Multi-Chain Design

For binding design, energy configs specify multiple chains with residue ranges:
- `GEN` chain = the generated sequence (binder)
- Named chains (B, etc.) = target proteins (sequence provided in energy config `target` field)
- Boltz/ESMFold receives all chains joined with `":"` separator

### Anti-Regression Mechanisms

Two features prevent energy from getting worse across cycles:

- **Elitism** (`elitism: true`): Tracks the global best sequence ever seen. Guarantees it position 0 in the injection set (survives token-budget trimming which trims from the end).
- **Conditional swap** (`accept_only_improvement: true`): Only updates the injection set when the new candidate's best energy improves over the previous. Optional simulated annealing via `annealing_initial_temp` / `annealing_decay`.

Both default to `False`. Swap decisions are logged in `cycle_stats.json` (`swap_accepted`, `swap_reason`, `global_elite`).

### Other Key Mechanisms

- **Constrained generation**: `enforce_template: true` forces specific residues via ProFam's logits processor; `false` assigns inf energy on mismatch
- **Memory pooling**: `n_memory > 0` includes sequences from previous N cycles in the selection pool
- **Residue spec notation**: `"0-43"`, `"1,2,5"`, `"0-5,10,20-25"`, `"all"`
- **PDB caching**: structures cached in `~/.cache/profam_bagel/pdb/`
- **ProFam diversity**: Seed is set once at model load (not per cycle), so repeated prompts still produce diverse sequences via stochastic sampling (`do_sample=True`, `top_p`, `temperature`)

## BAGEL Energy System

Energy terms live in the BAGEL library (installed package). The pipeline dynamically dispatches by type name from YAML config.

**Oracle types:** `ESMFold`, `Boltz`, `AlphaFast`

**Key energy terms for binder design:**
- `ipSAEEnergy` — interface quality from PAE matrix (primary binding metric, values ~-0.3 to -0.7 for binders)
- `SolMPNNPerplexityEnergy` — sequence designability via ProteinMPNN (runs on separate Modal app)
- `ChemicalPotentialEnergy` — size penalty: `weight * chemical_potential * |num_residues - target_size|^power`
- `GlobularEnergy` — compactness (minimizes spread of backbone atoms from centroid)
- `LISEnergy`, `PTMEnergy`, `PLDDTEnergy`, `PAEEnergy` — standard structure quality metrics

**Weight calibration:** ipSAE energies are ~0.3-0.5 magnitude. When combining with other terms, scale weights so contributions are similar (e.g., ChemicalPotentialEnergy weight=0.025 with target_size=240 gives ~0.45 at 18 residues deviation).

## Active Campaigns

Campaign configs are in `configs/pipelines/` following pattern `pipeline_campaign{N}_{scaffold}_{features}.yaml`. Current campaigns target 2GDZ with various scaffolds and energy combinations. All use `elitism: true` + `accept_only_improvement: true`.

Initial FASTA files are in `configs/sequences/`: hairpin (80aa), rfd3_inpaint (80aa), nanobody_like (63aa), tiny_barrel (87aa), repebody_7YC0 (258aa), short_helix, 3helix_bundle, ankyrin_repeat, human_single_domain_antibody_4D5.

## Plotting Convention

All plots use `plt.style.use("dark_background")` with black backgrounds and bright colors (`#00bfff` cyan, `#ff6b6b` red, `#00e676` green, `#ffab40` orange). Save with `facecolor="black", edgecolor="none"`.

## Output Structure

Each run writes to `output_dir/`:
- `pipeline_config.json` — full config snapshot for reproducibility
- `cycle_stats.json` — per-cycle energies, similarities, swap decisions, elite tracking
- `sequences_cycle_XXX/` — CIF structures for selected sequences (`sequence_0000.cif` = best)
- `energy_summary.png` — energy vs cycle plot (dark mode, with cumulative min and rejected swap markers)
- `animation_white.gif` / `animation_black.gif` — structural evolution animations (generated by `animate_campaign.py`)

## Dependencies

BAGEL and ProFam have conflicting pins for numpy, matplotlib, and transformers. The setup script installs BAGEL first (stricter pins), then ProFam in editable mode. ProFam is cloned into `.profam_repo/`. SolubleMPNN runs in a separate Modal container to avoid conflicts.

## Environment Variables

- `MODEL_DIR` — path to ESMFold weights (default: `~/.cache/bagel/models`)
- `HF_TOKEN` — HuggingFace token for gated model downloads (Modal uses `huggingface-secret`)
