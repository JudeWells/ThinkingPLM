# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProFam + BAGEL generative protein design pipeline. Iteratively generates protein sequences with a language model (ProFam), evaluates them via structure prediction and energy scoring (BAGEL/ESMFold), and selects promising candidates for the next cycle.

## Key Commands

### Environment Setup
```bash
chmod +x setup_environment.sh && ./setup_environment.sh
# Creates conda env "profam_bagel" with Python 3.11, installs BAGEL from GitHub, ProFam from cloned repo
conda activate profam_bagel
```

### Running the Pipeline
```bash
# Local (requires GPU)
python run_profam_bagel_pipeline.py --config pipeline_config_1.yaml

# Modal cloud (set run_on_modal: true in config YAML)
python run_profam_bagel_pipeline.py --config pipeline_config_1.yaml

# PBS cluster
qsub run_pipeline_pbs.sh -v CONFIG=pipeline_config_1.yaml

# CLI flags override YAML values
python run_profam_bagel_pipeline.py --config pipeline_config_1.yaml --max_cycles 5 --profam_num_samples 20
```

### Modal Setup
```bash
modal token new
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

## Architecture

The codebase is two files plus configuration:

**`run_profam_bagel_pipeline.py`** (~2150 lines) — the entire pipeline in one file:
- `PipelineConfig` dataclass + `build_arg_parser()` + `merge_config()` — config loading (YAML + CLI merge)
- `load_profam_model()` / `run_profam_generation()` — ProFam model loading and sequence generation
- `build_folding_oracle()` / `build_energy_terms_for_chain()` / `evaluate_sequences_with_bagel()` — BAGEL folding and energy evaluation
- `download_pdb_cif()` / `extract_chain_from_cif()` / `_load_structure_from_spec()` — PDB structure handling
- `softmax_from_energies()` / `sample_subset_indices()` / `compute_avg_sequence_similarity()` — sampling and statistics
- `run_pipeline()` — main loop (6 steps per cycle: generate → fold+score → probabilities → select → log → inject)
- `main()` — entry point

**`run_profam_bagel_modal_app.py`** — Modal cloud wrapper. Builds a container image with dependencies, runs `run_pipeline()` remotely, syncs results via Modal Volume.

### Configuration System

Two YAML files drive each run:
1. **Pipeline config** (e.g., `pipeline_config_1.yaml`) — ProFam settings, cycle count, injection fraction, output dir
2. **Energy config** (referenced by `energy_config` key) — folding oracle type and energy terms with weights

Energy config structure:
```yaml
folding_oracle:
  type: ESMFold
  kwargs: { use_modal: false }
energies:
  - type: TemplateMatchEnergy  # or LISEnergy, PTMEnergy, HydrophobicEnergy, etc.
    kwargs: { weight: 1.0, ... }
```

### Multi-Chain Design

For binding design, energy configs specify multiple chains with residue ranges:
- `GEN` chain = the generated sequence
- Named chains (A, B, etc.) = target proteins (from PDB or local file)
- ESMFold receives all chains joined with `":"` separator

### Key Mechanisms

- **Constrained generation**: `enforce_template: true` forces specific residues via ProFam's logits processor; `false` assigns inf energy on mismatch
- **Memory pooling**: `n_memory > 0` includes sequences from previous N cycles in the selection pool
- **Template matching**: extracts fixed residues from energy config and passes to ProFam as `fixed_positions`
- **PDB caching**: structures cached in `~/.cache/profam_bagel/pdb/`
- **Residue spec notation**: `"0-43"`, `"1,2,5"`, `"0-5,10,20-25"`, `"all"`

## BAGEL Energy System Internals

Energy terms live in the BAGEL library (installed package at `bagel/energies.py`, ~1300 lines). Understanding this is critical for adding new scoring functions.

**Class hierarchy:** `Oracle` → produces `OracleResult` → consumed by `EnergyTerm.compute()`

**Oracle types:**
- `FoldingOracle` (abstract) → `ESMFold` (concrete). Returns `FoldingResult` with `structure`, `local_plddt`, `ptm`, `pae`
- `EmbeddingOracle` (abstract) → `ESM2` (concrete). Returns `EmbeddingResult` with `embeddings`

**EnergyTerm base class** (ABC):
- `compute(oracles_result: OraclesResultDict) -> tuple[float, float]` — returns `(unweighted, weighted)` energy
- `oracle` attribute — which oracle this term needs
- `residue_groups` — target residues as `ResidueGroup = tuple[chain_ids_array, res_indices_array]`
- `weight` — multiplicative weight

**Existing energy terms:** PTMEnergy, PLDDTEnergy, OverallPLDDTEnergy, LISEnergy, PAEEnergy, TemplateMatchEnergy, SurfaceAreaEnergy, HydrophobicEnergy, FlexEvoBindEnergy, SeparationEnergy, RingSymmetryEnergy, GlobularEnergy, SecondaryStructureEnergy, EmbeddingsSimilarityEnergy, ChemicalPotentialEnergy

**Pipeline integration** (`run_profam_bagel_pipeline.py`):
- `build_folding_oracle()` — instantiates oracle from energy YAML `folding_oracle.type`
- `build_energy_terms_for_chain()` — instantiates each energy term from `energies` list, injects `oracle`, converts `residues` specs to `bg.Residue` objects
- `evaluate_sequences_with_bagel()` — folds each sequence, calls `energy_term.compute()`, sums weighted energies

**To add a new oracle type**, create a subclass of `FoldingOracle` with a `fold()` method returning a result with the needed metrics, then create `EnergyTerm` subclasses that consume those metrics.

## Current Branch: 15-PGDH Binder Design (`add-alphafast`)

### Challenge Context

This branch targets the Berlin Bio x AI Hackathon challenge: designing protein binders against **15-PGDH** (15-Hydroxyprostaglandin Dehydrogenase), an NAD+-dependent enzyme that degrades PGE2. 15-PGDH activity rises with age, accelerating PGE2 degradation and contributing to decline in muscle, brain, and joint tissue. It was identified as a "gerozyme" by Stanford's Blau Lab. Inhibiting 15-PGDH rejuvenates aged muscle stem cells, restores neuromuscular junctions, and repairs joint cartilage in preclinical models. Epirium Bio's small-molecule inhibitor MF-300 completed Phase 1 trials (Sept 2025), but no protein-based therapeutics exist. There are zero FDA-approved drugs for sarcopenia (>50M affected globally).

**Design targets:** NAD+ binding pocket (active site), homodimer interface (allosteric), or surface epitopes. Max sequence length: 250 aa.

**Target PDB structures:** 9PFL (multimer), 2GDZ (monomer)

### Working Configuration

`pipeline_berlin_hairpin_start.yaml` — uses ESMFold on Modal with LISEnergy against the 2GDZ target sequence. Energy config: `example_energy_lis_2GDZ.yaml`.

### Planned New Energy Functions

The goal is to add scoring based on higher-accuracy structure predictors beyond ESMFold:

**AlphaFast** (https://github.com/RomeroLab/alphafast) — GPU-accelerated AlphaFold3:
- Replaces CPU Jackhmmer with GPU MMseqs2, ~23x faster end-to-end
- Outputs: iPTM, pTM, pLDDT (per-atom), PAE matrix, ranking_score, CIF structures
- Runs via Modal serverless (~$0.035/prediction, ~28s) or Docker locally
- Input: AF3 JSON format with sequences and modelSeeds
- Requires AF3 model weights from Google DeepMind (request needed)
- Key scores for binding: `chain_pair_iptm` matrix (off-diagonal = interface quality)

**ipSAE** (https://github.com/DunbrackLab/IPSAE) — improved interface scoring:
- Computes ipSAE score from PAE matrix + structure coordinates (numpy only, CPU)
- More reliable than standard iPTM for evaluating protein-protein interfaces
- Takes AF2/AF3/Boltz output (PAE JSON + CIF) as input
- Threshold: ipSAE > 0.6 suggests likely binding
- Also computes: pDockQ, pDockQ2, LIS, per-residue contributions
- Usage: `python ipsae.py <pae_file> <structure_file> <pae_cutoff> <dist_cutoff>`

**Integration approach:** New energy terms should ideally be added to the BAGEL library (new oracle for AlphaFast, new energy terms for ipSAE/iPTM). The pipeline's `build_folding_oracle()` and `build_energy_terms_for_chain()` already support dynamic dispatch by type name from YAML config.

## Dependencies

BAGEL and ProFam have conflicting pins for numpy, matplotlib, and transformers. The setup script installs BAGEL first (stricter pins), then ProFam in editable mode. ProFam is cloned into `.profam_repo/`.

## Environment Variables

- `MODEL_DIR` — path to ESMFold weights (default: `~/.cache/bagel/models`)
- `HF_TOKEN` — HuggingFace token for gated model downloads (Modal uses `huggingface-secret`)

## Output Structure

Each run writes to `output_dir/`:
- `cycle_stats.json` — per-cycle energies, similarities, selected indices
- `sequences_cycle_XXX/` — CIF structures for selected sequences
- `sequences_cycle_all_XXX/` — CIF structures for all generated sequences
- `energy_summary.png` — energy vs cycle plot
