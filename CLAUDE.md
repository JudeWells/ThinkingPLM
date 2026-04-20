# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Open-Source Cleanup Guide

This repo is being cleaned up for open-source publication. The goal is to strip it to the minimal functionality needed to run the current experiments (scaffold comparison + multi-target binder design benchmarks). Below is a detailed guide on what to keep, remove, and refactor.

### What the Published Code Should Support

The pipeline iteratively generates protein binder sequences using ProFam (a protein language model), evaluates them via structure prediction (ESMFold) and energy scoring (BAGEL), and selects promising candidates for the next cycle.

**Three optimization methods** to compare:
1. **Random greedy** — random single-point mutations, greedy selection, 1 sample/cycle
2. **Proposal bandit** — Thompson bandit chooses between ProFam generation and random mutation, greedy selection, 1 sample/cycle
3. **Bandit + GRPO** — proposal bandit + GRPO preference optimization, 12 samples/cycle

**Two experiment types:**
1. **Scaffold comparison** — 4 scaffolds × 4 methods (includes Bradley-Terry) against a single target (2GDZ/15-PGDH)
2. **Multi-target benchmark** — 5 targets × 4 scaffolds × 3 methods (no BT)

**Energy function:** ESMFold (BatchedESMFold oracle) + LIS energy (weight 1.0) + optional PLDDT (weight 0.1)

### Files to KEEP (Core)

| File | Purpose | Notes |
|------|---------|-------|
| `run_profam_bagel_pipeline.py` | Main pipeline (~3800 lines) | Needs significant simplification (see below) |
| `pipeline/bandits.py` | ThompsonSampler, ProposalBandit | TemperatureBandit class can be removed |
| `pipeline/grpo.py` | GRPO training step | Keep as-is |
| `pipeline/bradley_terry.py` | Bradley-Terry preference learning | Keep as-is |
| `pipeline/selection.py` | GreedyPromptSelector, SelectionManager | ThompsonPromptSelector unused in current experiments but simple to keep |
| `pipeline/proposal.py` | ProFam and RandomMutation proposal generators | Keep as-is |
| `pipeline/logging.py` | Cycle stats logging, CSV export, structure saving | Keep as-is |
| `pipeline/plotting.py` | Energy summary plots | Keep as-is |
| `pipeline/utils.py` | Sequence identity, reward extraction, softmax | Keep as-is |
| `pipeline/__init__.py` | Package imports | Remove ColabFold, TemperatureBandit imports |
| `generate_scaffold_comparison_configs.py` | Scaffold comparison config generator | Keep as-is |
| `generate_grpo_multi_target_bench.py` | Multi-target bench config generator | Keep as-is |
| `setup_environment.sh` | Conda environment setup | Keep as-is |
| `setup_cloud.sh` | Cloud instance one-command setup | Keep as-is |
| `configs/energy/energy_lis_plddt_*.yaml` | ESMFold + LIS + PLDDT energy configs (6 files) | Active configs |
| `configs/energy/energy_lis_*_local.yaml` | ESMFold + LIS only configs | For reference/backward compat |
| `configs/sequences/*.fasta` | Starting scaffold sequences | Keep all |
| `configs/pipelines/scaffold_comparison/` | Generated pipeline configs | Keep |
| `configs/pipelines/multi_target_bench_mt2/` | MT2 bench configs | Keep |

### Files to REMOVE

| File/Directory | Reason |
|----------------|--------|
| `run_profam_bagel_modal_app.py` | Modal cloud execution — not used (run_on_modal=false everywhere) |
| `modal_proteinmpnn_score.py` | Legacy Modal SolMPNN scoring — superseded by the local subprocess backend shipped with BAGEL (`bagel.scripts.proteinmpnn_scorer`) |
| `pipeline/colabfold_oracle.py` | ColabFold oracle — not used in current experiments |
| `pipeline/batched_esmfold.py` | Replaced by BAGEL's built-in BatchedESMFold |
| `pipeline/temperature_bo.py` | Temperature Bayesian optimization — never used |
| `slurm_scripts/` (247 files) | Auto-generated SLURM scripts from old benchmarks |
| `slurm_*.sh`, `submit_all_*.sh` | Legacy SLURM submission scripts |
| `run_all_bench*.sh` | Legacy benchmark runners |
| `generate_2gdz_campaign_configs.py` | Superseded by scaffold_comparison generator |
| `generate_proposal_bandit_bench.py` | Superseded |
| `generate_ensemble_benchmark_configs.py` | Superseded |
| `generate_greedy_diverse_configs.py` | Superseded |
| `generate_thompson_eb8_configs.py` | Superseded |
| `generate_benchmark_configs.py` | Superseded |
| `create_greedy_slurm.py` | Legacy |
| `run_grpo_hp_sweep.py`, `_v2.py`, `_v3.py` | Superseded by v4 |
| `run_hp_search.py` | Legacy Optuna HP search |
| `analyze_*.py` (all) | Analysis scripts — keep separately if needed |
| `simulate_thompson*.py` (3 files) | Thompson sampling simulations |
| `experiment_boltz_vs_colabfold.py` | Legacy experiment |
| `test_boltz_determinism.py`, `test_ipsae_agreement.py` | Legacy tests |
| `diagnose_temperature_bo.py` | Legacy diagnostic |
| `benchmark_boltz_samples.py` | Legacy benchmark |
| `random_scripts/` | Campaign plotting/animation (can archive separately) |
| `berlin_hack_bio/` | Hackathon-specific code |
| `hp_search/` | HP search results |
| `configs/pipelines/benchmark_v0/` (132 files) | Legacy benchmark configs |
| `configs/pipelines/benchmark_v1/` | Empty |
| `configs/pipelines/2gdz_campaign/` (52 files) | Legacy campaign configs |
| `configs/pipelines/multi_target_bench/` | MT1 configs (superseded by mt2) |
| `configs/energy/example_energy_*.yaml` | Legacy example energy configs |
| `configs/energy/energy_boltz_*.yaml` | Boltz oracle configs — not used |
| `configs/energy/energy_colabfold_*.yaml` | ColabFold configs — not used |
| `docs/` | Internal design docs (review for useful content first) |
| `smoke_test.db` | Test artifact |
| `template.cif` | Test artifact |

### Simplifying `run_profam_bagel_pipeline.py`

The main file is ~3800 lines. Key simplifications:

**PipelineConfig fields to REMOVE** (unused in all active configs):
- `annealing_initial_temp`, `annealing_decay` — simulated annealing, never used
- `thompson_temperature_bins` — temperature bandit, always null
- `run_on_modal` — Modal cloud execution, always false
- `profam_max_generated_length` — always null
- All Modal-related code paths

**PipelineConfig fields to KEEP:**
- Core: `initial_fasta`, `energy_config`, `max_cycles`, `output_dir`, `random_seed`
- ProFam: `profam_checkpoint_dir`, `profam_sampler`, `profam_num_samples`, `profam_max_tokens`, `profam_temperature`, `profam_top_p`
- Sampling: `f_inject`, `softmax_temperature`, `sample_with_reinsertion`, `reinject_initial`
- Anti-regression: `elitism`, `accept_only_improvement`
- Proposal: `proposal_method`, `max_mutations`, `freeze_prompt`
- Selection: `selection_strategy`, `thompson_m_samples`, `thompson_reward_term`, `thompson_exploit_bias`
- Proposal bandit: `thompson_proposal_bandit`, `proposal_bandit_prior_alpha/beta`, `proposal_bandit_relative_reward`
- Diversity: `thompson_max_arms`, `thompson_max_identity`, `deduplicate_sequences`
- GRPO: `grpo_enabled`, `grpo_lr`, `grpo_beta`, `grpo_clip_ratio`, `grpo_temperature`, `grpo_group_size`, `grpo_replay_cycles`, `grpo_use_reference_model`, `rl_every_n_cycles`, `rl_steps_per_cycle`
- BT: `bt_enabled`, `bt_lr`, `bt_pool_size`, `bt_batch_size`, `bt_sub_batch_size`, `bt_every_n_cycles`, `bt_steps_per_cycle`
- Logging: `output_frequency`, `wandb_enabled`, `wandb_project`, `wandb_run_name`, `wandb_tags`, `save_structures`
- Random init: `random_init`, `random_init_max_residues`
- Template: `enforce_template`, `n_memory`
- Likelihood tracking: `likelihood_eval_every`, `likelihood_track_n`

**Code blocks to REMOVE from `run_profam_bagel_pipeline.py`:**
- All Modal-related functions and code paths (`run_on_modal`, `force_modal_folding`, Modal volume sync)
- Temperature bandit initialization and selection (search for `temp_bandit`)
- Simulated annealing logic (search for `annealing`)
- AlphaFast oracle code paths
- `profam_max_generated_length` handling
- The `checkpoint_callback` parameter (Modal-specific)

SolMPNN scoring is handled by BAGEL's `SolMPNNPerplexityEnergy` (see section below) and does not need pipeline-level code.

**Code blocks to KEEP (critical pipeline logic):**
- Config loading: `PipelineConfig`, `build_arg_parser()`, `merge_config()`
- Model loading: `load_profam_model()`, `run_profam_generation()`
- Evaluation: `build_folding_oracle()`, `build_energy_terms_for_chain()`, `evaluate_sequences_with_bagel()`
- Sampling: `softmax_from_energies()`, `sample_subset_indices()`
- Temperature adjustment: `adjust_temperature()` (gated on profam only)
- Main loop: `run_pipeline()` — generate → fold+score → dedup → select → elitism/swap → inject
- Dedup retry logic with escalating max_mutations for random_mutation
- Random init with cysteine exclusion and retry on folding failure
- GRPO and BT training steps within the main loop
- Proposal bandit selection and update
- Adaptive temperature (for profam proposals only)

### Pipeline Architecture (for reference during refactoring)

```
run_pipeline(cfg)
  ├── Load ProFam model
  ├── Build folding oracle (BatchedESMFold from BAGEL)
  ├── Load energy config (LIS + optional PLDDT)
  ├── Read initial sequences (from FASTA or random_init)
  ├── Evaluate seed sequences (with retry for random_init)
  ├── Initialize Thompson/proposal bandits (if enabled)
  ├── Initialize GRPO/BT optimizers (if enabled)
  │
  └── Main loop (for cycle in 1..max_cycles):
      ├── Proposal bandit selects method (profam or random_mutation)
      ├── Generate sequences (ProFam or random mutation)
      ├── Deduplicate against seen_sequences cache
      │   ├── Retry loop (10 for profam, 500 for random_mutation)
      │   ├── Escalating max_mutations every 20 retries (random only)
      │   └── Adaptive temperature adjustment (profam only)
      ├── Evaluate novel sequences with BAGEL (fold + energy)
      ├── Merge novel + cached results
      ├── Softmax selection → injection set
      ├── Elitism: preserve global best
      ├── Accept/reject swap (accept_only_improvement)
      ├── Update proposal bandit with improvement signal
      ├── GRPO/BT training step (if enabled, on profam cycles)
      ├── Log cycle stats, update CSV
      └── Periodic: generate energy plot, checkpoint
```

### SolubleMPNN (SolMPNN) Perplexity Energy

`SolMPNNPerplexityEnergy` scores how well a binder sequence matches a given
backbone using SolubleMPNN's autoregressive perplexity — a proxy for
designability / foldability. Lower = better. It works with any folding oracle
(ESMFold, Boltz, Chai-1, AF2BindCraft) because it only reads the predicted
structure from `oracles_result`.

**Implementation (local subprocess backend):**

The energy lives in the BAGEL package (`bagel/energies.py`) and calls a
bundled standalone scorer at `bagel/scripts/proteinmpnn_scorer.py`. Because
ProteinMPNN has its own dependency stack that conflicts with BAGEL/Boltz, the
scorer runs in an isolated conda env via `conda run -n <proteinmpnn_env>`.

Required setup:
1. Clone ProteinMPNN: `git clone https://github.com/dauparas/ProteinMPNN.git /mnt/disk2/ProteinMPNN`
2. Create a conda env with torch + numpy that can import ProteinMPNN (the
   existing `proteinmpnn` env works).
3. Point `SolMPNNPerplexityEnergy` at both via `proteinmpnn_env` and
   `proteinmpnn_path` kwargs.

**Hyperparameters (local mode):**

- `backbone_noise` (float, default 0.0) — Gaussian std applied to backbone
  coordinates each forward pass (`augment_eps` in ProteinMPNN). Use >0 to
  propagate structural uncertainty into the score.
- `ensemble_n` (int, default 10) — number of forward passes. Each pass uses
  an **independent backbone-noise draw** and an **independent decoding
  order**. The perplexity is `exp(mean NLL)` across passes.
- `decoding_order` (str, default `"random"`) — `"random"` draws fresh
  decoding-order noise each pass; `"fixed:<seed>"` is deterministic.
- `residues` — specifies which chain(s) the perplexity is computed on
  (typically the GEN/binder chain). Residues on other chains are visible to
  the encoder but don't contribute to the loss.

**Complex-context scoring (default):**

The energy is always evaluated on whatever structure the folding oracle
produced. In standard binder campaigns the oracle folds `binder + target`
together, so SolMPNN sees the full complex and the encoder has access to the
target residues as "binding context" — even though only the binder residues
enter the loss. This is the intended behaviour for interface-aware design.

**Validation on a real heterodimer (1YCR, p53 peptide bound to MDM2):**

To verify that the target context meaningfully changes the perplexity, we
ran `test_mpnn_context_significance.py` which scores the p53 peptide
(13 aa, chain B of 1YCR) under two conditions with 10 repeats of 10
ensemble passes each (backbone_noise=0.1):

| Condition | Mean perplexity | Std | t-stat vs monomer |
|---|:---:|:---:|:---:|
| p53 peptide alone (monomer) | **16.57** | 0.19 | — |
| p53 peptide in MDM2 context (complex) | **4.47** | 0.07 | **-188** |

The 73% drop in perplexity when the MDM2 context is visible confirms that
SolMPNN treats the p53 residues as "justified" specifically because they
pack against the MDM2 binding groove. Natural interface motifs can be very
hard to "design" in isolation but highly designable in context. This is
far outside the natural ensemble variance (std ~0.07-0.19) and validates
that the complex-context pipeline is working correctly.

Always evaluate SolMPNN in the full-complex context for binder design —
monomer-only scoring would penalise legitimate interface sequences.

### Multi-Oracle Energy Configs

The pipeline supports energy configs that span **multiple folding oracles**
simultaneously (ESMFold, Boltz2, Chai-1, AF2BindCraft). Each energy term in
the YAML can reference a named oracle via an `oracle:` key, so you can
extract the same metric (pLDDT, iPTM, ipSAE, SolMPNN) from every predictor.

**Schema:**

```yaml
folding_oracles:              # plural — map of name → oracle spec
  esmfold:
    type: BatchedESMFold
    kwargs: {use_modal: false}
  boltz2:
    type: Boltz
    kwargs:
      diffusion_samples: 5    # Boltz averages 5 samples inside the oracle
      recycling_steps: 3
  chai1:
    type: Chai1
    kwargs:
      num_diffn_samples: 3    # Chai-1 native ensemble
      num_trunk_recycles: 3
  af2:
    type: AF2BindCraft
    kwargs:
      target_pdb: /path/to/target.pdb
      target_chain: A
      conda_env: BindCraft
      num_recycles: 3
      prediction_models: [0, 1]

energies:
  - type: iPTMEnergy
    oracle: boltz2            # per-term oracle reference
    kwargs: {weight: 0.05, name: b2}
  - type: iPTMEnergy
    oracle: chai1
    kwargs: {weight: 0.05, name: chai}
  # … etc.
```

**Implementation notes:**

- The legacy single `folding_oracle:` (singular) schema is still supported
  for backwards compatibility — if the plural form is absent, the old
  single-oracle path is used unchanged.
- Energy terms without an explicit `oracle:` key fall back to the first
  oracle in `folding_oracles`.
- Every sequence is folded by every oracle referenced by any energy term
  (plus one `torch.cuda.empty_cache()` between oracle calls to stop
  in-process models from OOM-ing each other).
- **Each oracle returns exactly one `FoldingResult` per sequence.**
  Internal ensembling (Boltz `diffusion_samples`, Chai-1 `num_diffn_samples`,
  AF2 `prediction_models`) is absorbed inside the oracle and metrics are
  averaged into a single result before the energy terms see them.
- The old pipeline-level `boltz_ensemble_n` field and `run_boltz_ensemble`
  helper have been **removed**. To run multiple Boltz diffusion samples,
  set `diffusion_samples: N` in the Boltz oracle's `kwargs`. Doing it at
  the oracle level means each oracle in a multi-oracle config can specify
  its own ensemble size independently.
- The Boltz oracle's `fold()` method now parses all N samples internally
  when `diffusion_samples > 1`, averages the scalar/tensor metrics
  (pTM, iPTM, pLDDT, PAE), and returns a single averaged `BoltzResult`
  (structure taken from sample 0 since averaging coordinates is
  meaningless).
- **Memory budget**: running ESMFold + Chai-1 together in-process needs
  a GPU with >24 GB (ESMFold alone takes ~15–20 GB). On a 40 GB A100 this
  is fine. Boltz and AF2BindCraft use subprocess-isolated processes so
  they do not share main-process GPU memory.

### Key Design Decisions

- **1 sample per cycle** for random_greedy and proposal_bandit (immediate feedback)
- **12 samples per cycle** for GRPO (needs group for preference optimization)
- **5400 total evaluations** per run (equal budget: 5400×1 or 450×12)
- **`max_mutations=1`** — single point mutations only (escalates during dedup exhaustion)
- **`save_structures=false`** by default — CIF/PAE files consume ~65GB per experiment
- **`enforce_template=false`** for multi-target, `true` for scaffold_comparison
- **Cysteine excluded** from random_init sequences (causes ESMFold failures)
- **Temperature only adjusted for profam proposals** (random_mutation ignores temperature)
- **SolMPNN always sees the complex context** (not monomer) for binder scoring

### Tests to KEEP

| Test | Purpose |
|------|---------|
| `test_grpo_synthetic.py` | Validates GRPO training on synthetic copy/shift tasks |
| `test_encoder_decoder_grpo.py` | Tests encoder-decoder GRPO compatibility |
| `test_dedup_fixes.py` | Validates deduplication logic and novel_mask fix |
| `test_mpnn_context_significance.py` | Validates SolMPNN complex-vs-monomer context on 1YCR |

### Dependencies

BAGEL (`biobagel`) and ProFam have conflicting numpy pins. Install order matters:
1. BAGEL first (pins numpy, torch, transformers)
2. ProFam in editable mode (from `.profam_repo/`)
3. Force `numpy==1.26.4` last (numba requires <2.2, boltz requires <2.0, but both work with 1.26.4)
4. `modal` pinned to `0.73.45` to avoid deprecation errors in boileroom

See `setup_environment.sh` for the full installation sequence.

## Quick Start

```bash
# Setup
chmod +x setup_environment.sh && ./setup_environment.sh
conda activate profam_bagel

# Run scaffold comparison (single run)
python run_profam_bagel_pipeline.py --config configs/pipelines/scaffold_comparison/hairpin_random_greedy_rep3.yaml

# Generate all configs
python generate_scaffold_comparison_configs.py --replicate 1
python generate_grpo_multi_target_bench.py --variant mt2

# CLI flags override YAML values
python run_profam_bagel_pipeline.py --config <config.yaml> --max_cycles 5 --wandb_enabled false
```

## Output Structure

Each run writes to `output_dir/`:
- `pipeline_config.json` — full config snapshot for reproducibility
- `cycle_stats.json` — per-cycle energies, similarities, swap decisions, elite tracking
- `all_sequences.csv` — all evaluated sequences across all cycles
- `energy_summary.png` — energy vs cycle plot (dark mode)
- `cycle_NNN/profam_input.fasta` — the prompt sequence used for each cycle

## Configuration System

Two YAML files drive each run:
1. **Pipeline config** — ProFam settings, method selection, cycle count, anti-regression
2. **Energy config** (referenced by `energy_config` key) — folding oracle type and energy terms with weights

Energy config structure:
```yaml
folding_oracle:
  type: BatchedESMFold
  kwargs:
    use_modal: false
    max_batch_size: 16
energies:
  - type: LISEnergy
    kwargs:
      weight: 1.0
      pae_cutoff: 12.0
      intensive: true
      target: <TARGET_SEQUENCE>
      residues:
        GEN: "all"
        B: "all"
  - type: PLDDTEnergy  # optional
    kwargs:
      weight: 0.1
      residues:
        GEN: "all"
```

Multi-chain design: `GEN` chain = binder, `B` chain = target. ESMFold receives chains joined with `":"` separator.
