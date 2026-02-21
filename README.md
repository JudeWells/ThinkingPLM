## ProFam + BAGEL Generative Pipeline

This repository implements an iterative generative design pipeline that combines **ProFam** (protein sequence generation) and **BAGEL** (structure prediction + energies), driven by simple YAML configuration files.  ProFam and BAGEL are installed as external pip packages from their GitHub repositories — they are **not** part of this codebase.

The main entrypoint is `run_profam_bagel_pipeline.py` at the repository root, plus convenience scripts for running on a PBS cluster or on a local Mac.

---

### 1. High-level pipeline overview

For each cycle (1 … max_cycles), the pipeline:

1. **ProFam generation**
   - Reads an input FASTA containing the **initial sequences** and, from cycle 2 onward, the **subset selected in the previous cycle**.
   - Calls ProFam's `scripts/generate_sequences.py` to generate `profam_num_samples` new sequences.

2. **BAGEL folding + energy evaluation**
   - For each generated sequence:
     - Builds a `bagel.Chain` (chain ID `A`).  If any energy term specifies a `"target"` sequence, builds additional target chains (chain IDs `B`, `C`, …) and folds the multi-chain complex.
     - Uses an `ESMFold` folding oracle to predict the 3D structure (only when at least one energy term requires it).
     - Computes a **weighted total energy** as the sum of user-configured BAGEL energy terms.
   - Computes the **average sequence similarity** between generated sequences and the initial input sequences.

3. **Probability assignment**
   - Converts energies into probabilities via softmax over `-E / T` (where T = `softmax_temperature`).

4. **Subset selection and logging**
   - Samples a subset of size `floor(f_inject * N_output)` (at least 1), with or without replacement depending on `sample_with_reinsertion`.
   - Writes a JSON log per cycle with energies, sequence similarity, selected indices, sequences, and per-term breakdowns.
   - Saves CIF structures for all generated sequences and for the selected subset (when folding was performed).

5. **Injection into the next cycle**
   - When `reinject_initial` is `true` (default), the selected subset is merged with the initial sequences to form the input FASTA for the next ProFam call.
   - When `reinject_initial` is `false`, only the selected subset is used from cycle 2 onward.

6. **Summary plot**
   - After the final cycle, generates a plot of average and minimum energy vs cycle index, with average sequence similarity on a secondary y-axis.

---

### 2. Requirements and environment setup

| Requirement | Details |
|---|---|
| **Python** | 3.11 (BAGEL requires `>=3.11,<3.14`) |
| **conda** | Recommended for environment management |
| **GPU** | Needed for ESMFold folding (either locally, on Modal, or on a cluster) |

#### Dependency notes

ProFam and BAGEL have overlapping but conflicting dependency pins:

| Package | BAGEL requires | ProFam freeze |
|---|---|---|
| numpy | `>=2.2.0` | `==1.26.4` |
| matplotlib | `>=3.10.0` | `==3.9.4` |
| transformers | `>=4.49.0` | `==4.48.3` |

BAGEL also needs `boileroom==0.2.2`, `pydantic`, `modal`, and `biotite`, which are not in ProFam's dependency list. The provided setup script resolves all of these conflicts automatically.

Additionally, `boileroom==0.2.2` constrains the `torch` version it installs (currently `torch==2.6.0`), and `torchvision`/`torchaudio` must match exactly (e.g. `torchvision==0.21.0` for `torch==2.6.0`). The setup script handles this automatically by detecting the installed torch version and installing matching companion packages.

#### Quick setup (recommended)

```bash
# From the repository root:
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will:
1. Create a `profam_bagel` conda environment with Python 3.11.
2. Install BAGEL (`biobagel`) from GitHub — pulls `boileroom`, `biotite`, `numpy`, `pydantic`, etc.
3. Detect the `torch` version that `boileroom` installed and install matching `torchvision`/`torchaudio`.
4. Install ProFam from GitHub — pulls `transformers`, `lightning`, `hydra-core`, etc.
5. Install pipeline utilities (`pyyaml`, `modal`).
6. Verify that all key imports work.

After the script completes:

```bash
conda activate profam_bagel
python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
```

#### Manual setup

If you prefer to set up the environment manually:

```bash
conda create -n profam_bagel python=3.11 -y
conda activate profam_bagel

# 1. Install BAGEL from GitHub (sets the torch version via boileroom)
pip install "biobagel[local] @ git+https://github.com/softnanolab/bagel.git"

# 2. Install matching torchvision/torchaudio for the torch version boileroom pulled
#    Check: python -c "import torch; print(torch.__version__)"
#    For torch 2.6.x:
pip install torchvision==0.21.0 torchaudio==2.6.0

# 3. Install ProFam from GitHub
pip install "git+https://github.com/alex-hh/profam.git"

# 4. Install additional ProFam runtime dependencies not in its setup.py
pip install rootutils safetensors huggingface-hub biopython scipy scikit-learn

# 5. Install pipeline utilities
pip install pyyaml modal
```

#### Prerequisites

Before running the pipeline you also need:

1. **ProFam model checkpoint** — download it into `model_checkpoints/`:
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='model_checkpoints/profam-1')"
   ```
2. **Template structure** (if using `TemplateMatchEnergy`) — place the `.cif` file at the path specified in your energy config (e.g. `template.cif` in the repo root), or use a `pdb_code` to download it automatically.
3. **Modal token** (if using `run_on_modal: true`) — authenticate with:
   ```bash
   modal token new
   ```

---

### 3. Pipeline configuration

The pipeline is configured with two YAML files:

- A **pipeline YAML** file for high-level pipeline settings.
- An **energy YAML** file specifying the BAGEL folding oracle and energy terms.
- CLI flags can override any value from the pipeline YAML.

#### 3.1 YAML configuration (pipeline)

Minimal example (see `example_pipeline_config.yaml` for the full version):

```yaml
# Input sequences
initial_fasta: initial_sequences.fasta

# ProFam
profam_checkpoint_dir: model_checkpoints/profam-1
profam_sampler: single
profam_num_samples: 64
profam_temperature: 0.8
profam_top_p: 0.95

# BAGEL energy configuration
energy_config: example_energy_template_match.yaml

# Pipeline control
f_inject: 0.25
max_cycles: 10
output_dir: outputs/pipeline_run1
softmax_temperature: 1.0
random_seed: 42

# Subset selection
sample_with_reinsertion: true   # false = sample without replacement
reinject_initial: true          # false = don't prepend initial seqs from cycle 2+
n_memory: 0                     # pool previous cycles' sequences for selection

# Run entire pipeline on Modal (GPU in the cloud)
run_on_modal: true
output_frequency: 1             # sync results every N cycles
enforce_template: true          # force template residue identity during generation
```

**Required keys** (must be provided either in YAML or via CLI):

- `initial_fasta`
- `profam_checkpoint_dir`
- `energy_config`

#### 3.2 Energy YAML configuration

The energy YAML defines the folding oracle and a list of energy terms.

Example:

```yaml
folding_oracle:
  type: ESMFold
  kwargs:
    use_modal: false

energies:
  - type: PTMEnergy
    kwargs:
      weight: 1.0
  - type: OverallPLDDTEnergy
    kwargs:
      weight: 0.5
```

- `folding_oracle.type`: Must be `ESMFold`.
  - `kwargs` are passed to `bagel.oracles.folding.ESMFold.__init__`.
  - When `run_on_modal: true` in the pipeline YAML, `use_modal` is **forced to `true`** regardless of this setting.
- `energies`: Each entry is a BAGEL energy term class from `bagel.energies`:
  - `type`: e.g. `PTMEnergy`, `OverallPLDDTEnergy`, `TemplateMatchEnergy`, `SurfaceAreaEnergy`, `HydrophobicEnergy`, `PAEEnergy`, `SeparationEnergy`, etc.
  - `kwargs`: Passed to the energy term's `__init__`, with `oracle` set automatically.
  - If a `residues` field is present, it must be a **dictionary** mapping chain names to residue specifications. The generated chain always uses the key `GEN`. For multi-chain energy terms, additional keys identify target chains (see sections 3.5 and 3.7). Values can be compact range strings (e.g. `"0-43"`), integer lists, or `"all"`.
  - If a `target` field is present, the pipeline builds a multi-chain system for that energy term (see section 3.5).
  - Structure references (e.g. `template_structure_path`) can be replaced with `pdb_code` to download directly from the RCSB PDB (see section 3.8).

#### 3.3 TemplateMatchEnergy

`TemplateMatchEnergy` computes the RMSD between a subset of the generated structure and the corresponding subset of a reference template structure. The key concept is that the `residues.GEN` indices specify **the same 0-based positions** in both structures:

1. **`residues`**: A dictionary with a single key `GEN` whose value lists 0-based residue positions extracted from **both** the generated (folded) structure and the template chain. Values can be a compact range string (e.g. `"0-43"`, see section 3.7) or a list of integers. Because the same positions are used on both sides, the atom counts always match.

2. **Template structure loading**: Provide **either**:
   - `template_structure_path` + optional `template_chain_id`: a local CIF/PDB file and chain to filter by.
   - `pdb_code` + optional `template_chain_id`: a PDB identifier to download from RCSB (see section 3.8).

3. **`backbone_only`**: When `true`, only backbone atoms (CA, N, C — 3 per residue) are compared on both sides. When `false`, all atoms are compared — this requires that the amino-acid identity at each compared position is the same in both structures (otherwise the per-residue atom counts differ).

**Example with local file** (see `example_energy_template_match.yaml`):

```yaml
- type: TemplateMatchEnergy
  kwargs:
    weight: 1.0
    backbone_only: true
    template_structure_path: template.cif
    template_chain_id: A
    residues:
      GEN: "0-43"
```

**Example with PDB download:**

```yaml
- type: TemplateMatchEnergy
  kwargs:
    weight: 1.0
    backbone_only: true
    pdb_code: "1ubq"
    template_chain_id: A
    residues:
      GEN: "0-43"
```

Here the first 44 residues (0-based) are compared between the generated structure and the template chain.

#### 3.4 Constrained generation (`enforce_template`)

When `TemplateMatchEnergy` is used, the generated sequence must have the correct amino-acid identity at the template-matching positions, otherwise the atom counts will differ and evaluation fails. The `enforce_template` YAML flag controls how this is handled:

| `enforce_template` | Behaviour |
|---|---|
| `true` (default) | During ProFam generation, the amino acids at the `residues` positions are **forced** to match the template sequence using a logits processor. This guarantees the correct identity at constrained positions while letting the model freely generate the remaining positions. |
| `false` | ProFam generates freely. If a generated sequence has a different amino acid at a template position, the atom count mismatch causes a `ValueError` during energy evaluation. The pipeline catches this error and assigns **infinity energy** to that sequence. If **all** sequences in a cycle receive infinity energy, the cycle is retried (up to 5 attempts). |

#### 3.5 Multi-chain energy terms (`"target"`)

Energy terms that operate on **two groups of residues** (e.g. `PAEEnergy`, `SeparationEnergy`, `LISEnergy`) can be used to evaluate interactions between the ProFam-generated chain and a fixed **target** chain. This is useful for designing sequences that bind to or interact with a known protein.

To enable this, specify the target chain in one of two ways:

**Option A — inline sequence:**

```yaml
- type: LISEnergy
  kwargs:
    weight: 1.0
    target: MKTAYIAKQRQISFVKSH...
    residues:
      GEN: "0-19"
      B: "0-9"
```

The non-`GEN` key (`B` here) becomes the target chain's identifier in the output CIF files.

**Option B — download from PDB:**

```yaml
- type: LISEnergy
  kwargs:
    weight: 1.0
    target_pdb_code: "1ubq"
    target_chain_id: A
    residues:
      GEN: "0-19"
      A: "0-9"
```

The non-`GEN` key must match the `target_chain_id` value.

When a target is present:

1. The pipeline builds a **target chain** from the provided sequence (or downloaded from PDB). Its chain ID is taken from the non-`GEN` key in the `residues` dict.
2. The generated sequence uses chain ID `GEN`, the target uses the key from `residues`.
3. ESMFold folds both chains **together** as a multi-chain complex (using its native `":"` separator).
4. The `"residues"` dict maps each chain key to its residue indices. `GEN` = residues on the generated chain, the other key = residues on the target chain. Compact range strings are supported (see section 3.7).
5. The energy term (e.g. `LISEnergy`) computes the metric across the two chains using the predicted complex structure.
6. Output CIF files use these chain IDs (`GEN` for the generated chain, the target key for the target chain).

A full example is provided in `example_energy_lis_binding.yaml`.

Multiple energy entries can each have their own target — if two entries share the same target sequence and chain ID, the pipeline de-duplicates them into a single target chain.

#### 3.6 Sequence similarity tracking

At each cycle, the pipeline automatically computes the **average sequence similarity** between the ProFam-generated sequences and the initial input sequences (from `initial_fasta`). For each generated sequence, the best-match similarity (fraction of identical residues, aligned from position 0) against all initial sequences is calculated. The mean of these best-match values is:

- Printed to the console during the run.
- Stored in `cycle_stats.json` as `"all_avg_similarity"` per cycle.
- Plotted on a secondary y-axis in `energy_summary.png`.

This metric helps monitor how much the generated sequences drift from the starting point over successive cycles.

#### 3.7 Compact residue notation

The `residues` field is a dictionary mapping chain names to residue specifications. Each value can be a compact range string instead of a full list of integers:

| Format | Expands to |
|---|---|
| `"5"` | `[5]` |
| `"1,2,5"` | `[1, 2, 5]` |
| `"0-43"` | `[0, 1, 2, ..., 43]` |
| `"0-5,10,20-25"` | `[0, 1, 2, 3, 4, 5, 10, 20, 21, 22, 23, 24, 25]` |

**Single-chain** energy terms (e.g. `TemplateMatchEnergy`) use only the `GEN` key:

```yaml
residues:
  GEN: "0-43"
```

**Multi-chain** energy terms (e.g. `PAEEnergy`, `LISEnergy`, `SeparationEnergy`) use `GEN` for the generated chain and the target's chain ID as the second key:

```yaml
residues:
  GEN: "0-19"
  A: "0-9"
```

This maps residues 0–19 on the generated chain (`GEN`) and residues 0–9 on the target chain (`A`). The `GEN` key is always group 0, the target key is group 1.

The explicit integer list format (e.g. `[0,1,2,3]`) and the `"all"` shorthand continue to work as values within the dict.

#### 3.8 PDB structure download

Instead of providing a local CIF/PDB file, you can specify a **PDB code** and the pipeline will download the structure from the [RCSB PDB](https://www.rcsb.org/):

```yaml
pdb_code: "1ubq"
template_chain_id: A
```

This replaces `template_structure_path`. Downloaded files are cached in `~/.cache/profam_bagel/pdb/` so they are not re-downloaded on subsequent runs.

For multi-chain energy targets, use `target_pdb_code` and `target_chain_id` instead of an inline `target` sequence:

```yaml
target_pdb_code: "1ubq"
target_chain_id: A
```

The pipeline downloads the CIF, extracts the specified chain, and uses its sequence as the target.

The utility functions `download_pdb_cif()` and `extract_chain_from_cif()` are also available for standalone use from `run_profam_bagel_pipeline`.

#### 3.9 Selection memory (`n_memory`)

By default, when selecting which sequences to inject into the next ProFam cycle, only the sequences generated in the **current** cycle are considered. The `n_memory` parameter widens this selection pool to include sequences from previous cycles:

```yaml
n_memory: 3   # include sequences from the last 3 cycles in the pool
```

| `n_memory` | Pool contents |
|---|---|
| `0` (default) | Current cycle only |
| `N > 0` | Current cycle + up to the last N cycles |

The number of sequences selected for injection stays the same (`floor(f_inject * profam_num_samples)`); only the candidate pool grows. Probabilities are computed via softmax over the entire pool, so good sequences from earlier cycles can survive even if they were not selected at the time.

Every generated sequence receives a **global unique ID** that increments monotonically across cycles. With 10 sequences per cycle: cycle 1 produces IDs 0–9, cycle 2 produces IDs 10–19, and so on. These IDs appear in `cycle_stats.json` as `"selected_ids"` and in each sequence entry's `"id"` field, making it unambiguous which sequence is which regardless of cycle of origin.

---

### 4. Running the pipeline locally

From the repository root:

```bash
conda activate profam_bagel
python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
```

Override any YAML key via CLI:

```bash
python run_profam_bagel_pipeline.py \
  --config example_pipeline_config.yaml \
  --max_cycles 5 \
  --f_inject 0.1 \
  --run_on_modal false
```

Run without a YAML file by supplying all required flags:

```bash
python run_profam_bagel_pipeline.py \
  --initial_fasta initial_sequences.fasta \
  --profam_checkpoint_dir model_checkpoints/profam-1 \
  --energy_config example_energy_template_match.yaml \
  --profam_num_samples 64 \
  --f_inject 0.25 \
  --max_cycles 10 \
  --output_dir outputs/pipeline_run1
```

> **Note:** Running locally with `run_on_modal: false` requires a GPU with enough memory for ESMFold, and the `MODEL_DIR` environment variable must point to the folder containing the ESMFold model weights (e.g. `export MODEL_DIR=~/.cache/bagel/models`).

A convenience wrapper for local runs is also available:

```bash
./run_pipeline_mac.sh example_pipeline_config.yaml [extra CLI args...]
```

---

### 5. Running the full pipeline on Modal (cloud GPU)

[Modal](https://modal.com/) lets you run the entire pipeline — ProFam generation, ESMFold folding, and energy evaluation — on cloud GPUs without managing any infrastructure.

#### Setup

1. Install and authenticate with Modal:
   ```bash
   pip install modal    # already included by setup_environment.sh
   modal token new      # opens a browser to authenticate
   ```

2. Set `run_on_modal: true` in your YAML config (this is the default in `example_pipeline_config.yaml`).

#### How it works

When `run_on_modal: true`:

- `run_profam_bagel_pipeline.py` serialises the pipeline configuration and dispatches it to a remote Modal job defined in `run_profam_bagel_modal_app.py`.
- The Modal container receives a Docker image with all dependencies pre-installed (PyTorch, transformers, boileroom, BAGEL, etc.).
- Your local repository is uploaded to `/workspace` inside the container.
- If cached ESMFold model weights exist locally (at `MODEL_DIR` or `~/.cache/bagel/models`), they are uploaded to `/models/bagel` to avoid re-downloading.
- ESMFold's `use_modal` is **forced to `true`** inside the container, regardless of the energy config setting.
- The job runs on an NVIDIA A10G GPU with a 24-hour timeout.

#### Launch

```bash
conda activate profam_bagel
python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
```

That's it — the script handles the rest. You'll see Modal's output stream in your terminal.

---

### 6. Running on a PBS/HPC cluster

For institutional HPC clusters that use the PBS job scheduler, a batch script is provided: `run_pipeline_pbs.sh`.

#### 6.1 Setting up the environment on the cluster

The same `setup_environment.sh` script works on Linux HPC nodes. The only difference is that on Linux with NVIDIA GPUs, the script installs CUDA-enabled PyTorch packages automatically.

```bash
# On the cluster login node:
git clone <this-repo-url> profam_bagel
cd profam_bagel

# Load conda (adapt to your cluster's module system)
module load anaconda3            # or: module load miniconda
# or: source ~/miniconda3/etc/profile.d/conda.sh

# Run the setup script
chmod +x setup_environment.sh
./setup_environment.sh

# Download the ProFam checkpoint
conda activate profam_bagel
python -c "from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='model_checkpoints/profam-1')"
```

If the cluster does not have internet access on compute nodes, run the setup and model download on the login node (which typically does have internet access).

#### 6.2 Choosing between local folding and Modal folding on the cluster

You have two options for ESMFold folding when running on a cluster:

| Option | YAML setting | Energy config `use_modal` | Requirements |
|---|---|---|---|
| **A. Local GPU folding** | `run_on_modal: false` | `false` | GPU node, `MODEL_DIR` set to ESMFold weights |
| **B. Modal folding** | `run_on_modal: false` | `true` | Internet access from compute nodes, Modal token configured |

**Option A** (recommended for GPU clusters): The pipeline runs entirely on the cluster node. Set `run_on_modal: false` in the pipeline YAML and `use_modal: false` in the energy YAML. You must set the `MODEL_DIR` environment variable to the directory containing ESMFold model weights:

```bash
export MODEL_DIR=/path/to/esmfold/models
```

**Option B**: The pipeline runs on the cluster but ESMFold folding is offloaded to Modal. Set `run_on_modal: false` in the pipeline YAML but `use_modal: true` in the energy YAML. This requires internet access from compute nodes and a configured Modal token (`modal token new`).

> **Note:** Setting `run_on_modal: true` on a cluster would send the *entire* pipeline (including ProFam) to Modal, which is usually not what you want on an HPC system — use Option A or B instead.

#### 6.3 Submitting the PBS job

Edit `run_pipeline_pbs.sh` to match your cluster's resource configuration and add the conda activation command:

```bash
# Inside run_pipeline_pbs.sh, uncomment and adapt:
module load anaconda3
source activate profam_bagel

# If using local folding (Option A), also add:
export MODEL_DIR=/path/to/esmfold/models
```

Then submit:

```bash
qsub run_pipeline_pbs.sh -v CONFIG=example_pipeline_config.yaml
```

The default resource request is:
- 1 node, 8 CPUs, 1 GPU, 64 GB RAM, 24-hour walltime

Adjust the `#PBS -l` directives in the script as needed for your cluster.

---

### 7. Outputs

For a run with `output_dir: outputs/pipeline_run1`, the pipeline creates:

- **`outputs/pipeline_run1/cycle_stats.json`**
  - Dictionary keyed by cycle number (as a string), e.g. `"1"`, `"2"`, ...
  - Each entry contains:
    - `cycle`: integer cycle index
    - `all_avg_energy`, `all_min_energy`: statistics over all generated sequences
    - `all_avg_similarity`: average sequence similarity to initial sequences
    - `best_sequence`: dict with the lowest-energy sequence's details
    - `selected_avg_energy`, `selected_min_energy`: statistics over the selected subset
    - `num_selected`, `selected_indices`: subset selection info
    - `selected_sequences`: list of dicts with `index`, `sequence`, `energy`, and `energy_terms`

- **Structures** (when folding was performed):
  - `outputs/pipeline_run1/cycle_XXX/sequences_cycle_all_XXX/sequence_XXXX.cif` — all sequences folded in that cycle
  - `outputs/pipeline_run1/sequences_cycle_XXX/sequence_XXXX.cif` — only the selected subset

- **Plot:**
  - `outputs/pipeline_run1/energy_summary.png` — line plot of average and minimum energies vs cycle index (left y-axis), with average sequence similarity on the right y-axis

---

### 8. Troubleshooting

| Problem | Solution |
|---|---|
| `modal.exception.AuthError: Token missing` | Run `modal token new` to authenticate |
| `AssertionError: MODEL_DIR must be set` | Set `export MODEL_DIR=~/.cache/bagel/models` (or wherever your ESMFold weights are) |
| `ImportError: lightning` or `torchvision` errors | torch/torchvision version mismatch — re-run `setup_environment.sh` or manually install matching versions (see Section 2) |
| `boileroom` not found | Ensure BAGEL was installed: `pip install "biobagel[local] @ git+https://github.com/softnanolab/bagel.git"` |
| ProFam checkpoint not found | Download with `python -c "from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='model_checkpoints/profam-1')"` |
| Slow first Modal run | First run builds the container image; subsequent runs reuse the cached image |

---

### 9. Project structure

```
profam_bagel/
├── run_profam_bagel_pipeline.py       # Main pipeline entrypoint
├── run_profam_bagel_modal_app.py      # Modal app for cloud execution
├── setup_environment.sh               # Environment setup script
├── example_pipeline_config.yaml       # Example YAML config
├── example_energy_template_match.yaml # Example energy config (template matching)
├── example_energy_lis_binding.yaml    # Example energy config (LIS binding with target chain)
├── run_pipeline_pbs.sh                # PBS cluster batch script
├── run_pipeline_mac.sh                # Local convenience wrapper
├── initial_sequences.fasta            # Example initial sequences
└── model_checkpoints/                 # ProFam model weights (user-downloaded)
```

**External dependencies** (installed via pip from GitHub):
- **BAGEL** (`biobagel`): `pip install "biobagel[local] @ git+https://github.com/softnanolab/bagel.git"` — structure prediction + energy terms
- **ProFam**: `pip install "git+https://github.com/alex-hh/profam.git"` — protein sequence generation
