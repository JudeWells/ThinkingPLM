#!/usr/bin/env python
"""
End-to-end ProFam + BAGEL generative pipeline.

High-level protocol (per cycle)
------------------------------
1) Use ProFam to generate N_output sequences from an initial FASTA (plus, after
   the first cycle, injected sequences from the previous cycle).
2) For each generated sequence, run BAGEL's folding oracle and compute a
   user-defined weighted energy using a YAML energy configuration.
3) Turn energies into sampling probabilities via a softmax over -energy.
4) Sample with replacement a fraction f_inject of the N_output sequences.
   For the selected subset, save:
   - Cycle number, average and minimum energy, and all selected sequences +
     energies into a JSON log keyed by cycle index.
   - The folded structures for the selected sequences into a folder
     `sequences_cycle_<cycle>`, as CIF files written by BAGEL.
5) Repeat for max_cycles, then write a summary plot of average and minimum
   energies versus cycle index.

All required inputs can be provided via a YAML config file, via CLI flags,
or a combination (YAML + CLI overrides).

Expected YAML schema (flat keys)
--------------------------------
initial_fasta: path/to/initial.fasta
profam_checkpoint_dir: path/to/profam/checkpoint_dir
profam_sampler: "single"            # or "ensemble" (optional, default: single)
profam_num_samples: 64              # N_output
profam_max_tokens: 8192            # optional
profam_max_generated_length: null   # optional
profam_temperature: 0.7             # optional
profam_top_p: 0.95                  # optional
energy_config: path/to/energy.yaml
f_inject: 0.25
max_cycles: 10
output_dir: outputs/pipeline_run1
softmax_temperature: 1.0            # optional, default 1.0
random_seed: 42                     # optional, default 42

Energy YAML schema (flexible, minimal)
--------------------------------------
The energy YAML file is expected to look like:

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

All entries in "kwargs" are passed directly to the corresponding BAGEL
energy term __init__ (with the exception that `oracle` is injected
automatically, and optional "residues" specs are converted to lists
of `bagel.Residue` objects based on the current chain sequence).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from biotite.structure.io.pdb import PDBFile  # type: ignore
from biotite.structure.io.pdbx import CIFFile, get_structure  # type: ignore

import pipeline.custom_energies  # noqa: F401  # registers custom energies into bagel.energies
from pipeline.bandits import (
    ProposalBandit,
    TemperatureBandit,
    ThompsonSampler,
)
from pipeline.logging import (
    append_cycle_csv,
    save_selected_structures,
    update_cycle_log,
)
from pipeline.plotting import make_energy_summary_plot
from pipeline.proposal import (
    ProFamProposalGenerator,
    RandomMutationProposalGenerator,
)
from pipeline.selection import (
    GreedyPromptSelector,
    SelectionManager,
    ThompsonPromptSelector,
)
from pipeline.utils import (
    compute_avg_sequence_similarity,
    sample_subset_indices,
    softmax_from_energies,
)

try:
  import yaml  # type: ignore
except ImportError as e:  # pragma: no cover - import-time check
  raise ImportError(
    "PyYAML is required to run the pipeline. Install it with `pip install pyyaml`."
  ) from e

# BAGEL — installed via: pip install git+https://github.com/softnanolab/bagel.git
import bagel as bg  # type: ignore
from bagel.oracles import ESMFold  # type: ignore
from bagel.oracles.folding import FoldingOracle, AlphaFast, Boltz  # type: ignore
from bagel.oracles.folding.utils import sequence_from_atomarray  # type: ignore

# ProFam — installed via: pip install git+https://github.com/alex-hh/profam.git
from src.data.objects import ProteinDocument  # type: ignore
from src.data.processors.preprocessing import (  # type: ignore
  AlignedProteinPreprocessingConfig,
  ProteinDocumentPreprocessor,
)
from src.models.inference import (  # type: ignore
  EnsemblePromptBuilder,
  ProFamEnsembleSampler,
  ProFamSampler,
  PromptBuilder,
)
from src.models.llama import LlamaLitModule  # type: ignore
from src.sequence.fasta import read_fasta, output_fasta  # type: ignore
from src.utils.utils import seed_all  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent




# ---------------------------------------------------------------------------
# Config dataclass & CLI/YAML handling
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
  initial_fasta: Path | None
  profam_checkpoint_dir: Path
  profam_sampler: str = "single"
  profam_num_samples: int = 10
  profam_max_tokens: int = 8192
  profam_max_generated_length: int | None = None
  profam_temperature: float | None = None
  profam_top_p: float | None = 0.95
  profam_generation_batch_size: int | None = None  # defaults to profam_num_samples

  energy_config: Path = Path("energy.yaml")

  f_inject: float = 0.5
  max_cycles: int = 5
  output_dir: Path = Path("pipeline_outputs")
  softmax_temperature: float = 1.0
  random_seed: int = 42
  run_on_modal: bool = False
  enforce_template: bool = True
  output_frequency: int = 1
  sample_with_reinsertion: bool = True
  reinject_initial: bool = True
  n_memory: int = 0
  elitism: bool = False
  accept_only_improvement: bool = False
  annealing_initial_temp: float | None = None
  annealing_decay: float = 0.95

  proposal_method: str = "profam"  # "profam" or "random_mutation"
  max_mutations: int = 5
  freeze_prompt: bool = False

  selection_strategy: str = "greedy"       # "greedy" or "thompson"
  thompson_m_samples: int = 1             # max-seeking: sample m times per arm
  thompson_exploit_bias: float = 1.0      # >1 = more exploitation (concentrate posteriors)
  thompson_temperature_bins: List[float] | None = None  # e.g. [0.6, 0.8, 1.0, 1.3]; None = fixed temperature
  thompson_proposal_bandit: bool = False   # True = Thompson bandit over proposal methods (profam vs random_mutation)
  proposal_bandit_prior_alpha: float = 2.0  # Beta prior α for proposal bandit (higher = more confident prior)
  proposal_bandit_prior_beta: float = 2.0   # Beta prior β for proposal bandit
  proposal_bandit_relative_reward: bool = False  # Use relative reward (improvement over parent) instead of absolute ipSAE
  thompson_max_arms: int = 0              # max arms to retain (0 = unlimited); prunes to top-K diverse arms
  thompson_max_identity: float = 0.95     # max sequence identity between retained arms (diversity threshold)
  deduplicate_sequences: bool = True       # skip folding for already-seen sequences, retry generation
  save_structures: bool = False            # save CIF/PAE/PLDDT files per cycle (disk-heavy, disable for large runs)

  random_init: bool = False                # if True, generate a random initial sequence instead of reading from FASTA
  random_init_max_residues: int = 80       # max length of randomly generated initial sequence

  # GRPO (online RL fine-tuning of ProFam).  There is no explicit
  # group-size knob: the effective GRPO batch is the full replay buffer,
  # size = profam_num_samples * (grpo_replay_cycles + 1).
  grpo_enabled: bool = False
  grpo_beta: float = 0.05
  grpo_clip_ratio: float = 0.2
  grpo_lr: float = 1e-5
  grpo_weight_decay: float = 0.01
  grpo_temperature: float = 1.0
  grpo_top_p: float = 0.95
  grpo_max_tokens: int = 8000
  grpo_normalize_rewards: bool = True
  grpo_reward_baseline: str = "mean"       # "mean" | "min" | "none"
  grpo_use_reference_model: bool = False
  rl_every_n_cycles: int = 1
  rl_steps_per_cycle: int = 1
  grpo_replay_cycles: int = 7    # number of past cycles cached for larger effective group
  grpo_micro_batch_size: int = 4  # sequences per micro-batch for gradient accumulation

  # Likelihood tracking: evaluate model log-likelihood of best/worst sequences
  likelihood_eval_every: int = 0    # evaluate every N cycles (0 = disabled)
  likelihood_track_n: int = 10      # number of best/worst sequences to track

  # Bradley-Terry ranking loss (alternative to GRPO)
  bt_enabled: bool = False
  bt_lr: float = 1e-5
  bt_pool_size: int = 64            # max sequences in the ranking pool
  bt_batch_size: int = 32           # sequences per BT training batch
  bt_sub_batch_size: int = 4        # sub-batch for forward pass (memory)
  bt_every_n_cycles: int = 1        # train every N cycles
  bt_steps_per_cycle: int = 1       # gradient steps per training invocation

  # Weights & Biases logging
  wandb_enabled: bool = False
  wandb_project: str = "profam-bagel-pipeline"
  wandb_entity: str | None = None
  wandb_run_name: str | None = None    # defaults to output_dir stem
  wandb_tags: List[str] | None = None


def _to_path(x: Any) -> Path:
  return x if isinstance(x, Path) else Path(str(x))


def load_yaml_config(path: Path | None) -> Dict[str, Any]:
  if path is None:
    return {}
  with path.open("r") as f:
    data = yaml.safe_load(f) or {}
  if not isinstance(data, dict):
    raise ValueError(f"YAML config at {path} must define a mapping at top level.")
  return data


def merge_config(yaml_cfg: Dict[str, Any], args: argparse.Namespace) -> PipelineConfig:
  """
  Merge YAML config with CLI arguments. CLI flags (if provided) override YAML.
  """

  def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
      return val
    if isinstance(val, str):
      return val.lower() in ("true", "1", "yes")
    return bool(val)

  def pick(name: str, default: Any = None) -> Any:
    cli_val = getattr(args, name, None)
    if cli_val is not None:
      return cli_val
    if name in yaml_cfg and yaml_cfg[name] is not None:
      return yaml_cfg[name]
    return default

  _init_fasta_raw = pick("initial_fasta", None)
  cfg = PipelineConfig(
    initial_fasta=_to_path(_init_fasta_raw) if _init_fasta_raw else None,
    profam_checkpoint_dir=_to_path(pick("profam_checkpoint_dir", ".")),
    profam_sampler=str(pick("profam_sampler", "single")),
    profam_num_samples=int(pick("profam_num_samples", 10)),
    profam_max_tokens=int(pick("profam_max_tokens", 8192)),
    profam_max_generated_length=(
      None
      if pick("profam_max_generated_length", None) is None
      else int(pick("profam_max_generated_length"))
    ),
    profam_temperature=(
      None
      if pick("profam_temperature", None) is None
      else float(pick("profam_temperature"))
    ),
    profam_top_p=(
      None
      if pick("profam_top_p", None) is None
      else float(pick("profam_top_p"))
    ),
    profam_generation_batch_size=(
      None
      if pick("profam_generation_batch_size", None) is None
      else int(pick("profam_generation_batch_size"))
    ),
    energy_config=_to_path(pick("energy_config")),
    f_inject=float(pick("f_inject", 0.5)),
    max_cycles=int(pick("max_cycles", 5)),
    output_dir=_to_path(pick("output_dir", "pipeline_outputs")),
    softmax_temperature=float(pick("softmax_temperature", 1.0)),
    random_seed=int(pick("random_seed", 42)),
    run_on_modal=_to_bool(pick("run_on_modal", False)),
    enforce_template=_to_bool(pick("enforce_template", True)),
    output_frequency=int(pick("output_frequency", 1)),
    sample_with_reinsertion=_to_bool(pick("sample_with_reinsertion", True)),
    reinject_initial=_to_bool(pick("reinject_initial", True)),
    n_memory=int(pick("n_memory", 0)),
    elitism=_to_bool(pick("elitism", False)),
    accept_only_improvement=_to_bool(pick("accept_only_improvement", False)),
    annealing_initial_temp=(
      None
      if pick("annealing_initial_temp", None) is None
      else float(pick("annealing_initial_temp"))
    ),
    annealing_decay=float(pick("annealing_decay", 0.95)),
    proposal_method=str(pick("proposal_method", "profam")),
    max_mutations=int(pick("max_mutations", 5)),
    freeze_prompt=_to_bool(pick("freeze_prompt", False)),
    selection_strategy=str(pick("selection_strategy", "greedy")),
    thompson_m_samples=int(pick("thompson_m_samples", 1)),
    thompson_exploit_bias=float(pick("thompson_exploit_bias", 1.0)),
    thompson_temperature_bins=(
      None
      if pick("thompson_temperature_bins", None) is None
      else [float(x) for x in pick("thompson_temperature_bins")]
    ),
    thompson_proposal_bandit=_to_bool(pick("thompson_proposal_bandit", False)),
    proposal_bandit_prior_alpha=float(pick("proposal_bandit_prior_alpha", 2.0)),
    proposal_bandit_prior_beta=float(pick("proposal_bandit_prior_beta", 2.0)),
    proposal_bandit_relative_reward=_to_bool(pick("proposal_bandit_relative_reward", False)),
    thompson_max_arms=int(pick("thompson_max_arms", 0)),
    thompson_max_identity=float(pick("thompson_max_identity", 0.95)),
    deduplicate_sequences=_to_bool(pick("deduplicate_sequences", True)),
    save_structures=_to_bool(pick("save_structures", False)),
    random_init=_to_bool(pick("random_init", False)),
    random_init_max_residues=int(pick("random_init_max_residues", 80)),
    # GRPO fields
    grpo_enabled=_to_bool(pick("grpo_enabled", False)),
    grpo_beta=float(pick("grpo_beta", 0.05)),
    grpo_clip_ratio=float(pick("grpo_clip_ratio", 0.2)),
    grpo_lr=float(pick("grpo_lr", 1e-5)),
    grpo_weight_decay=float(pick("grpo_weight_decay", 0.01)),
    grpo_temperature=float(pick("grpo_temperature", 1.0)),
    grpo_top_p=float(pick("grpo_top_p", 0.95)),
    grpo_max_tokens=int(pick("grpo_max_tokens", 8000)),
    grpo_normalize_rewards=_to_bool(pick("grpo_normalize_rewards", True)),
    grpo_reward_baseline=str(pick("grpo_reward_baseline", "mean")),
    grpo_use_reference_model=_to_bool(pick("grpo_use_reference_model", False)),
    rl_every_n_cycles=int(pick("rl_every_n_cycles", 1)),
    rl_steps_per_cycle=int(pick("rl_steps_per_cycle", 1)),
    grpo_replay_cycles=int(pick("grpo_replay_cycles", 7)),
    grpo_micro_batch_size=int(pick("grpo_micro_batch_size", 4)),
    likelihood_eval_every=int(pick("likelihood_eval_every", 0)),
    likelihood_track_n=int(pick("likelihood_track_n", 10)),
    bt_enabled=_to_bool(pick("bt_enabled", False)),
    bt_lr=float(pick("bt_lr", 1e-5)),
    bt_pool_size=int(pick("bt_pool_size", 64)),
    bt_batch_size=int(pick("bt_batch_size", 32)),
    bt_sub_batch_size=int(pick("bt_sub_batch_size", 4)),
    bt_every_n_cycles=int(pick("bt_every_n_cycles", 1)),
    bt_steps_per_cycle=int(pick("bt_steps_per_cycle", 1)),
    # wandb
    wandb_enabled=_to_bool(pick("wandb_enabled", False)),
    wandb_project=str(pick("wandb_project", "profam-bagel-pipeline")),
    wandb_entity=pick("wandb_entity", None),
    wandb_run_name=pick("wandb_run_name", None),
    wandb_tags=pick("wandb_tags", None),
  )

  if cfg.proposal_method not in ("profam", "random_mutation"):
    raise ValueError(
      f"proposal_method must be 'profam' or 'random_mutation', got '{cfg.proposal_method}'"
    )
  if cfg.selection_strategy not in ("greedy", "thompson"):
    raise ValueError(
      f"selection_strategy must be 'greedy' or 'thompson', got '{cfg.selection_strategy}'"
    )
  if cfg.max_mutations < 1:
    raise ValueError(f"max_mutations must be >= 1, got {cfg.max_mutations}")

  if not 0.0 < cfg.f_inject <= 1.0:
    raise ValueError(f"f_inject must be in (0, 1], got {cfg.f_inject}")
  if cfg.profam_num_samples <= 0:
    raise ValueError("profam_num_samples (N_output) must be > 0.")
  if cfg.random_init:
    if cfg.random_init_max_residues < 1:
      raise ValueError(f"random_init_max_residues must be >= 1, got {cfg.random_init_max_residues}")
  elif cfg.initial_fasta is None or not cfg.initial_fasta.is_file():
    raise FileNotFoundError(
      f"Initial FASTA not found: {cfg.initial_fasta}. "
      f"Provide a valid initial_fasta or set random_init: true."
    )
  if cfg.proposal_method == "profam" and not cfg.profam_checkpoint_dir.is_dir():
    raise FileNotFoundError(f"ProFam checkpoint_dir not found: {cfg.profam_checkpoint_dir}")
  if not cfg.energy_config.is_file():
    raise FileNotFoundError(f"Energy config not found: {cfg.energy_config}")

  return cfg


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Run a ProFam + BAGEL generative design pipeline."
  )
  p.add_argument(
    "--config",
    type=str,
    default=None,
    help="YAML config file (optional; CLI flags override it).",
  )

  # Core required parameters (can be supplied via YAML or CLI).
  p.add_argument("--initial_fasta", type=str, help="Initial input sequences (FASTA).")
  p.add_argument(
    "--profam_checkpoint_dir",
    type=str,
    help="ProFam checkpoint run directory (contains .hydra & checkpoints).",
  )
  p.add_argument(
    "--energy_config",
    type=str,
    help="YAML config describing BAGEL folding oracle and energy terms.",
  )

  # ProFam sampling-related.
  p.add_argument(
    "--profam_sampler",
    type=str,
    choices=["single", "ensemble"],
    help="ProFam sampler type (default: single).",
  )
  p.add_argument(
    "--profam_num_samples",
    type=int,
    help="Number of sequences to generate per ProFam call (N_output).",
  )
  p.add_argument(
    "--profam_max_tokens",
    type=int,
    help="Max tokens for ProFam sampling (default: 8192).",
  )
  p.add_argument(
    "--profam_max_generated_length",
    type=int,
    help="Cap on generated length (optional).",
  )
  p.add_argument(
    "--profam_temperature",
    type=float,
    help="Sampling temperature (optional).",
  )
  p.add_argument(
    "--profam_top_p",
    type=float,
    help="Nucleus sampling probability mass (optional).",
  )
  p.add_argument(
    "--profam_generation_batch_size",
    type=int,
    help="Batch size for parallel sequence generation (default: profam_num_samples).",
  )

  # Pipeline controls.
  p.add_argument(
    "--f_inject",
    type=float,
    help="Fraction of ProFam outputs to inject back each cycle (0 < f <= 1).",
  )
  p.add_argument(
    "--max_cycles",
    type=int,
    help="Number of pipeline cycles to run.",
  )
  p.add_argument(
    "--output_dir",
    type=str,
    help="Directory in which to store all pipeline outputs.",
  )
  p.add_argument(
    "--softmax_temperature",
    type=float,
    help="Temperature used when converting energies to sampling probabilities.",
  )
  p.add_argument(
    "--random_seed",
    type=int,
    help="Random seed for reproducible sampling.",
  )
  p.add_argument(
    "--enforce_template",
    type=str,
    default=None,
    help=(
      "If true, force template-matching residues during ProFam generation. "
      "If false, allow free generation and assign inf energy on mismatch."
    ),
  )
  p.add_argument(
    "--output_frequency",
    type=int,
    default=None,
    help=(
      "When running on Modal, push results back to local machine every "
      "output_frequency cycles (and at the end). Default: 1 (every cycle)."
    ),
  )
  p.add_argument(
    "--sample_with_reinsertion",
    type=str,
    default=None,
    help=(
      "If true (default), sample injected sequences with replacement "
      "(a sequence can appear multiple times). If false, sample without "
      "replacement; when not enough candidates have finite energy, "
      "fall back to reinjecting only the best candidate."
    ),
  )
  p.add_argument(
    "--reinject_initial",
    type=str,
    default=None,
    help=(
      "If true (default), reinject the initial FASTA sequences alongside "
      "the selected subset as ProFam input each cycle. If false, only "
      "reinject the selected subset from the previous cycle's generation."
    ),
  )
  p.add_argument(
    "--n_memory",
    type=int,
    default=None,
    help=(
      "Number of previous cycles whose generated sequences are included in "
      "the selection pool alongside the current cycle's sequences. "
      "0 = only use the current cycle (default). When > 0, sequences from "
      "up to the last n_memory cycles are pooled together before softmax "
      "selection, allowing good candidates from earlier cycles to survive."
    ),
  )
  p.add_argument(
    "--elitism",
    type=str,
    default=None,
    help=(
      "If true, track the global best sequence (lowest energy ever seen) "
      "and guarantee it a slot at position 0 in the injection set each cycle."
    ),
  )
  p.add_argument(
    "--accept_only_improvement",
    type=str,
    default=None,
    help=(
      "If true, only swap the injection set when the new candidate set's "
      "best energy improves over the previous cycle's injection set."
    ),
  )
  p.add_argument(
    "--annealing_initial_temp",
    type=float,
    default=None,
    help=(
      "Initial temperature for simulated annealing when accept_only_improvement "
      "is true. If set, worse swaps are accepted with probability exp(-delta/T). "
      "Temperature decays each cycle by annealing_decay."
    ),
  )
  p.add_argument(
    "--annealing_decay",
    type=float,
    default=None,
    help=(
      "Decay factor for simulated annealing temperature each cycle "
      "(default 0.95). Only used when annealing_initial_temp is set."
    ),
  )

  p.add_argument(
    "--proposal_method",
    type=str,
    default=None,
    choices=["profam", "random_mutation"],
    help=(
      "Sequence proposal method: 'profam' (default) uses ProFam language model, "
      "'random_mutation' generates candidates via random amino acid substitutions."
    ),
  )
  p.add_argument(
    "--max_mutations",
    type=int,
    default=None,
    help=(
      "Maximum number of point mutations per candidate when proposal_method "
      "is 'random_mutation' (default 5). Each candidate receives between 1 "
      "and max_mutations mutations uniformly at random."
    ),
  )
  p.add_argument(
    "--freeze_prompt",
    type=str,
    default=None,
    help=(
      "If true, keep the ProFam prompt (injection set) frozen at its initial "
      "state across all cycles. The pipeline still generates and scores sequences "
      "but never updates the prompt. Useful as a baseline to test whether prompt "
      "updating improves results."
    ),
  )

  # Thompson sampling.
  p.add_argument(
    "--selection_strategy",
    type=str,
    default=None,
    choices=["greedy", "thompson"],
    help=(
      "Selection strategy for choosing conditioning sequences: 'greedy' (default) "
      "uses softmax sampling with elitism/swap, 'thompson' uses Thompson sampling "
      "with Beta posteriors to learn which sequences produce the best progeny."
    ),
  )
  p.add_argument(
    "--thompson_m_samples",
    type=int,
    default=None,
    help=(
      "Max-seeking variant: sample m times from each arm's Beta posterior and "
      "take the max. Higher values bias toward high-variance arms. Default: 1."
    ),
  )
  p.add_argument(
    "--thompson_exploit_bias",
    type=float,
    default=None,
    help=(
      "Exploitation bias for Thompson sampling. Scales α and β by this factor "
      "before sampling θ, concentrating the Beta distribution around its mean. "
      "1.0 = standard Thompson (default), 5.0 = strongly exploitative, "
      "10.0 = nearly greedy."
    ),
  )
  p.add_argument(
    "--thompson_temperature_bins",
    type=float,
    nargs="+",
    default=None,
    help=(
      "Discrete temperature values for the temperature bandit. When set, "
      "Thompson sampling selects a temperature each cycle alongside the "
      "conditioning sequence. E.g.: --thompson_temperature_bins 0.6 0.8 1.0 1.3"
    ),
  )
  p.add_argument(
    "--deduplicate_sequences",
    type=str,
    default=None,
    help=(
      "If true, skip structure prediction for generated sequences that are "
      "identical to previously seen sequences (reuse cached energies) and "
      "retry generation to obtain novel sequences. Default: true."
    ),
  )

  # Random initialization.
  p.add_argument(
    "--random_init",
    type=str,
    default=None,
    help=(
      "If true, generate a random initial protein sequence instead of reading "
      "from initial_fasta. Useful for benchmarking without a scaffold."
    ),
  )
  p.add_argument(
    "--random_init_max_residues",
    type=int,
    default=None,
    help=(
      "Maximum number of residues for the randomly generated initial sequence "
      "when random_init is true (default: 80)."
    ),
  )

  # GRPO (online RL fine-tuning).
  p.add_argument("--grpo_enabled", type=str, default=None,
                  help="Enable GRPO online RL fine-tuning of ProFam model weights.")
  p.add_argument("--grpo_beta", type=float, default=None,
                  help="KL penalty coefficient for GRPO (default: 0.05).")
  p.add_argument("--grpo_clip_ratio", type=float, default=None,
                  help="PPO-style clipping epsilon (default: 0.2).")
  p.add_argument("--grpo_lr", type=float, default=None,
                  help="Learning rate for GRPO optimizer (default: 1e-5).")
  p.add_argument("--grpo_temperature", type=float, default=None,
                  help="Sampling temperature during GRPO generation (default: 1.0).")
  p.add_argument("--rl_every_n_cycles", type=int, default=None,
                  help="Run GRPO every N pipeline cycles (default: 1).")
  p.add_argument("--rl_steps_per_cycle", type=int, default=None,
                  help="Gradient steps per GRPO invocation (default: 1).")

  # Weights & Biases.
  p.add_argument("--wandb_enabled", type=str, default=None,
                  help="Enable W&B logging (default: false).")
  p.add_argument("--wandb_project", type=str, default=None,
                  help="W&B project name (default: profam-bagel-pipeline).")
  p.add_argument("--wandb_entity", type=str, default=None,
                  help="W&B entity/team name.")
  p.add_argument("--wandb_run_name", type=str, default=None,
                  help="W&B run name (default: output_dir stem).")

  return p


# ---------------------------------------------------------------------------
# ProFam integration — direct API calls (model loaded once, reused)
# ---------------------------------------------------------------------------


def load_profam_model(cfg: PipelineConfig) -> Tuple[Any, str]:
  """
  Load the ProFam model from checkpoint.  Called once at pipeline start;
  the returned (model, device) tuple is passed to ``run_profam_generation``
  on each cycle to avoid reloading from disk.
  """
  ckpt_path = cfg.profam_checkpoint_dir / "checkpoints" / "last.ckpt"
  if not ckpt_path.is_file():
    raise FileNotFoundError(
      f"ProFam checkpoint not found at {ckpt_path}. "
      "Run the download script or check profam_checkpoint_dir."
    )

  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = torch.bfloat16

  # Detect best attention implementation
  attn_impl = "sdpa"
  try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
  except ImportError:
    pass

  # Load checkpoint and override attention implementation
  print(f"Loading ProFam model from {ckpt_path} (device={device}, attn={attn_impl})...")
  ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  hyper_params = ckpt_blob.get("hyper_parameters", {})
  cfg_obj = hyper_params.get("config")
  if cfg_obj is None:
    raise RuntimeError(
      "Could not find 'config' in checkpoint hyper_parameters "
      "to override attention implementation."
    )
  setattr(cfg_obj, "attn_implementation", attn_impl)
  setattr(cfg_obj, "_attn_implementation", attn_impl)

  model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
    ckpt_path, config=cfg_obj, strict=False, weights_only=False,
  )
  model.eval()
  model.to(device, dtype=dtype)

  seed_all(cfg.random_seed)
  print("ProFam model loaded successfully.")

  return model, device


def run_profam_generation(
  cfg: PipelineConfig,
  input_fasta: Path,
  cycle_dir: Path,
  model: Any,
  device: str,
  capture_grpo_tokens: bool = False,
) -> Tuple[List[str], List[str]] | Tuple[List[str], List[str], Dict[str, Any]]:
  """
  Generate sequences using ProFam's Python API.

  This calls the sampler directly (no subprocess), reusing the model loaded
  once by ``load_profam_model()``.

  When ``capture_grpo_tokens=True``, also returns a dict with
  ``input_ids``, ``generated_tokens``, ``old_per_token_lps``, and
  ``old_per_token_mask`` for use in GRPO training.
  """
  # Build a ProteinDocument from the input FASTA.
  names, seqs = read_fasta(
    str(input_fasta), keep_insertions=True, to_upper=True, keep_gaps=False,
  )
  rep = names[0] if len(names) > 0 else "representative"
  pool = ProteinDocument(
    sequences=seqs,
    accessions=names,
    identifier=input_fasta.stem,
    representative_accession=rep,
  )

  # Compute generation length cap.
  longest_prompt_len = int(max(pool.sequence_lengths))
  max_sequence_length_multiplier = 1.2
  default_cap = int(longest_prompt_len * max_sequence_length_multiplier)
  if cfg.profam_max_generated_length is None:
    max_gen_len = default_cap
  else:
    max_gen_len = min(int(cfg.profam_max_generated_length), default_cap)

  doc_token = "[RAW]"

  # Build preprocessor and sampler.
  if cfg.profam_sampler == "ensemble":
    preproc_cfg = AlignedProteinPreprocessingConfig(
      document_token=doc_token,
      defer_sampling=True,
      padding="do_not_pad",
      shuffle_proteins_in_document=True,
      keep_insertions=True,
      to_upper=True,
      keep_gaps=False,
      use_msa_pos=False,
      max_tokens_per_example=None,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=preproc_cfg)
    builder = EnsemblePromptBuilder(
      preprocessor=preprocessor, shuffle=True, seed=cfg.random_seed,
    )
    sampler_obj = ProFamEnsembleSampler(
      name="ensemble_sampler",
      model=model,
      prompt_builder=builder,
      document_token=doc_token,
      reduction="mean_probs",
      temperature=cfg.profam_temperature,
      top_p=cfg.profam_top_p,
      add_final_sep=True,
    )
    sampler_obj.to(device)
    sequences, scores, _ = sampler_obj.sample_seqs_ensemble(
      protein_document=pool,
      num_samples=cfg.profam_num_samples,
      max_tokens=cfg.profam_max_tokens,
      num_prompts_in_ensemble=min(8, len(pool.sequences)),
      max_generated_length=max_gen_len,
      continuous_sampling=False,
      minimum_sequence_length_proportion=0.5,
      minimum_sequence_identity=None,
      maximum_retries=5,
      repeat_guard=True,
    )
  else:
    preproc_cfg = AlignedProteinPreprocessingConfig(
      document_token=doc_token,
      defer_sampling=False,
      padding="do_not_pad",
      shuffle_proteins_in_document=True,
      keep_insertions=True,
      to_upper=True,
      keep_gaps=False,
      use_msa_pos=False,
      max_tokens_per_example=cfg.profam_max_tokens - max_gen_len,
    )
    preprocessor = ProteinDocumentPreprocessor(cfg=preproc_cfg)
    builder = PromptBuilder(
      preprocessor=preprocessor, prompt_is_aligned=True, seed=cfg.random_seed,
    )
    sampling_kwargs: Dict[str, Any] = {}
    if cfg.profam_top_p is not None:
      sampling_kwargs["top_p"] = cfg.profam_top_p
    if cfg.profam_temperature is not None:
      sampling_kwargs["temperature"] = cfg.profam_temperature
    # Enable batched generation: generate all samples in parallel batches
    # rather than one-at-a-time.  Defaults batch size to profam_num_samples.
    gen_batch_size = cfg.profam_generation_batch_size or cfg.profam_num_samples
    sampling_kwargs["batch_generation"] = True
    sampling_kwargs["generation_batch_size"] = gen_batch_size
    sampler_obj = ProFamSampler(
      name="single_sampler",
      model=model,
      prompt_builder=builder,
      document_token=doc_token,
      sampling_kwargs=sampling_kwargs if sampling_kwargs else None,
      add_final_sep=True,
    )
    sampler_obj.to(device)
    sample_kwargs: Dict[str, Any] = dict(
      protein_document=pool,
      num_samples=cfg.profam_num_samples,
      max_tokens=cfg.profam_max_tokens,
      max_generated_length=max_gen_len,
      continuous_sampling=False,
      minimum_sequence_length_proportion=0.5,
      minimum_sequence_identity=None,
      maximum_retries=5,
      repeat_guard=True,
    )
    sequences, scores, _ = sampler_obj.sample_seqs(**sample_kwargs)

  # Build accession names (matching the format used by generate_sequences.py).
  base = input_fasta.stem
  accessions = [
    f"{base}_sample_{i}_log_likelihood_{score:.3f}"
    for i, score in enumerate(scores)
  ]

  # Optionally save generated FASTA for debugging/reproducibility.
  profam_out_dir = cycle_dir / "profam_outputs"
  profam_out_dir.mkdir(parents=True, exist_ok=True)
  out_fasta = profam_out_dir / f"{base}_generated_{cfg.profam_sampler}.fasta"
  output_fasta(accessions, sequences, str(out_fasta))

  if not capture_grpo_tokens:
    return list(accessions), list(sequences)

  # Capture GRPO token data: re-tokenize the prompt and generated sequences
  # so that grpo_step_from_rewards can compute proper importance ratios.
  tok = model.tokenizer
  sep_id = tok.sep_token_id
  pad_id = tok.pad_token_id

  # Build prompt input_ids: [BOS] [doc_token] [SEP] seq1 [SEP] seq2 [SEP] ...
  prompt_token_ids = [tok.bos_token_id, tok.convert_tokens_to_ids(doc_token)]
  for seq in seqs:  # original prompt sequences from FASTA
    prompt_token_ids.append(sep_id)
    for aa in seq:
      prompt_token_ids.append(tok.convert_tokens_to_ids(aa))
  prompt_token_ids.append(sep_id)
  # Truncate if needed
  max_prompt_tokens = cfg.profam_max_tokens - max_gen_len
  if len(prompt_token_ids) > max_prompt_tokens:
    prompt_token_ids = prompt_token_ids[:max_prompt_tokens]
  grpo_input_ids = torch.tensor([prompt_token_ids], device=device)

  # Tokenize generated sequences as completion tokens (no SEP prefix — raw AA tokens + SEP)
  gen_token_lists = []
  for seq in sequences:
    toks = []
    for aa in seq:
      toks.append(tok.convert_tokens_to_ids(aa))
    toks.append(sep_id)
    gen_token_lists.append(toks)

  # Pad to same length
  max_gen_tok_len = max(len(t) for t in gen_token_lists) if gen_token_lists else 0
  padded_gen = [t + [pad_id] * (max_gen_tok_len - len(t)) for t in gen_token_lists]
  generated_tokens = torch.tensor(padded_gen, dtype=torch.long)  # (G, L_gen) on CPU

  # Compute old per-token log-probs under current policy (no gradients)
  # Format: strip trailing SEP from prompt, prepend SEP to completions
  input_ids_for_scoring = grpo_input_ids[:, :-1] if int(grpo_input_ids[0, -1].item()) == sep_id else grpo_input_ids
  sep_prefix = torch.full((generated_tokens.shape[0], 1), sep_id, dtype=torch.long, device=device)
  completion_ids = torch.cat([sep_prefix, generated_tokens.to(device)], dim=1).unsqueeze(0)  # (1, G, 1+L_gen)

  with torch.no_grad():
    old_per_token_lps, old_per_token_mask = model._compute_per_token_log_probs_for_grpo(
      input_ids=input_ids_for_scoring,
      completion_ids=completion_ids,
    )

  grpo_data = {
    "input_ids": grpo_input_ids,           # (1, L_prompt)
    "generated_tokens": generated_tokens,  # (G, L_gen) on CPU
    "old_per_token_lps": old_per_token_lps.cpu(),   # (G, L_completion-1)
    "old_per_token_mask": old_per_token_mask.cpu(),  # (G, L_completion-1)
  }
  return list(accessions), list(sequences), grpo_data


# ---------------------------------------------------------------------------
# Random mutation proposer
# ---------------------------------------------------------------------------

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def run_random_mutation_generation(
  seed_sequences: List[str],
  num_samples: int,
  max_mutations: int,
  rng: np.random.Generator,
) -> Tuple[List[str], List[str]]:
  """
  Generate candidate sequences by introducing random point mutations.

  For each candidate, a seed sequence is picked uniformly at random from
  *seed_sequences*.  Between 1 and *max_mutations* positions (inclusive) are
  then chosen uniformly without replacement and each is replaced with a
  uniformly sampled amino acid (which may be the same as the original).

  Parameters
  ----------
  seed_sequences : list of str
      One or more parent sequences to mutate.
  num_samples : int
      Number of mutant candidates to produce.
  max_mutations : int
      Upper bound on the number of substitutions per candidate.
  rng : numpy.random.Generator
      Reproducible random state.

  Returns
  -------
  names : list of str
      Accession-style labels for each candidate.
  sequences : list of str
      The mutated amino acid sequences.
  """
  aa_list = list(STANDARD_AMINO_ACIDS)
  names: List[str] = []
  sequences: List[str] = []

  for i in range(num_samples):
    parent = seed_sequences[rng.integers(len(seed_sequences))]
    seq = list(parent)
    n_mut = int(rng.integers(1, max_mutations + 1))
    n_mut = min(n_mut, len(seq))
    positions = rng.choice(len(seq), size=n_mut, replace=False)

    mutations_desc: List[str] = []
    for pos in positions:
      old_aa = seq[pos]
      new_aa = aa_list[rng.integers(len(aa_list))]
      mutations_desc.append(f"{old_aa}{pos + 1}{new_aa}")
      seq[pos] = new_aa

    mutant = "".join(seq)
    names.append(f"random_mutant_{i}_{'+'.join(mutations_desc)}")
    sequences.append(mutant)

  return names, sequences


# ---------------------------------------------------------------------------
# BAGEL integration: energies & folding
# ---------------------------------------------------------------------------


def load_energy_config(path: Path) -> Dict[str, Any]:
  with path.open("r") as f:
    cfg = yaml.safe_load(f)
  if not isinstance(cfg, dict):
    raise ValueError(f"Energy config at {path} must define a dictionary at top level.")
  return cfg


# --- PDB download and chain extraction utilities ---

_PDB_CACHE_DIR = Path.home() / ".cache" / "profam_bagel" / "pdb"


def download_pdb_cif(pdb_code: str, cache_dir: Path | None = None) -> Path:
  """
  Download a CIF structure file from the RCSB PDB.

  Parameters
  ----------
  pdb_code : str
      Four-character PDB identifier (e.g. ``"1ubq"``).
  cache_dir : Path, optional
      Directory to store the downloaded file.  Defaults to
      ``~/.cache/profam_bagel/pdb``.

  Returns
  -------
  Path
      Path to the downloaded CIF file.
  """
  import urllib.request

  cache = cache_dir or _PDB_CACHE_DIR
  cache.mkdir(parents=True, exist_ok=True)

  code = pdb_code.strip().lower()
  dest = cache / f"{code}.cif"
  if dest.is_file():
    print(f"  Using cached PDB structure: {dest}")
    return dest

  url = f"https://files.rcsb.org/download/{code}.cif"
  print(f"  Downloading PDB structure {code} from {url} ...")
  try:
    urllib.request.urlretrieve(url, str(dest))
  except Exception as exc:
    raise RuntimeError(
      f"Failed to download CIF for PDB code {pdb_code!r}: {exc}"
    ) from exc

  return dest


def extract_chain_from_cif(
  cif_path: Path,
  chain_id: str,
  output_path: Path | None = None,
) -> Path:
  """
  Read a CIF file and write a new CIF containing only the specified chain.

  Parameters
  ----------
  cif_path : Path
      Input CIF file.
  chain_id : str
      Chain identifier to extract (e.g. ``"A"``).
  output_path : Path, optional
      Where to write the filtered CIF.  Defaults to
      ``<cif_stem>_chain_<chain_id>.cif`` in the same directory.

  Returns
  -------
  Path
      Path to the output CIF file.
  """
  from biotite.structure.io.pdbx import set_structure  # type: ignore

  cif = CIFFile.read(str(cif_path))
  atoms = get_structure(cif, model=1)

  chain_atoms = atoms[atoms.chain_id == chain_id]
  if len(chain_atoms) == 0:
    available = sorted(set(atoms.chain_id))
    raise ValueError(
      f"Chain {chain_id!r} not found in {cif_path}. "
      f"Available chains: {available}"
    )

  if output_path is None:
    output_path = cif_path.parent / f"{cif_path.stem}_chain_{chain_id}.cif"

  out_cif = CIFFile()
  set_structure(out_cif, chain_atoms)
  out_cif.write(str(output_path))
  return output_path


def _load_structure_from_spec(kwargs: Dict[str, Any]) -> Any:
  """
  Load a structure ``AtomArray`` from kwargs, supporting both local file
  paths and PDB download.

  Consumes and removes the following keys from ``kwargs``:
  - ``template_structure_path`` **or** ``pdb_code``
  - ``template_chain_id``
  - ``template_residue_start`` / ``template_residue_end`` (metadata only)

  Returns ``(atoms, chain_id_was_applied)`` where ``chain_id_was_applied``
  is True when the atoms have already been filtered by chain.
  Returns ``(None, False)`` if neither key is present.
  """
  pdb_code = kwargs.pop("pdb_code", None)
  template_path_str = kwargs.pop("template_structure_path", None)
  chain_id = kwargs.pop("template_chain_id", None)
  kwargs.pop("template_residue_start", None)
  kwargs.pop("template_residue_end", None)

  if pdb_code is not None and template_path_str is not None:
    raise ValueError(
      "Provide either 'pdb_code' (to download from RCSB PDB) or "
      "'template_structure_path' (for a local CIF/PDB file), not both."
    )

  if pdb_code is not None:
    cif_path = download_pdb_cif(pdb_code)
    if chain_id is not None:
      cif_path = extract_chain_from_cif(cif_path, chain_id,
        output_path=_PDB_CACHE_DIR / f"{pdb_code.strip().lower()}_chain_{chain_id}.cif")
    cif = CIFFile.read(str(cif_path))
    atoms = get_structure(cif, model=1)
    return atoms, True  # chain already filtered

  if template_path_str is not None:
    template_path = Path(template_path_str)
    if not template_path.is_file():
      raise FileNotFoundError(
        f"Template structure file not found: {template_path}"
      )
    suffix = template_path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
      cif = CIFFile.read(str(template_path))
      atoms = get_structure(cif, model=1)
    elif suffix == ".pdb":
      pdb = PDBFile.read(str(template_path))
      atoms = pdb.get_structure(model=1)
    else:
      raise ValueError(
        f"Unsupported template structure format {suffix!r}; "
        "use .pdb, .cif or .mmcif."
      )
    if chain_id is not None:
      atoms = atoms[atoms.chain_id == str(chain_id)]
    return atoms, chain_id is not None

  return None, False


def _instantiate_folding_oracle(oracle_type: str, kwargs: Dict[str, Any], force_modal: bool = False) -> FoldingOracle:
  """Instantiate a single folding oracle from a type string and kwargs dict."""
  kwargs = dict(kwargs)  # don't mutate caller's dict
  if oracle_type == "ESMFold":
    if force_modal:
      kwargs["use_modal"] = True
    return ESMFold(**kwargs)
  elif oracle_type == "BatchedESMFold":
    if force_modal:
      kwargs["use_modal"] = True
    from pipeline.batched_esmfold import BatchedESMFold
    return BatchedESMFold(**kwargs)
  elif oracle_type == "AlphaFast":
    return AlphaFast(**kwargs)
  elif oracle_type == "Boltz":
    return Boltz(**kwargs)
  elif oracle_type == "Chai1":
    from bagel.oracles.folding import Chai1
    return Chai1(**kwargs)
  elif oracle_type == "AF2BindCraft":
    from bagel.oracles.folding import AF2BindCraft
    return AF2BindCraft(**kwargs)
  elif oracle_type == "ColabFold":
    from pipeline.colabfold_oracle import ColabFold
    return ColabFold(**kwargs)
  else:
    raise ValueError(
      f"Unsupported folding oracle type: {oracle_type!r}. "
      f"Use 'ESMFold', 'BatchedESMFold', 'AlphaFast', 'Boltz', 'Chai1', "
      f"'AF2BindCraft', or 'ColabFold'."
    )


def build_folding_oracles(
  energy_cfg: Dict[str, Any],
  force_modal: bool = False,
) -> Dict[str, FoldingOracle]:
  """Build a mapping of name → FoldingOracle from the energy config.

  Two YAML schemas are supported:

  **Single-oracle (legacy):**
  ::
      folding_oracle:
        type: ESMFold
        kwargs: {...}

  Returns ``{"default": <oracle>}``.

  **Multi-oracle (new):**
  ::
      folding_oracles:
        esmfold: {type: ESMFold, kwargs: {...}}
        boltz2:  {type: Boltz,   kwargs: {...}}

  Each energy entry can then reference one of these oracles by name via an
  ``oracle: <name>`` key.  Energy entries that omit the ``oracle`` key fall
  back to the first oracle in insertion order (treated as the "primary").
  """
  if "folding_oracles" in energy_cfg and energy_cfg["folding_oracles"]:
    oracles_cfg = energy_cfg["folding_oracles"]
    if not isinstance(oracles_cfg, dict):
      raise ValueError("energy_cfg.folding_oracles must be a dict of {name: {type, kwargs}}.")
    oracles: Dict[str, FoldingOracle] = {}
    for name, entry in oracles_cfg.items():
      if not isinstance(entry, dict):
        raise ValueError(f"folding_oracles[{name!r}] must be a dict with 'type' and 'kwargs'.")
      otype = entry.get("type")
      if not isinstance(otype, str):
        raise ValueError(f"folding_oracles[{name!r}].type must be a string, got {otype!r}.")
      okwargs = entry.get("kwargs", {}) or {}
      if not isinstance(okwargs, dict):
        raise ValueError(f"folding_oracles[{name!r}].kwargs must be a dict.")
      oracles[name] = _instantiate_folding_oracle(otype, okwargs, force_modal=force_modal)
    if not oracles:
      raise ValueError("folding_oracles must define at least one oracle.")
    return oracles

  # Legacy single-oracle path
  folding_cfg = energy_cfg.get("folding_oracle", {}) or {}
  oracle_type = folding_cfg.get("type", "ESMFold")
  kwargs = folding_cfg.get("kwargs", {}) or {}
  if not isinstance(kwargs, dict):
    raise ValueError("folding_oracle.kwargs must be a dictionary.")
  return {"default": _instantiate_folding_oracle(oracle_type, kwargs, force_modal=force_modal)}


def build_folding_oracle(energy_cfg: Dict[str, Any], force_modal: bool = False) -> FoldingOracle:
  """Legacy single-oracle helper. Returns the first oracle from
  :func:`build_folding_oracles` (for backwards compatibility with code paths
  that still expect a single oracle object)."""
  oracles = build_folding_oracles(energy_cfg, force_modal=force_modal)
  return next(iter(oracles.values()))


def parse_residue_range_string(spec: str) -> List[int]:
  """
  Parse a compact residue specification string into a sorted list of 0-based
  integer indices.

  Supported formats:
  - ``"5"``         → [5]
  - ``"1,2,5"``     → [1, 2, 5]
  - ``"1-5"``       → [1, 2, 3, 4, 5]
  - ``"1,2,5-10"``  → [1, 2, 5, 6, 7, 8, 9, 10]

  Whitespace around commas, dashes, and at the string edges is stripped.
  """
  indices: List[int] = []
  for part in spec.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part:
      tokens = part.split("-", 1)
      start = int(tokens[0].strip())
      end = int(tokens[1].strip())
      indices.extend(range(start, end + 1))
    else:
      indices.append(int(part))
  return sorted(set(indices))


def _normalise_residue_spec(spec: Any) -> Any:
  """
  Pre-process a residue specification before it is converted to
  ``bg.Residue`` objects.

  If ``spec`` is a **dict** (the standard format), each value is normalised
  independently — compact range strings are expanded to integer lists.

  If ``spec`` is a compact range string (e.g. ``"1,2,5-10"``), it is
  expanded to a flat list of integers.  All other types (int, list[int],
  ``"all"``, etc.) are returned unchanged.
  """
  if isinstance(spec, dict):
    result = {}
    for key, val in spec.items():
      if isinstance(val, str) and val.lower() != "all":
        result[key] = parse_residue_range_string(val)
      else:
        result[key] = val
    return result

  if isinstance(spec, str) and spec.lower() != "all":
    return parse_residue_range_string(spec)

  if isinstance(spec, list) and spec and isinstance(spec[0], str):
    return [parse_residue_range_string(s) for s in spec]

  return spec


def _convert_residue_spec_for_chain(
  spec: Any,
  chain: bg.Chain,
) -> Any:
  """
  Convert a JSON-friendly residue specification into one or more `bg.Residue`
  objects anchored to `chain`.

  Supported input formats:
  - int: single residue index on the chain
  - str: compact range string like ``"1,2,5-10"`` (see
    :func:`parse_residue_range_string`), or ``"all"``
  - list[int]: list of residue indices
  - list[str]: each element is a range string; produces nested groups
  - list[dict]: each dict must contain 'index' (0-based), optional 'chain_id'
  - nested lists (for multi-group energies like PAEEnergy / LISEnergy):
      [[0, 1], [10, 11]]  -> list[list[Residue]]
  """
  # Normalise compact range strings before further processing.
  spec = _normalise_residue_spec(spec)

  def to_residue(idx: int, chain_id: str | None = None) -> bg.Residue:
    if idx < 0 or idx >= chain.length:
      raise IndexError(
        f"Residue index {idx} is out of bounds for chain of length {chain.length}."
      )
    ref = chain.residues[idx]
    cid = chain_id or ref.chain_ID
    return bg.Residue(name=ref.name, chain_ID=cid, index=ref.index, mutable=ref.mutable)

  # Single integer
  if isinstance(spec, int):
    return [to_residue(spec)]

  # Nested lists (e.g. [[0,1],[10,11]])
  if isinstance(spec, list) and spec and isinstance(spec[0], list):
    return [_convert_residue_spec_for_chain(sub, chain) for sub in spec]

  # Flat list
  if isinstance(spec, list):
    residues: List[bg.Residue] = []
    for item in spec:
      if isinstance(item, int):
        residues.append(to_residue(item))
      elif isinstance(item, dict):
        idx = item.get("index")
        if idx is None:
          raise ValueError("Residue dict must contain an 'index' field.")
        cid = item.get("chain_id", chain.chain_ID)
        residues.append(to_residue(int(idx), chain_id=str(cid)))
      else:
        raise TypeError(
          f"Unsupported residue list element type {type(item)}: {item!r}"
        )
    return residues

  # Convenience: the string "all" means "all residues in the chain" in order.
  if isinstance(spec, str) and spec.lower() == "all":
    return [to_residue(i) for i in range(chain.length)]

  raise TypeError(f"Unsupported residue specification type: {type(spec)}")


def _collect_target_sequences(
  energy_cfg: Dict[str, Any],
) -> Dict[int, Tuple[str, str]]:
  """
  Scan the energy configuration for entries whose ``kwargs`` contain a
  ``"target"`` key (a reference amino-acid sequence string) or a
  ``"target_pdb_code"`` key (a PDB ID from which the sequence is extracted).

  Returns a mapping from energy-entry index to ``(target_sequence,
  target_chain_id)``.  The ``target_chain_id`` is derived from the
  ``residues`` dict (the non-``"GEN"`` key) for inline targets, or from
  ``target_chain_id`` in kwargs for PDB-downloaded targets.
  """
  targets: Dict[int, Tuple[str, str]] = {}
  energies_spec = energy_cfg.get("energies", [])
  for i, entry in enumerate(energies_spec):
    if not isinstance(entry, dict):
      continue
    kwargs = entry.get("kwargs", {}) or {}

    if "target" in kwargs:
      # For inline target, derive chain_ID from the residues dict.
      residues_spec = kwargs.get("residues", {})
      if not isinstance(residues_spec, dict):
        raise ValueError(
          f"Energy entry {i}: 'residues' must be a dict (with chain-name "
          f"keys) when 'target' is specified.  Got {type(residues_spec).__name__}."
        )
      non_gen_keys = [k for k in residues_spec if k != "GEN"]
      if len(non_gen_keys) != 1:
        raise ValueError(
          f"Energy entry {i}: residues dict must have exactly one "
          f"non-'GEN' key for the target chain.  "
          f"Found keys: {list(residues_spec.keys())}"
        )
      target_chain_id = non_gen_keys[0]
      targets[i] = (str(kwargs["target"]), target_chain_id)

    elif "target_pdb_code" in kwargs:
      # Download the structure and extract the chain sequence.
      pdb_code = str(kwargs["target_pdb_code"])
      target_chain_id = str(kwargs.get("target_chain_id", "A"))

      # Validate that the residues dict key matches target_chain_id.
      residues_spec = kwargs.get("residues", {})
      if isinstance(residues_spec, dict):
        non_gen_keys = [k for k in residues_spec if k != "GEN"]
        if non_gen_keys and non_gen_keys[0] != target_chain_id:
          raise ValueError(
            f"Energy entry {i}: residues dict key {non_gen_keys[0]!r} does "
            f"not match target_chain_id {target_chain_id!r}."
          )

      cif_path = download_pdb_cif(pdb_code)
      chain_cif = extract_chain_from_cif(
        cif_path, target_chain_id,
        output_path=_PDB_CACHE_DIR / f"{pdb_code.strip().lower()}_chain_{target_chain_id}.cif",
      )
      cif = CIFFile.read(str(chain_cif))
      atoms = get_structure(cif, model=1)
      seq = sequence_from_atomarray(atoms)
      targets[i] = (seq, target_chain_id)
      print(
        f"  Target for energy entry {i}: PDB {pdb_code} chain {target_chain_id} "
        f"({len(seq)} residues)"
      )

  # ------------------------------------------------------------------
  # Second pass: propagate discovered target sequences to entries that
  # reference the same chain_ID in their `residues` dict but do not
  # specify `target`/`target_pdb_code` themselves.  This lets
  # chain-free energies (PAEEnergy, SeparationEnergy, LISEnergy, …) be
  # used in multi-chain configs without repeating the target sequence.
  # ------------------------------------------------------------------
  if targets:
    target_by_chain_id: Dict[str, str] = {}
    for _, (seq, cid) in targets.items():
      target_by_chain_id.setdefault(cid, seq)

    for i, entry in enumerate(energies_spec):
      if not isinstance(entry, dict) or i in targets:
        continue
      kwargs = entry.get("kwargs", {}) or {}
      residues_spec = kwargs.get("residues", {})
      if not isinstance(residues_spec, dict):
        continue
      non_gen_keys = [k for k in residues_spec if k != "GEN"]
      if len(non_gen_keys) != 1:
        continue
      target_chain_id = non_gen_keys[0]
      if target_chain_id in target_by_chain_id:
        targets[i] = (target_by_chain_id[target_chain_id], target_chain_id)

  return targets


def _convert_residue_spec_for_chains(
  spec: Dict[str, Any],
  chains_by_id: Dict[str, "bg.Chain"],
) -> Any:
  """
  Convert a dict-format residue specification for a multi-chain system.

  ``spec`` is a dict like ``{"GEN": [0,1,2], "A": [5,6,7]}`` where each
  key is a chain identifier and each value is a residue specification for
  that chain.

  ``chains_by_id`` maps chain_ID → ``bg.Chain``.

  Returns a ``list[list[bg.Residue]]`` with ``"GEN"`` as group 0 and the
  remaining chain(s) as subsequent groups — the format expected by
  multi-group energy terms (PAEEnergy, SeparationEnergy, LISEnergy, etc.).
  """
  result = []
  # Process "GEN" first so it is always group 0.
  for chain_key in sorted(spec.keys(), key=lambda k: (k != "GEN", k)):
    if chain_key not in chains_by_id:
      raise ValueError(
        f"Residue spec references chain {chain_key!r} but available chains "
        f"are: {list(chains_by_id.keys())}"
      )
    result.append(
      _convert_residue_spec_for_chain(spec[chain_key], chains_by_id[chain_key])
    )
  return result


def build_energy_terms_for_chain(
  energy_cfg: Dict[str, Any],
  oracle: FoldingOracle | Dict[str, FoldingOracle],
  chain: bg.Chain,
  target_chains: Dict[int, "bg.Chain"] | None = None,
) -> List[bg.energies.EnergyTerm]:
  """
  Instantiate BAGEL EnergyTerm objects for a given chain, based on the
  energy YAML configuration.

  ``oracle`` may be either a single :class:`FoldingOracle` (legacy,
  single-oracle mode) or a ``dict[str, FoldingOracle]`` mapping oracle
  names to instances (multi-oracle mode). In multi-oracle mode, each energy
  entry may specify which oracle to use via the top-level ``oracle`` key
  (not inside ``kwargs``). Entries without an ``oracle`` key fall back to
  the first oracle in the dict.
  """
  energies_spec = energy_cfg.get("energies", [])
  if not isinstance(energies_spec, list):
    raise ValueError("energy_config must contain an 'energies' list.")

  # Normalise `oracle` arg to a dict; keep a "primary" reference for fallback.
  if isinstance(oracle, dict):
    oracles_by_name = oracle
    primary_oracle = next(iter(oracles_by_name.values()))
  else:
    oracles_by_name = {"default": oracle}
    primary_oracle = oracle

  terms: List[bg.energies.EnergyTerm] = []
  for entry_idx, entry in enumerate(energies_spec):
    if not isinstance(entry, dict):
      raise ValueError(f"Each energy entry must be a dict, got {entry!r}")
    etype = entry.get("type")
    if not isinstance(etype, str):
      raise ValueError(f"Energy 'type' must be a string, got {etype!r}")

    # Resolve which folding oracle instance this energy term should use.
    # Precedence: entry['oracle'] (name lookup) > primary oracle.
    oracle_name = entry.get("oracle")
    if oracle_name is not None:
      if oracle_name not in oracles_by_name:
        raise ValueError(
          f"Energy entry {entry_idx} ({etype}): oracle {oracle_name!r} not found. "
          f"Available oracles: {list(oracles_by_name.keys())}"
        )
      this_oracle = oracles_by_name[oracle_name]
    else:
      this_oracle = primary_oracle

    kwargs = dict(entry.get("kwargs", {}) or {})

    # Pop target-related keys — they are consumed by the pipeline to build
    # multi-chain systems and must not be forwarded to the BAGEL energy
    # constructor.
    kwargs.pop("target", None)
    kwargs.pop("target_pdb_code", None)
    kwargs.pop("target_chain_id", None)

    # Normalise compact residue range strings (e.g. "1-10,15") to flat
    # integer lists so that all downstream code sees the same format.
    # For dict-format specs, each value is normalised independently.
    if "residues" in kwargs:
      kwargs["residues"] = _normalise_residue_spec(kwargs["residues"])

    # Save the raw (integer) residue indices for the generated chain —
    # needed by TemplateMatchEnergy to select atoms from the template.
    raw_residue_indices = None
    if "residues" in kwargs:
      spec = kwargs["residues"]
      if isinstance(spec, dict):
        raw_residue_indices = spec.get("GEN", None)
      else:
        raw_residue_indices = spec

    # Convert residue specifications to `bg.Residue` objects.
    # Dict-format specs map chain-name keys to per-chain index lists.
    if "residues" in kwargs:
      spec = kwargs["residues"]
      if isinstance(spec, dict):
        non_gen_keys = [k for k in spec if k != "GEN"]
        if non_gen_keys and target_chains and entry_idx in target_chains:
          # Multi-chain: build a chain lookup and use the dict converter.
          tgt_chain = target_chains[entry_idx]
          chains_by_id: Dict[str, bg.Chain] = {
            "GEN": chain,
            tgt_chain.chain_ID: tgt_chain,
          }
          kwargs["residues"] = _convert_residue_spec_for_chains(
            spec, chains_by_id,
          )
        elif "GEN" in spec:
          # Single-chain: convert only the GEN portion.
          kwargs["residues"] = _convert_residue_spec_for_chain(
            spec["GEN"], chain,
          )
        else:
          raise ValueError(
            f"Energy entry {entry_idx}: residues dict must contain a 'GEN' key."
          )
      else:
        # Non-dict fallback (e.g. plain list or string).
        if target_chains and entry_idx in target_chains:
          raise ValueError(
            f"Energy entry {entry_idx}: 'residues' must be a dict (with "
            f"chain-name keys) when a target chain is present."
          )
        kwargs["residues"] = _convert_residue_spec_for_chain(spec, chain)

    # Special handling for TemplateMatchEnergy, which requires an AtomArray
    # `template_atoms` rather than a simple JSON-serialisable object.
    # Supports both local file paths and PDB code download.
    if etype == "TemplateMatchEnergy":
      atoms, _ = _load_structure_from_spec(kwargs)
      if atoms is None:
        raise ValueError(
          "TemplateMatchEnergy requires 'template_structure_path' or "
          "'pdb_code' in kwargs."
        )

      # Extract only the residues at the 0-based positions listed in
      # "residues".  The same positions are used by BAGEL to mask the
      # generated structure, so the atom counts will always match.
      # When "all" is specified, keep every residue in the template.
      if (
        raw_residue_indices is not None
        and not (isinstance(raw_residue_indices, str) and raw_residue_indices.lower() == "all")
        and len(raw_residue_indices) > 0
      ):
        # Flatten nested lists for TemplateMatchEnergy (always single-group).
        flat_indices = raw_residue_indices
        if isinstance(flat_indices, list) and flat_indices and isinstance(flat_indices[0], list):
          flat_indices = flat_indices[0]
        ca_mask = atoms.atom_name == "CA"
        template_res_ids = atoms[ca_mask].res_id
        max_idx = max(int(i) for i in flat_indices)
        if max_idx >= len(template_res_ids):
          raise IndexError(
            f"Residue index {max_idx} is out of bounds for template chain "
            f"with {len(template_res_ids)} residues."
          )
        selected_res_ids = [int(template_res_ids[int(i)]) for i in flat_indices]
        atoms = atoms[np.isin(atoms.res_id, selected_res_ids)]

      kwargs["template_atoms"] = atoms

    # Instantiate the BAGEL energy term dynamically
    try:
      energy_cls = getattr(bg.energies, etype)
    except AttributeError as e:
      raise ValueError(
        f"Unknown BAGEL energy type {etype!r}. "
        "Ensure it matches a class name in bagel.energies."
      ) from e

    term = energy_cls(oracle=this_oracle, **kwargs)
    terms.append(term)

  return terms


def evaluate_sequences_with_bagel(
  sequences: Sequence[str],
  energy_cfg: Dict[str, Any],
  folding_oracle: FoldingOracle | Dict[str, FoldingOracle],
  cycle_index: int,
  cycle_dir: Path,
  enforce_template: bool = True,
  save_structures: bool = False,
) -> Tuple[List[float], List[Dict[str, Any]], List[Any]]:
  """
  For each sequence, build a single-chain BAGEL System, run the required
  oracles (folding, embedding, …), compute total weighted energy from
  configured energy terms, and — when a folding oracle was invoked — save
  the predicted structures for later export.

  ``folding_oracle`` may be either a single :class:`FoldingOracle` (legacy)
  or a ``dict[str, FoldingOracle]`` mapping oracle names to instances.  In
  multi-oracle mode, each energy entry is bound to a specific oracle (see
  :func:`build_energy_terms_for_chain`), and each sequence is folded by
  every oracle that any active energy term references.  Results from all
  oracles are merged into a single ``OraclesResultDict`` for energy
  computation.

  Folding oracles are only invoked when at least one energy term requires
  one; otherwise no structure prediction is performed and no CIF files are
  written.

  Ensembling for Boltz2 is now handled **inside the Boltz oracle itself**
  via the ``diffusion_samples`` kwarg (set on the oracle in the energy YAML).
  The oracle runs ``boltz predict --diffusion_samples N`` in a single
  subprocess call, parses all N outputs, and returns a single
  ``BoltzResult`` whose scalar and tensor metrics are averaged across
  samples.  For other oracles (Chai-1, AF2), use their own native
  ensembling kwargs (e.g. ``num_diffn_samples`` for Chai-1).

  Returns:
    - energies: list of total energies, one per input sequence
    - details: list of dicts including per-sequence energy breakdown
    - folding_results: list of FoldingResult objects from the primary
      oracle (entries are None when no folding was needed)
  """
  from bagel.oracles import OraclesResultDict  # type: ignore
  from bagel.oracles.folding import FoldingOracle  # type: ignore

  # Normalise oracle argument to a dict and pick a primary (first entry).
  if isinstance(folding_oracle, dict):
    oracles_by_name: Dict[str, FoldingOracle] = folding_oracle
  else:
    oracles_by_name = {"default": folding_oracle}
  primary_oracle = next(iter(oracles_by_name.values()))

  # Pre-scan energy config for entries that require a target chain.
  target_seqs = _collect_target_sequences(energy_cfg)

  # ------------------------------------------------------------------
  # Phase 1: Build chains and energy terms for every sequence.
  # ------------------------------------------------------------------
  per_seq_data: List[Dict[str, Any]] = []
  for idx, seq in enumerate(sequences):
    residues = [
      bg.Residue(name=aa, chain_ID="GEN", index=i, mutable=False)
      for i, aa in enumerate(seq)
    ]
    chain = bg.Chain(residues=residues)

    # Build target chains for energy entries that have a "target" key.
    target_chains_map: Dict[int, bg.Chain] = {}
    seen_targets: Dict[Tuple[str, str], bg.Chain] = {}
    for entry_idx, (tgt_seq, tgt_chain_id) in target_seqs.items():
      dedup_key = (tgt_seq, tgt_chain_id)
      if dedup_key in seen_targets:
        target_chains_map[entry_idx] = seen_targets[dedup_key]
      else:
        tgt_residues = [
          bg.Residue(name=aa, chain_ID=tgt_chain_id, index=i, mutable=False)
          for i, aa in enumerate(tgt_seq)
        ]
        tgt_chain = bg.Chain(residues=tgt_residues)
        target_chains_map[entry_idx] = tgt_chain
        seen_targets[dedup_key] = tgt_chain

    energy_terms = build_energy_terms_for_chain(
      energy_cfg, folding_oracle, chain,
      target_chains=target_chains_map if target_chains_map else None,
    )

    all_chains = [chain] + list({id(c): c for c in target_chains_map.values()}.values())
    oracles_needed = list(set(term.oracle for term in energy_terms))

    per_seq_data.append({
      "chain": chain,
      "all_chains": all_chains,
      "energy_terms": energy_terms,
      "oracles_needed": oracles_needed,
    })

  # ------------------------------------------------------------------
  # Phase 2: Fold each sequence with every folding oracle it needs.
  #
  # Ensembling (e.g. Boltz --diffusion_samples > 1) is handled inside the
  # oracle itself: each oracle's .fold() method returns a single result
  # (averaged over internal samples if applicable).
  # ------------------------------------------------------------------

  # Check which sequences need a folding oracle.
  needs_folding = [
    any(isinstance(o, FoldingOracle) for o in d["oracles_needed"])
    for d in per_seq_data
  ]

  # Collect the set of folding oracles that are actually needed by any term,
  # preserving the oracles_by_name insertion order.
  used_oracles: Dict[str, FoldingOracle] = {}
  for d in per_seq_data:
    for o in d["oracles_needed"]:
      if isinstance(o, FoldingOracle):
        for name, inst in oracles_by_name.items():
          if inst is o and name not in used_oracles:
            used_oracles[name] = inst
  # Fallback: if energies don't reference the oracles by identity, use all.
  if any(needs_folding) and not used_oracles:
    used_oracles = dict(oracles_by_name)
  primary_name = next(iter(used_oracles), None) if used_oracles else None

  # Per-sequence, per-oracle folding results.  Each oracle contributes
  # exactly one FoldingResult per sequence (any internal ensembling is
  # absorbed inside the oracle and returned as a single averaged result).
  #
  #   batch_folding_results[i] = {oracle_name: FoldingResult} | None
  #
  # None means folding failed for that sequence on at least one oracle.
  batch_folding_results: List[Any] = [None] * len(sequences)

  if any(needs_folding) and primary_name is not None:
    batch_indices = [i for i, nf in enumerate(needs_folding) if nf]
    batch_chains = [per_seq_data[i]["all_chains"] for i in batch_indices]

    if len(used_oracles) > 1:
      print(f"  Multi-oracle evaluation: {list(used_oracles.keys())}")

    def _clear_gpu_mem():
      """Free any cached GPU memory between oracle calls to avoid OOM when
      stacking multiple large models (e.g. Chai-1 after Boltz in the same
      process)."""
      import gc as _gc
      try:
        import torch as _torch
        if _torch.cuda.is_available():
          _torch.cuda.empty_cache()
      except Exception:
        pass
      _gc.collect()

    for i, chains in zip(batch_indices, batch_chains):
      per_oracle_results: Dict[str, Any] = {}
      fold_failed = False

      for name, oracle_inst in used_oracles.items():
        _clear_gpu_mem()
        try:
          per_oracle_results[name] = oracle_inst.predict(chains=chains)
        except Exception as exc:
          print(f"  Sequence {i}: oracle {name!r} folding failed ({exc})")
          fold_failed = True
          break

      batch_folding_results[i] = None if fold_failed else per_oracle_results

  # ------------------------------------------------------------------
  # Phase 3: Compute energies using the (pre-computed) oracle results.
  # Each oracle contributed exactly one FoldingResult per sequence; any
  # internal ensembling (e.g. Boltz diffusion_samples) is already absorbed
  # into a single averaged result by the oracle itself.
  # ------------------------------------------------------------------
  energies: List[float] = []
  details: List[Dict[str, Any]] = []
  folding_results: List[Any] = []

  for idx, seq in enumerate(sequences):
    d = per_seq_data[idx]
    energy_terms = d["energy_terms"]

    per_oracle_results = batch_folding_results[idx]
    if per_oracle_results is None and needs_folding[idx]:
      print(f"  Sequence {idx}: folding failed, marking as folding_failed")
      energies.append(float("inf"))
      folding_results.append(None)
      details.append({"energy": float("inf"), "error": "folding_failed"})
      continue

    if per_oracle_results is None:
      # No folding needed for this sequence.
      per_oracle_results = {}

    # Build OraclesResultDict: one folding result per oracle, plus any
    # non-folding oracles called inline.
    oracles_result = OraclesResultDict()
    non_folding_failed = False

    for oracle in d["oracles_needed"]:
      if isinstance(oracle, FoldingOracle):
        matched_name = None
        for name, inst in used_oracles.items():
          if inst is oracle:
            matched_name = name
            break
        if matched_name is None or matched_name not in per_oracle_results:
          print(f"  Sequence {idx}: no result for oracle {type(oracle).__name__}")
          non_folding_failed = True
          break
        oracles_result[oracle] = per_oracle_results[matched_name]
      else:
        try:
          oracles_result[oracle] = oracle.predict(chains=d["all_chains"])
        except Exception as exc:
          print(f"  Sequence {idx}: non-folding oracle {type(oracle).__name__} failed: {exc}")
          non_folding_failed = True
          break

    if non_folding_failed:
      energies.append(float("inf"))
      folding_results.append(None)
      details.append({"energy": float("inf"), "error": "oracle_failed"})
      continue

    # Compute weighted energy across all energy terms.
    total_energy = 0.0
    per_term: Dict[str, float] = {}
    term_failed = False

    for term in energy_terms:
      try:
        unweighted, weighted = term.compute(oracles_result=oracles_result)
        per_term[term.name] = float(unweighted)
        total_energy += float(weighted)
      except Exception as exc:
        if not enforce_template:
          print(
            f"  Sequence {idx}: term {term.name} ({type(term).__name__}) "
            f"raised {type(exc).__name__}: {exc}"
          )
          per_term[term.name] = float("inf")
          total_energy = float("inf")
          term_failed = True
          break
        else:
          raise

    if term_failed or total_energy >= float("inf"):
      energies.append(float("inf"))
      folding_results.append(None)
      details.append({"energy": float("inf"), "error": "energy_term_failed"})
      continue

    # Pick the primary oracle's structure for saving / visualisation.
    primary_fr = None
    if primary_name is not None and primary_name in per_oracle_results:
      primary_fr = per_oracle_results[primary_name]

    energies.append(total_energy)
    folding_results.append(primary_fr)
    details.append({
      "index": idx,
      "sequence": seq,
      "energy": total_energy,
      "energy_terms": per_term,
    })

  # Save structures for sequences where the folding oracle was called.
  if save_structures and any(fr is not None for fr in folding_results):
    structures_dir = cycle_dir / f"sequences_cycle_all_{cycle_index}"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for idx, fr in enumerate(folding_results):
      if fr is not None:
        cif_path = structures_dir / f"sequence_{idx:04d}.cif"
        fr.to_cif(cif_path)
        if hasattr(fr, "save_attributes"):
          fr.save_attributes(structures_dir / f"sequence_{idx:04d}")

  return energies, details, folding_results



# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------


def _save_and_log_best_structure(
  folding_result: Any,
  output_dir: Path,
  wandb_run: Any,
  cycle: int,
  energy: float,
  sequence: str,
  name: str,
) -> None:
  """Persist the campaign-best predicted structure to disk and log it to wandb.

  Called from the elitism/new-best check.  Overwrites
  ``<output_dir>/best_structure.cif`` in place so only one CIF is ever kept
  on disk, no matter how many cycles run.  When wandb is active, logs the
  same CIF as ``best/structure`` via :class:`wandb.Molecule`, which renders
  as an interactive 3D viewer on the run page.  The logging cadence is
  bounded by how often a new best is accepted (so wandb isn't spammed).

  Failures are non-fatal — a warning is printed and the pipeline continues.
  """
  if folding_result is None or not hasattr(folding_result, "to_cif"):
    return
  best_cif_path = output_dir / "best_structure.cif"
  try:
    folding_result.to_cif(best_cif_path)
  except Exception as exc:
    print(f"  Warning: failed to save best structure: {exc}")
    return
  if wandb_run is None:
    return
  try:
    import wandb
    wandb.log({
      "best/structure": wandb.Molecule(str(best_cif_path)),
      "best/energy": float(energy),
      "best/cycle": int(cycle),
      "best/sequence": sequence,
      "best/name": name,
    }, step=cycle)
  except Exception as exc:
    print(f"  Warning: failed to log best structure to wandb: {exc}")


def _collect_checkpoint_results(cfg: PipelineConfig) -> Dict[str, Any]:
  """
  Read the current cycle_stats.json and energy_summary.png from the output
  directory and return them as a dict suitable for serialization over the wire.
  """
  results: Dict[str, Any] = {"cycle_stats": None, "energy_plot_png": None}

  stats_path = cfg.output_dir / "cycle_stats.json"
  if stats_path.is_file():
    with stats_path.open("r") as f:
      results["cycle_stats"] = json.load(f)

  plot_path = cfg.output_dir / "energy_summary.png"
  if plot_path.is_file():
    results["energy_plot_png"] = plot_path.read_bytes()

  return results


def adjust_temperature(
    raw_unique_fraction: float,
    raw_novel_fraction: float,
    avg_sim_to_prompt: float,
    current_temperature: float,
    starting_temperature: float,
) -> Tuple[float, str | None]:
  """Adaptive temperature heuristic.

  Raises temperature when diversity is low (within batch, across campaign,
  or prompt collapse). Lowers toward starting temp when diversity is healthy.

  Returns (new_temperature, reason_string_or_None).
  """
  low_batch_diversity = raw_unique_fraction < 0.5
  prompt_collapse = avg_sim_to_prompt >= 1.0 - 1e-5
  low_novelty = raw_novel_fraction < 1.0 / 3.0

  if low_batch_diversity or prompt_collapse or low_novelty:
    reasons = []
    if low_batch_diversity:
      reasons.append(f"low batch diversity ({raw_unique_fraction:.2f})")
    if prompt_collapse:
      reasons.append(f"prompt collapse (sim={avg_sim_to_prompt:.3f})")
    if low_novelty:
      reasons.append(f"low novelty ({raw_novel_fraction:.2f})")
    return current_temperature + 0.1, ", ".join(reasons)

  if raw_unique_fraction == 1.0 and raw_novel_fraction > 0.8 and current_temperature > starting_temperature:
    reduction = min(current_temperature - starting_temperature, 0.05)
    return current_temperature - reduction, None

  return current_temperature, None


def run_pipeline(
  cfg: PipelineConfig,
  force_modal_folding: bool = False,
  checkpoint_callback: Any = None,
  shared_model: Any = None,
  shared_device: str | None = None,
) -> Dict[str, Any]:
  cfg.output_dir.mkdir(parents=True, exist_ok=True)

  # Save all pipeline settings to the output folder for reproducibility.
  config_snapshot = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
  config_path = cfg.output_dir / "pipeline_config.json"
  with config_path.open("w") as f:
    json.dump(config_snapshot, f, indent=2)
  print(f"Config saved to {config_path}")

  # Initialize Weights & Biases logging (optional).
  wandb_run = None
  if cfg.wandb_enabled:
    try:
      import wandb
      run_name = cfg.wandb_run_name or cfg.output_dir.name
      wandb_run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        tags=cfg.wandb_tags,
        config=config_snapshot,
        dir=str(cfg.output_dir),
        reinit=True,
      )
      print(f"W&B run: {wandb_run.url}")
    except Exception as e:
      print(f"Warning: wandb init failed ({e}), continuing without wandb")
      wandb_run = None

  if cfg.freeze_prompt and not cfg.reinject_initial:
    print("Warning: freeze_prompt requires reinject_initial=True. Setting it.")
    cfg.reinject_initial = True

  # Load ProFam model once and reuse across all cycles (skip for non-ProFam proposals).
  # When the proposal bandit is enabled, always load ProFam since it may be selected.
  # If shared_model is provided (e.g. from HP search), reuse it instead of loading.
  profam_model, profam_device = (None, "cpu")
  if shared_model is not None:
    profam_model = shared_model
    profam_device = shared_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using shared ProFam model (device={profam_device})")
  elif cfg.proposal_method == "profam" or cfg.thompson_proposal_bandit:
    profam_model, profam_device = load_profam_model(cfg)
    if cfg.thompson_proposal_bandit:
      print("Proposal bandit enabled: ProFam model loaded (may also use random_mutation)")
  else:
    print(f"Proposal method: {cfg.proposal_method} (max_mutations={cfg.max_mutations})")

  # Load energy configuration & instantiate BAGEL folding oracle(s).
  # Multi-oracle mode activates when energy_cfg uses `folding_oracles:` (plural).
  energy_cfg = load_energy_config(cfg.energy_config)
  folding_oracles = build_folding_oracles(energy_cfg, force_modal=force_modal_folding)
  folding_oracle = next(iter(folding_oracles.values()))  # legacy single-oracle handle
  if len(folding_oracles) > 1:
    print(f"Multi-oracle mode: {list(folding_oracles.keys())}")

  # GRPO online RL fine-tuning (optional)
  grpo_step = None
  if cfg.grpo_enabled and profam_model is not None:
    from pipeline.grpo import GRPOConfig, PipelineGRPOStep
    grpo_config = GRPOConfig(
      enabled=True,
      grpo_beta=cfg.grpo_beta,
      grpo_clip_ratio=cfg.grpo_clip_ratio,
      grpo_lr=cfg.grpo_lr,
      grpo_weight_decay=cfg.grpo_weight_decay,
      grpo_temperature=cfg.grpo_temperature,
      grpo_top_p=cfg.grpo_top_p,
      grpo_max_tokens=cfg.grpo_max_tokens,
      grpo_normalize_rewards=cfg.grpo_normalize_rewards,
      grpo_reward_baseline=cfg.grpo_reward_baseline,
      grpo_use_reference_model=cfg.grpo_use_reference_model,
      rl_every_n_cycles=cfg.rl_every_n_cycles,
      rl_steps_per_cycle=cfg.rl_steps_per_cycle,
    )
    grpo_step = PipelineGRPOStep(
      model=profam_model,
      config=grpo_config,
      device=profam_device,
    )
    # Encoder-decoder split: frozen encoder computes prompt KV cache,
    # trainable decoder processes completions and receives GRPO updates.
    profam_model.init_encoder_decoder_grpo()
    print(f"GRPO enabled: lr={cfg.grpo_lr}, "
          f"every {cfg.rl_every_n_cycles} cycles, {cfg.rl_steps_per_cycle} steps/cycle")
    print("  Encoder-decoder mode: frozen encoder for prompt, trainable decoder for completions")

  # ── Bradley-Terry ranking loss setup ────────────────────────────────
  bt_optimizer = None
  bt_pool: List[Tuple[float, str]] = []  # (energy, sequence) — maintained sorted by energy
  if cfg.bt_enabled and profam_model is not None:
    # Reuse the encoder-decoder split (init if not already done by GRPO)
    if profam_model._encoder_model is None:
      profam_model.init_encoder_decoder_grpo()
    bt_optimizer = torch.optim.Adam(
      [p for p in profam_model.parameters() if p.requires_grad],
      lr=cfg.bt_lr,
    )
    print(f"Bradley-Terry enabled: lr={cfg.bt_lr}, pool_size={cfg.bt_pool_size}, "
          f"batch_size={cfg.bt_batch_size}, every {cfg.bt_every_n_cycles} cycles")
    print("  Encoder-decoder mode: frozen encoder for prompt, trainable decoder for completions")

  # ── Likelihood tracking: best/worst sequences ──────────────────────
  # Each entry: (energy, sequence, prompt_input_ids)
  # Sorted so best[0] has the lowest (best) energy, worst[0] the highest.
  likelihood_best: List[Tuple[float, str, torch.Tensor]] = []
  likelihood_worst: List[Tuple[float, str, torch.Tensor]] = []

  # Thompson reward signal is always the composite weighted total energy
  # (lower is better).  No per-term validation is required.

  rng = np.random.default_rng(cfg.random_seed)
  cycle_log_path = cfg.output_dir / "cycle_stats.json"

  # Read initial sequences S1 from FASTA, or generate randomly.
  if cfg.random_init:
    # Exclude cysteine — disulfide bonds cause ESMFold failures on random sequences
    _AMINO_ACIDS_NO_CYS = "ADEFGHIKLMNPQRSTVWY"
    rand_len = rng.integers(cfg.random_init_max_residues // 2, cfg.random_init_max_residues + 1)
    rand_seq = "".join(rng.choice(list(_AMINO_ACIDS_NO_CYS), size=rand_len))
    init_names = [f"random_init_len{rand_len}"]
    init_seqs = [rand_seq]
    print(f"Random init: generated sequence of length {rand_len}")
  else:
    init_names, init_seqs = read_fasta(
      str(cfg.initial_fasta),
      keep_insertions=True,
      keep_gaps=False,
      to_upper=True,
    )
  base_initial_names = list(init_names)
  base_initial_seqs = list(init_seqs)

  injected_names: List[str] = []
  injected_seqs: List[str] = []

  # Global unique ID counter and memory buffer for n_memory support.
  # Each generated sequence receives a monotonically increasing ID across
  # all cycles (e.g. cycle 1 → IDs 0-9, cycle 2 → IDs 10-19, etc.).
  next_global_id = 0
  # Memory buffer: list of (ids, names, seqs, energies) tuples, one per past cycle.
  # At most cfg.n_memory entries are kept.
  memory_buffer: List[tuple] = []

  temp_bandit: TemperatureBandit | None = None

  # GRPO replay buffer: cache the last N cycles' token data + rewards
  # for larger effective group sizes.  Each entry is a dict with keys:
  #   generated_tokens, old_per_token_lps, old_per_token_mask, rewards
  # The input_ids (prompt) always comes from the current cycle.
  grpo_replay_buffer: List[Dict[str, torch.Tensor]] = []

  # Adaptive temperature: raise on low diversity, pull back toward starting temp.
  starting_temperature = cfg.profam_temperature if cfg.profam_temperature is not None else 0.7
  current_temperature = starting_temperature

  # Anti-regression state: elitism + conditional swap.
  elite_seq: str | None = None
  elite_name: str | None = None
  elite_energy: float = float("inf")
  elite_cycle: int = -1
  prev_injection_best_energy: float = float("inf")
  annealing_temp: float | None = cfg.annealing_initial_temp

  # Thompson sampling state.
  thompson_sampler: ThompsonSampler | None = None
  temp_bandit: TemperatureBandit | None = None
  proposal_bandit: ProposalBandit | None = None
  if cfg.selection_strategy == "thompson" or cfg.thompson_proposal_bandit:
    thompson_sampler = ThompsonSampler(
      m_samples=cfg.thompson_m_samples,
      exploit_bias=cfg.thompson_exploit_bias,
      rng=rng,
    )
    if cfg.thompson_temperature_bins:
      temp_bandit = TemperatureBandit(
        bins=cfg.thompson_temperature_bins,
        exploit_bias=cfg.thompson_exploit_bias,
        rng=rng,
      )
      print(f"  Temperature bandit: bins={temp_bandit.bins}")
  # Proposal bandit can be used with any selection_strategy (greedy or thompson).
  # With greedy: elitist prompt selection + bandit chooses proposal method.
  if cfg.thompson_proposal_bandit:
    proposal_bandit = ProposalBandit(
      prior_alpha=cfg.proposal_bandit_prior_alpha,
      prior_beta=cfg.proposal_bandit_prior_beta,
      rng=rng,
    )
    print(f"  Proposal bandit: methods={ProposalBandit.METHODS}, "
          f"prior=Beta({cfg.proposal_bandit_prior_alpha}, {cfg.proposal_bandit_prior_beta})")

  # SelectionManager: unified orchestrator for Thompson sampling / greedy selection.
  # Used when selection_strategy="thompson" OR when proposal_bandit is enabled
  # (to ensure progeny are registered as new arms even in the greedy+bandit case).
  selection_manager: SelectionManager | None = None
  if thompson_sampler is not None:
    # Choose prompt selector based on selection_strategy
    if cfg.selection_strategy == "thompson":
      prompt_selector = ThompsonPromptSelector()
    else:
      prompt_selector = GreedyPromptSelector()
    selection_manager = SelectionManager(
      thompson_sampler=thompson_sampler,
      prompt_selector=prompt_selector,
      max_arms=cfg.thompson_max_arms,
      max_identity=cfg.thompson_max_identity,
    )
    print(f"  SelectionManager: selector={prompt_selector.method_name}, "
          f"reward=total_energy, max_arms={cfg.thompson_max_arms}")

  # Sequence deduplication cache: maps sequence string → (energy, details_dict).
  # Populated during evaluation; checked before folding to skip duplicates.
  seen_sequences: Dict[str, tuple] = {}

  # Evaluate initial seed sequence(s) to establish a baseline energy.
  # This ensures cycle 1 must improve over the seed to be accepted.
  if cfg.elitism or cfg.accept_only_improvement or cfg.selection_strategy == "thompson":
    _AMINO_ACIDS_NO_CYS = "ADEFGHIKLMNPQRSTVWY"
    max_seed_retries = 10 if cfg.random_init else 1
    for _seed_attempt in range(max_seed_retries):
      print(f"=== Evaluating initial seed sequence(s) (attempt {_seed_attempt + 1}) ===")
      seed_energies, seed_details, seed_folding_results = evaluate_sequences_with_bagel(
        sequences=base_initial_seqs,
        energy_cfg=energy_cfg,
        folding_oracle=folding_oracles,
        cycle_index=0,
        cycle_dir=cfg.output_dir / "cycle_000_seed",
        enforce_template=cfg.enforce_template,
        save_structures=cfg.save_structures,
      )
      seed_best_idx = int(np.argmin(seed_energies))
      seed_best_energy = float(seed_energies[seed_best_idx])
      if seed_best_energy < float("inf") or not cfg.random_init:
        break
      # Folding failed on random init — regenerate and retry
      rand_len = rng.integers(cfg.random_init_max_residues // 2, cfg.random_init_max_residues + 1)
      rand_seq = "".join(rng.choice(list(_AMINO_ACIDS_NO_CYS), size=rand_len))
      base_initial_seqs = [rand_seq]
      base_initial_names = [f"random_init_len{rand_len}_retry{_seed_attempt + 1}"]
      init_seqs = list(base_initial_seqs)
      print(f"  Seed folding failed, regenerating random sequence (len={rand_len})")
    print(f"  Seed sequence best energy: {seed_best_energy:.4f}")
    if seed_details[seed_best_idx] and isinstance(seed_details[seed_best_idx], dict):
      terms = seed_details[seed_best_idx].get("energy_terms", {})
      print(f"  Seed energy terms: {terms}")
    # Log seed baseline as cycle 0 in cycle_stats.json.
    seed_entry = {
      "cycle": 0,
      "num_generated": len(base_initial_seqs),
      "all_avg_energy": float(np.mean(seed_energies)),
      "all_min_energy": seed_best_energy,
      "best_sequence": {
        "sequence": base_initial_seqs[seed_best_idx],
        "energy": seed_best_energy,
        "energy_terms": (
          seed_details[seed_best_idx].get("energy_terms", {})
          if seed_details[seed_best_idx] and isinstance(seed_details[seed_best_idx], dict)
          else {}
        ),
      },
      "swap_accepted": None,
      "swap_reason": "seed_baseline",
    }
    log_path = cfg.output_dir / "cycle_stats.json"
    if log_path.is_file():
      with log_path.open("r") as f:
        log_data = json.load(f)
    else:
      log_data = {}
    log_data["0"] = seed_entry
    with log_path.open("w") as f:
      json.dump(log_data, f, indent=2)

    prev_injection_best_energy = seed_best_energy
    if seed_best_energy < elite_energy:
      elite_energy = seed_best_energy
      elite_seq = base_initial_seqs[seed_best_idx]
      elite_name = base_initial_names[seed_best_idx]
      elite_cycle = 0
      print(f"  Initial elite set: energy={elite_energy:.4f}")
      _save_and_log_best_structure(
        folding_result=(
          seed_folding_results[seed_best_idx]
          if seed_best_idx < len(seed_folding_results) else None
        ),
        output_dir=cfg.output_dir,
        wandb_run=wandb_run,
        cycle=0,
        energy=elite_energy,
        sequence=elite_seq,
        name=elite_name,
      )

    # Cache seed sequences for deduplication.
    if cfg.deduplicate_sequences:
      for i, seq in enumerate(base_initial_seqs):
        if seq not in seen_sequences:
          seen_sequences[seq] = (float(seed_energies[i]), seed_details[i])
      print(f"  Dedup cache: {len(seen_sequences)} seed sequences cached")

    # Log seed sequences as cycle 0 in the CSV.
    append_cycle_csv(
      csv_path=cfg.output_dir / "all_sequences.csv",
      cycle_index=0,
      names=base_initial_names,
      sequences=base_initial_seqs,
      energies=seed_energies,
      details=seed_details,
      folding_results=seed_folding_results,
      initial_seqs=base_initial_seqs,
      prompt_seqs=None,
      proposal_method="seed",
    )

    # Register seed sequences as initial Thompson arms.  Reward signal is the
    # composite weighted total energy (lower = better).
    if thompson_sampler is not None:
      print(f"  Thompson SEED REGISTRATION (m_samples={thompson_sampler.m_samples}):")
      seed_reward_values = [
        float(d.get("energy", float("inf"))) if isinstance(d, dict) else float("inf")
        for d in seed_details
      ]
      n_seed_registered = 0
      for i, (name, seq) in enumerate(zip(base_initial_names, base_initial_seqs)):
        energy_val = seed_reward_values[i]
        if math.isfinite(energy_val):
          arm = thompson_sampler.add_arm(
            sequence=seq, name=name, ipsae_raw=energy_val,
            parent_arm_id=None, cycle=0,
          )
          n_seed_registered += 1
          print(f"    arm {arm.arm_id}: {name}, "
                f"energy={energy_val:.4f}, "
                f"reward={arm.reward:.4f}, "
                f"α={arm.alpha:.4f}, β={arm.beta_param:.4f}, "
                f"seq_len={len(seq)}")
        else:
          print(f"    SKIPPED {name}: energy=inf (folding failure)")
      print(f"  Thompson: {n_seed_registered}/{len(base_initial_seqs)} seeds registered as arms")

      # Prune to top-K diverse arms if configured.
      if cfg.thompson_max_arms > 0:
        prune_stats = thompson_sampler.prune_to_top_k_diverse(
          k=cfg.thompson_max_arms,
          max_identity=cfg.thompson_max_identity,
        )
        if prune_stats["arms_before"] > prune_stats["arms_after"]:
          print(f"  Thompson PRUNING: {prune_stats['arms_before']} → "
                f"{prune_stats['arms_after']} arms "
                f"(max_identity={cfg.thompson_max_identity:.2f})")
          print(f"    retained: {prune_stats['retained_ids']}")
          print(f"    pruned: {prune_stats['pruned_ids']}")

      # Set the best seed arm as the initial parent — cycle 1's progeny
      # are conditioned on the seeds, so the best seed should get credit.
      if thompson_sampler.arms:
        best_seed_arm = min(
          thompson_sampler.arms.values(), key=lambda a: a.ipsae_raw,
        )
        thompson_sampler._last_selected_arm_id = best_seed_arm.arm_id  # type: ignore[attr-defined]
        print(f"  Thompson: initial parent set to arm {best_seed_arm.arm_id} "
              f"({best_seed_arm.name}, energy={best_seed_arm.ipsae_raw:.4f})")
      # Save initial arms state.
      arms_path = cfg.output_dir / "thompson_arms.json"
      with arms_path.open("w") as f:
        json.dump(thompson_sampler.get_state_dict(), f, indent=2)

  for cycle in range(1, cfg.max_cycles + 1):
    print(f"=== Starting cycle {cycle} / {cfg.max_cycles} ===")
    cycle_dir = cfg.output_dir / f"cycle_{cycle:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Build ProFam input FASTA for this cycle.
    #
    # When reinject_initial is True (default) the original initial sequences
    # are always included alongside any injected sequences from the previous
    # cycle.  When reinject_initial is False the initial sequences are only
    # used for the very first cycle (where nothing has been generated yet);
    # from cycle 2 onwards only the selected subset is fed back.
    profam_input_fasta = cycle_dir / "profam_input.fasta"

    include_initial = cfg.reinject_initial or not injected_seqs
    if include_initial:
      all_names = base_initial_names + injected_names
      all_seqs = base_initial_seqs + injected_seqs
    else:
      all_names = list(injected_names)
      all_seqs = list(injected_seqs)

    # Guardrail: ensure the total prompt length stays within the ProFam
    # max_tokens budget by trimming injected sequences if necessary.
    #
    # This is an approximation: we treat each amino acid as one token and
    # ignore special tokens/overheads, which makes the check conservative
    # for typical use cases.
    total_prompt_len = sum(len(s) for s in all_seqs)
    if total_prompt_len > cfg.profam_max_tokens:
      if include_initial:
        base_len = sum(len(s) for s in base_initial_seqs)
        remaining_budget = cfg.profam_max_tokens - base_len
      else:
        remaining_budget = cfg.profam_max_tokens

      if remaining_budget <= 0 and include_initial:
        # Fall back to only the initial sequences if even they exceed
        # the token budget when combined.
        injected_names = []
        injected_seqs = []
        all_names = base_initial_names
        all_seqs = base_initial_seqs
      else:
        trimmed_injected_names: List[str] = []
        trimmed_injected_seqs: List[str] = []
        accumulated = 0
        for name, seq in zip(injected_names, injected_seqs):
          seq_len = len(seq)
          if accumulated + seq_len > remaining_budget:
            break
          trimmed_injected_names.append(name)
          trimmed_injected_seqs.append(seq)
          accumulated += seq_len

        injected_names = trimmed_injected_names
        injected_seqs = trimmed_injected_seqs
        if include_initial:
          all_names = base_initial_names + injected_names
          all_seqs = base_initial_seqs + injected_seqs
        else:
          all_names = list(injected_names)
          all_seqs = list(injected_seqs)
    output_fasta(all_names, all_seqs, str(profam_input_fasta))

    # Log the prompt sequences being used for this cycle.
    print(f"  Prompt sequences ({len(all_seqs)} total):")
    for i, (name, seq) in enumerate(zip(all_names, all_seqs)):
      if i < 3 or i == len(all_seqs) - 1:  # show first 3 and last
        print(f"    [{i}] {name}: {seq}")
      elif i == 3:
        print(f"    ... ({len(all_seqs) - 4} more sequences) ...")

    # Temperature bandit: sample a temperature for this cycle.
    cycle_temperature: float | None = None
    if temp_bandit is not None:
      cycle_temperature = temp_bandit.select()
      original_temperature = cfg.profam_temperature
      cfg.profam_temperature = cycle_temperature
      print(f"  Temperature bandit: sampled T={cycle_temperature}")
      for t_info in temp_bandit.get_state_dict():
        t = t_info["temperature"]
        print(f"    T={t:.2f}: α={t_info['alpha']:.2f}, β={t_info['beta_param']:.2f}, "
              f"E[θ]={t_info['expected_reward']:.3f}, selected {t_info['times_selected']}x")


    # Proposal bandit: sample which proposal method to use this cycle.
    cycle_proposal_method: str = cfg.proposal_method
    if proposal_bandit is not None:
      cycle_proposal_method = proposal_bandit.select()
      print(f"  Proposal bandit: sampled method={cycle_proposal_method}")
      for p_info in proposal_bandit.get_state_dict():
        m = p_info["method"]
        print(f"    {m}: α={p_info['alpha']:.2f}, β={p_info['beta_param']:.2f}, "
              f"E[θ]={p_info['expected_reward']:.3f}, selected {p_info['times_selected']}x")

    # Step 1 & 2: generation + evaluation, with retry logic for
    # enforce_template=False and deduplication.
    max_generation_attempts = 5
    dedup_retries = 0
    # Random mutation is cheap (no GPU), so allow many more retries.
    max_dedup_retries = 500 if cycle_proposal_method == "random_mutation" else 10
    for attempt in range(1, max_generation_attempts + max_dedup_retries + 1):
      if cycle_proposal_method == "random_mutation":
        # Escalate max_mutations every 10 failed dedup retries to explore further
        effective_max_mutations = cfg.max_mutations + dedup_retries // 20
        gen_names, gen_seqs = run_random_mutation_generation(
          seed_sequences=all_seqs,
          num_samples=cfg.profam_num_samples,
          max_mutations=effective_max_mutations,
          rng=rng,
        )
        grpo_token_data = None  # no token data for random mutation
      else:
        _grpo_active = (grpo_step is not None and grpo_step.should_run(cycle))
        _bt_active = (bt_optimizer is not None and cycle % cfg.bt_every_n_cycles == 0)
        gen_result = run_profam_generation(
          cfg=cfg,
          input_fasta=profam_input_fasta,
          cycle_dir=cycle_dir,
          model=profam_model,
          device=profam_device,
          capture_grpo_tokens=(_grpo_active or _bt_active),
        )
        if _grpo_active or _bt_active:
          gen_names, gen_seqs, grpo_token_data = gen_result
          grpo_token_data["_original_seqs"] = list(gen_seqs)  # snapshot for dedup matching
        else:
          gen_names, gen_seqs = gen_result
          grpo_token_data = None
      if len(gen_seqs) != cfg.profam_num_samples:
        print(
          f"Warning: expected {cfg.profam_num_samples} generated sequences, "
          f"got {len(gen_seqs)}."
        )

      # ── Fix 1 & 4: Compute raw diversity BEFORE dedup (true model diversity) ──
      n_raw_batch = len(gen_seqs)
      n_unique_in_batch = len(set(gen_seqs))
      n_seen_before = sum(1 for s in gen_seqs if s in seen_sequences) if cfg.deduplicate_sequences else 0
      n_prompt_copies = sum(1 for s in gen_seqs if s in set(all_seqs))
      raw_unique_fraction = n_unique_in_batch / n_raw_batch if n_raw_batch > 0 else 0.0
      raw_novel_fraction = (n_raw_batch - n_seen_before) / n_raw_batch if n_raw_batch > 0 else 0.0
      print(f"  Raw diversity: {n_unique_in_batch}/{n_raw_batch} unique in batch, "
            f"{n_raw_batch - n_seen_before}/{n_raw_batch} novel, "
            f"{n_prompt_copies} prompt copies")

      # Adjust temperature inside the retry loop so we get un-stuck faster.
      # Only relevant when using profam proposals — random_mutation ignores temperature.
      if temp_bandit is None and cycle_proposal_method == "profam":
        avg_sim_for_temp = compute_avg_sequence_similarity(gen_seqs, all_seqs)
        prev_temp = current_temperature
        current_temperature, temp_reason = adjust_temperature(
          raw_unique_fraction, raw_novel_fraction, avg_sim_for_temp,
          current_temperature, starting_temperature,
        )
        if temp_reason:
          print(f"  Adaptive temp (in-loop): {temp_reason}, "
                f"T {prev_temp:.2f} → {current_temperature:.2f}")
          cfg.profam_temperature = current_temperature

      # ── Fix 6: Invalidate grpo_token_data if dedup changes the batch ──
      batch_modified = False

      # Ensure all sequences in the batch are unique (within the batch).
      # When replacing duplicates, capture token data so GRPO/BT stays valid.
      _capture_tokens = (_grpo_active or _bt_active) if cycle_proposal_method == "profam" else False
      if cycle_proposal_method == "profam" and n_unique_in_batch < n_raw_batch:
        unique_seqs_set = set()
        keep_idx = []
        for i, s in enumerate(gen_seqs):
          if s not in unique_seqs_set:
            unique_seqs_set.add(s)
            keep_idx.append(i)
        n_need = len(gen_seqs) - len(keep_idx)
        print(f"  Intra-batch dedup: {len(keep_idx)} unique, {n_need} duplicates — regenerating extras")

        # Collect token data for replacement sequences
        extra_token_chunks = []
        for resample_attempt in range(3):
          if n_need <= 0:
            break
          extra_result = run_profam_generation(
            cfg=cfg, input_fasta=profam_input_fasta, cycle_dir=cycle_dir,
            model=profam_model, device=profam_device,
            capture_grpo_tokens=_capture_tokens,
          )
          if _capture_tokens:
            extra_names, extra_seqs, extra_tokens = extra_result
          else:
            extra_names, extra_seqs = extra_result[0], extra_result[1]
            extra_tokens = None
          for j, s in enumerate(extra_seqs):
            if s not in unique_seqs_set and n_need > 0:
              unique_seqs_set.add(s)
              gen_names.append(extra_names[j])
              gen_seqs.append(s)
              if extra_tokens is not None:
                extra_token_chunks.append({
                  "generated_tokens": extra_tokens["generated_tokens"][j:j+1],
                  "old_per_token_lps": extra_tokens["old_per_token_lps"][j:j+1],
                  "old_per_token_mask": extra_tokens["old_per_token_mask"][j:j+1],
                })
              n_need -= 1

        # Trim back to original batch size, tracking which indices survive
        seen_in_batch = set()
        final_names, final_seqs = [], []
        final_keep_orig = []  # indices into original gen_seqs (< n_raw_batch)
        final_extra_idx = []  # indices into extra_token_chunks
        extra_counter = 0
        for idx, (nm, s) in enumerate(zip(gen_names, gen_seqs)):
          if s not in seen_in_batch and len(final_seqs) < cfg.profam_num_samples:
            seen_in_batch.add(s)
            final_names.append(nm)
            final_seqs.append(s)
            if idx < n_raw_batch:
              final_keep_orig.append(idx)
            else:
              final_extra_idx.append(extra_counter)
              extra_counter += 1 if idx >= n_raw_batch else 0
        gen_names, gen_seqs = final_names, final_seqs

        # Rebuild grpo_token_data: original survivors + replacement token data
        if grpo_token_data is not None and _capture_tokens:
          orig_t = grpo_token_data["generated_tokens"]
          orig_lp = grpo_token_data["old_per_token_lps"]
          orig_m = grpo_token_data["old_per_token_mask"]

          parts_t, parts_lp, parts_m = [], [], []
          # Original survivors
          if final_keep_orig:
            idx_t = torch.tensor(final_keep_orig)
            parts_t.append(orig_t[idx_t])
            parts_lp.append(orig_lp[idx_t])
            parts_m.append(orig_m[idx_t])
          # Replacement sequences
          for ei in final_extra_idx:
            if ei < len(extra_token_chunks):
              parts_t.append(extra_token_chunks[ei]["generated_tokens"])
              parts_lp.append(extra_token_chunks[ei]["old_per_token_lps"])
              parts_m.append(extra_token_chunks[ei]["old_per_token_mask"])

          if parts_t:
            # Pad to common length before concatenating
            max_t_len = max(p.shape[1] for p in parts_t)
            max_lp_len = max(p.shape[1] for p in parts_lp)
            pad_id = profam_model.tokenizer.pad_token_id
            padded_t = [torch.nn.functional.pad(p, (0, max_t_len - p.shape[1]), value=pad_id) for p in parts_t]
            padded_lp = [torch.nn.functional.pad(p, (0, max_lp_len - p.shape[1]), value=0.0) for p in parts_lp]
            padded_m = [torch.nn.functional.pad(p, (0, max_lp_len - p.shape[1]), value=False) for p in parts_m]
            grpo_token_data = {
              "input_ids": grpo_token_data["input_ids"],
              "generated_tokens": torch.cat(padded_t, dim=0),
              "old_per_token_lps": torch.cat(padded_lp, dim=0),
              "old_per_token_mask": torch.cat(padded_m, dim=0),
              "_original_seqs": list(gen_seqs),
            }
            print(f"  grpo_token_data rebuilt: {len(final_keep_orig)} original + "
                  f"{len(final_extra_idx)} replacement entries")
          batch_modified = False  # token data is now consistent
        else:
          batch_modified = True
        batch_modified = True

      # Deduplication against previously seen sequences.
      if cfg.deduplicate_sequences:
        novel_mask = [seq not in seen_sequences for seq in gen_seqs]
        n_novel = sum(novel_mask)
        n_dup = len(gen_seqs) - n_novel
        if n_dup > 0:
          dup_seqs_preview = [s[:30] for s, is_novel in zip(gen_seqs, novel_mask) if not is_novel]
          print(f"  Dedup: {n_novel} novel, {n_dup} duplicate(s) "
                f"(cache size: {len(seen_sequences)})")
          for dp in dup_seqs_preview[:3]:
            print(f"    dup: {dp}...")
        if n_novel == 0 and dedup_retries < max_dedup_retries:
          dedup_retries += 1
          print(f"  Dedup retry {dedup_retries}/{max_dedup_retries}: "
                f"all {len(gen_seqs)} sequences are duplicates, regenerating...")
          continue
        if n_novel == 0:
          # ── Fix 2: Exhausted retries — mutate prompt sequences to get novel ones ──
          print(f"  Dedup: exhausted {max_dedup_retries} retries, "
                f"generating novel sequences via random mutation of prompt")
          mut_names, mut_seqs = run_random_mutation_generation(
            seed_sequences=all_seqs,
            num_samples=cfg.profam_num_samples,
            max_mutations=cfg.max_mutations if cfg.max_mutations else 1,
            rng=rng,
          )
          # Keep only truly novel mutations
          novel_muts = [(nm, s) for nm, s in zip(mut_names, mut_seqs)
                        if s not in seen_sequences and s not in set(gen_seqs)]
          if novel_muts:
            gen_names = [nm for nm, _ in novel_muts[:cfg.profam_num_samples]]
            gen_seqs = [s for _, s in novel_muts[:cfg.profam_num_samples]]
            novel_mask = [True] * len(gen_seqs)
            n_novel = len(gen_seqs)
            batch_modified = True
            print(f"  Generated {len(gen_seqs)} novel mutants")
          else:
            # True exhaustion — use cached results
            print(f"  Could not generate novel mutants either, using cached results")
            energies: List[float] = []
            details: List[Dict[str, Any]] = []
            folding_results: List[Any] = []
            for seq in gen_seqs:
              cached_energy, cached_detail = seen_sequences[seq]
              energies.append(cached_energy)
              details.append(cached_detail)
              folding_results.append(None)
            break

        # We have at least some novel sequences. Separate novel vs cached.
        novel_seqs = [s for s, is_novel in zip(gen_seqs, novel_mask) if is_novel]
        novel_names = [n for n, is_novel in zip(gen_names, novel_mask) if is_novel]

        # Evaluate only novel sequences.
        novel_energies, novel_details, novel_folding = evaluate_sequences_with_bagel(
          sequences=novel_seqs,
          energy_cfg=energy_cfg,
          folding_oracle=folding_oracles,
          cycle_index=cycle,
          cycle_dir=cycle_dir,
          enforce_template=cfg.enforce_template,
            save_structures=cfg.save_structures,
        )

        # Merge results: novel gets fresh evaluation, duplicates get cached.
        energies = []
        details = []
        folding_results = []
        novel_idx = 0
        for i, seq in enumerate(gen_seqs):
          if novel_mask[i]:
            energies.append(novel_energies[novel_idx])
            details.append(novel_details[novel_idx])
            folding_results.append(novel_folding[novel_idx])
            # Add novel sequence to cache.
            seen_sequences[seq] = (novel_energies[novel_idx], novel_details[novel_idx])
            novel_idx += 1
          else:
            cached_energy, cached_detail = seen_sequences[seq]
            energies.append(cached_energy)
            details.append(cached_detail)
            folding_results.append(None)
      else:
        # No deduplication — evaluate all sequences.
        energies, details, folding_results = evaluate_sequences_with_bagel(
          sequences=gen_seqs,
          energy_cfg=energy_cfg,
          folding_oracle=folding_oracles,
          cycle_index=cycle,
          cycle_dir=cycle_dir,
          enforce_template=cfg.enforce_template,
            save_structures=cfg.save_structures,
        )

      # When enforce_template is False, sequences with template mismatches
      # receive inf energy. If ALL sequences have inf, regenerate.
      if not cfg.enforce_template and all(e == float("inf") for e in energies):
        print(
          f"  Attempt {attempt}/{max_generation_attempts}: all sequences have "
          f"inf energy (template mismatch), regenerating..."
        )
        if attempt < max_generation_attempts:
          continue
      break
    else:
      print(
        f"Warning: all generation attempts produced only inf-energy or "
        f"duplicate sequences in cycle {cycle}. Proceeding with last batch."
      )

    # Fix 6: If dedup modified the batch, token data is stale for replaced
    # positions.  Keep only the subset of sequences that survived unchanged.
    # The token IDs, log-probs, and masks at those positions are still valid.
    if batch_modified and grpo_token_data is not None:
      orig_seqs = grpo_token_data.get("_original_seqs")
      if orig_seqs is not None:
        # Find indices of original sequences that are still in the final batch
        keep = [i for i, s in enumerate(orig_seqs) if s in gen_seqs]
        if keep:
          keep_t = torch.tensor(keep)
          grpo_token_data = {
            "input_ids": grpo_token_data["input_ids"],
            "generated_tokens": grpo_token_data["generated_tokens"][keep_t],
            "old_per_token_lps": grpo_token_data["old_per_token_lps"][keep_t],
            "old_per_token_mask": grpo_token_data["old_per_token_mask"][keep_t],
          }
          print(f"  grpo_token_data: kept {len(keep)}/{len(orig_seqs)} valid entries after dedup")
        else:
          grpo_token_data = None
          print("  grpo_token_data invalidated (no original sequences survived dedup)")
      else:
        grpo_token_data = None
        print("  grpo_token_data invalidated (batch was modified by dedup)")

    # Log all generated sequences to CSV.
    append_cycle_csv(
      csv_path=cfg.output_dir / "all_sequences.csv",
      cycle_index=cycle,
      names=gen_names,
      sequences=gen_seqs,
      energies=energies,
      details=details,
      folding_results=folding_results,
      initial_seqs=base_initial_seqs,
      prompt_seqs=all_seqs,
      proposal_method=cycle_proposal_method,
    )

    # Assign global unique IDs to this cycle's sequences.
    gen_ids = list(range(next_global_id, next_global_id + len(gen_seqs)))
    next_global_id += len(gen_seqs)

    # Compute average sequence similarity to original and current prompt.
    avg_sim = compute_avg_sequence_similarity(gen_seqs, base_initial_seqs)
    avg_sim_to_prompt = compute_avg_sequence_similarity(gen_seqs, all_seqs)
    print(f"  Avg sequence similarity to original: {avg_sim:.4f}")
    print(f"  Avg sequence similarity to prompt:   {avg_sim_to_prompt:.4f}")

    # Update global elite if this cycle's best beats it.
    elite_energy_before_cycle = elite_energy
    cycle_best_idx = int(np.argmin(energies))
    cycle_best_energy = float(energies[cycle_best_idx])
    if cycle_best_energy < elite_energy:
      elite_energy = cycle_best_energy
      elite_seq = gen_seqs[cycle_best_idx]
      elite_name = gen_names[cycle_best_idx]
      elite_cycle = cycle
      print(f"  New global elite: energy={elite_energy:.4f} (cycle {elite_cycle})")
      _save_and_log_best_structure(
        folding_result=(
          folding_results[cycle_best_idx]
          if cycle_best_idx < len(folding_results) else None
        ),
        output_dir=cfg.output_dir,
        wandb_run=wandb_run,
        cycle=cycle,
        energy=elite_energy,
        sequence=elite_seq,
        name=elite_name,
      )

    # ---- Likelihood tracking: update best/worst sequence pools ----
    if cfg.likelihood_eval_every > 0 and grpo_token_data is not None:
      track_n = cfg.likelihood_track_n
      prompt_ids = grpo_token_data["input_ids"].cpu()  # (1, L_prompt)
      for i, (seq, energy) in enumerate(zip(gen_seqs, energies)):
        if not np.isfinite(energy):
          continue
        entry = (energy, seq, prompt_ids.clone())
        # Update best (lowest energy)
        likelihood_best.append(entry)
        likelihood_best.sort(key=lambda x: x[0])
        if len(likelihood_best) > track_n:
          likelihood_best = likelihood_best[:track_n]
        # Update worst (highest energy)
        likelihood_worst.append(entry)
        likelihood_worst.sort(key=lambda x: x[0], reverse=True)
        if len(likelihood_worst) > track_n:
          likelihood_worst = likelihood_worst[:track_n]

      # Every N cycles, evaluate model likelihoods of best/worst sequences
      if cycle > 0 and cycle % cfg.likelihood_eval_every == 0 and profam_model is not None:
        profam_model.eval()
        tok = profam_model.tokenizer
        sep_id = tok.sep_token_id
        pad_id = tok.pad_token_id

        def _compute_avg_ll(entries):
          """Compute average model log-likelihood for a list of (energy, seq, prompt_ids)."""
          if not entries:
            return float("nan")
          all_lls = []
          for _, seq, p_ids in entries:
            seq_token_ids = [tok.convert_tokens_to_ids(aa) for aa in seq]
            seq_tensor = torch.tensor([seq_token_ids], device=profam_device)
            sep_col = torch.full((1, 1), sep_id, device=profam_device)
            comp_ids = torch.cat([sep_col, seq_tensor], dim=1).unsqueeze(0)  # (1, 1, L+1)
            with torch.no_grad():
              lps, mask = profam_model._compute_per_token_log_probs_for_grpo(
                input_ids=p_ids.to(profam_device),
                completion_ids=comp_ids,
              )
            valid_lps = lps[mask]
            if valid_lps.numel() > 0:
              all_lls.append(valid_lps.mean().item())
          return float(np.mean(all_lls)) if all_lls else float("nan")

        ll_best = _compute_avg_ll(likelihood_best)
        ll_worst = _compute_avg_ll(likelihood_worst)
        n_best = len(likelihood_best)
        n_worst = len(likelihood_worst)
        avg_e_best = np.mean([e for e, _, _ in likelihood_best]) if likelihood_best else float("nan")
        avg_e_worst = np.mean([e for e, _, _ in likelihood_worst]) if likelihood_worst else float("nan")
        print(f"  Likelihood eval: best({n_best}) avg_ll={ll_best:.4f} avg_e={avg_e_best:.4f} | "
              f"worst({n_worst}) avg_ll={ll_worst:.4f} avg_e={avg_e_worst:.4f}")
        if wandb_run is not None:
          import wandb
          wandb.log({
            "likelihood/best_avg_ll": ll_best,
            "likelihood/worst_avg_ll": ll_worst,
            "likelihood/best_avg_energy": avg_e_best,
            "likelihood/worst_avg_energy": avg_e_worst,
            "likelihood/ll_gap": ll_best - ll_worst,
          }, step=cycle)

    # ---- GRPO online RL step (optional, same-batch dual use) ----
    # True GRPO with PPO clipping.  Merges the current cycle's scored batch
    # with up to GRPO_REPLAY_SIZE cached past cycles for a larger effective
    # group size (e.g. 8 seqs/cycle × 4 = 32 effective group).  Older data
    # is slightly off-policy but PPO clipping handles this naturally.
    rl_metrics = None
    if grpo_step is not None and grpo_step.should_run(cycle) and grpo_token_data is None:
      print(f"  GRPO skipped: no token data (proposal_method={cycle_proposal_method})")
    if grpo_step is not None and grpo_step.should_run(cycle) and grpo_token_data is not None:
      # Convert current cycle's energies to rewards
      rewards_np = np.array([-e if np.isfinite(e) else 0.0 for e in energies], dtype=np.float32)
      current_rewards = torch.tensor(rewards_np, device=profam_device)

      # Save current cycle to replay buffer.
      # Ensure token data and rewards have matching batch sizes — they can
      # diverge if template enforcement or dedup changes the sequence count.
      n_gen = grpo_token_data["generated_tokens"].shape[0]
      n_rew = current_rewards.shape[0]
      n_keep = min(n_gen, n_rew)
      current_entry = {
        "generated_tokens": grpo_token_data["generated_tokens"][:n_keep],      # (G, L) CPU
        "old_per_token_lps": grpo_token_data["old_per_token_lps"][:n_keep],    # (G, L-1) CPU
        "old_per_token_mask": grpo_token_data["old_per_token_mask"][:n_keep],  # (G, L-1) CPU
        "rewards": current_rewards[:n_keep].cpu(),                              # (G,) CPU
        "sequences": list(gen_seqs[:n_keep]),                                   # for seen-before check
      }
      if n_gen != n_rew:
        print(f"  GRPO: trimmed batch {n_gen} tokens vs {n_rew} rewards -> {n_keep}")
      grpo_replay_buffer.append(current_entry)
      if len(grpo_replay_buffer) > cfg.grpo_replay_cycles + 1:  # +1 for current
        grpo_replay_buffer.pop(0)

      # Merge all cached entries: pad to same token length, concatenate
      all_gen_tokens = []
      all_old_lps = []
      all_old_masks = []
      all_rewards = []
      for entry in grpo_replay_buffer:
        all_gen_tokens.append(entry["generated_tokens"])
        all_old_lps.append(entry["old_per_token_lps"])
        all_old_masks.append(entry["old_per_token_mask"])
        all_rewards.append(entry["rewards"])

      # Pad token tensors to the max length across all cached cycles
      max_gen_len = max(t.shape[1] for t in all_gen_tokens)
      max_lp_len = max(t.shape[1] for t in all_old_lps)
      pad_id = profam_model.tokenizer.pad_token_id

      padded_tokens = []
      padded_lps = []
      padded_masks = []
      for tokens, lps, masks in zip(all_gen_tokens, all_old_lps, all_old_masks):
        if tokens.shape[1] < max_gen_len:
          tokens = torch.nn.functional.pad(tokens, (0, max_gen_len - tokens.shape[1]), value=pad_id)
        if lps.shape[1] < max_lp_len:
          lps = torch.nn.functional.pad(lps, (0, max_lp_len - lps.shape[1]), value=0.0)
          masks = torch.nn.functional.pad(masks, (0, max_lp_len - masks.shape[1]), value=False)
        padded_tokens.append(tokens)
        padded_lps.append(lps)
        padded_masks.append(masks)

      merged_tokens = torch.cat(padded_tokens, dim=0)   # (total_G, L)
      merged_lps = torch.cat(padded_lps, dim=0)         # (total_G, L-1)
      merged_masks = torch.cat(padded_masks, dim=0)     # (total_G, L-1)
      merged_rewards = torch.cat(all_rewards, dim=0).to(profam_device)  # (total_G,)

      n_total = merged_tokens.shape[0]
      n_current = current_rewards.shape[0]
      n_cached = n_total - n_current

      if n_total >= 2:
        profam_model.train()
        mbs = cfg.grpo_micro_batch_size
        n_micro = (n_total + mbs - 1) // mbs  # ceiling division

        # Verify shapes match across merged tensors before GRPO
        assert merged_tokens.shape[0] == merged_rewards.shape[0], (
          f"GRPO shape mismatch: tokens {merged_tokens.shape[0]} vs rewards {merged_rewards.shape[0]}"
        )
        # Pre-compute advantages over the full group so micro-batches
        # use consistent normalisation.
        full_advantages = profam_model._compute_grpo_advantages(merged_rewards.to(profam_device))

        # Fix 5: Clamp advantage to ≤ 0 for CURRENT-CYCLE sequences that
        # were already seen in a prior cycle.  This prevents the model from
        # being rewarded for re-generating known sequences, while leaving
        # replay buffer entries untouched (they serve as a normalisation
        # baseline for advantage computation, not as generation targets).
        current_seqs = grpo_replay_buffer[-1].get("sequences", [])
        if current_seqs and cfg.deduplicate_sequences:
          # Sequences first seen this cycle won't be in prior_seen
          prior_seen = set()
          for entry in grpo_replay_buffer[:-1]:
            prior_seen.update(entry.get("sequences", []))

          # Current-cycle entries are at the END of the merged tensor
          offset = n_total - len(current_seqs)
          n_clamped = 0
          for j, seq in enumerate(current_seqs):
            idx = offset + j
            if idx < len(full_advantages) and seq in prior_seen:
              full_advantages[idx] = torch.clamp(full_advantages[idx], max=0.0)
              n_clamped += 1
          frac_clamped = n_clamped / len(current_seqs) if current_seqs else 0.0
          if n_clamped > 0:
            print(f"  GRPO: clamped advantage ≤ 0 for {n_clamped}/{len(current_seqs)} "
                  f"current-cycle sequences seen in prior cycles ({frac_clamped:.1%})")
          if wandb_run is not None:
            import wandb
            wandb.log({"grpo/seen_clamped_fraction": frac_clamped}, step=cycle)

        for rl_step_i in range(cfg.rl_steps_per_cycle):
          grpo_step.optimizer.zero_grad()
          accum_metrics: Dict[str, float] = {}
          for mb_i in range(n_micro):
            mb_start = mb_i * mbs
            mb_end = min(mb_start + mbs, n_total)
            mb_tokens = merged_tokens[mb_start:mb_end]
            mb_lps = merged_lps[mb_start:mb_end]
            mb_masks = merged_masks[mb_start:mb_end]
            # Pass pre-computed advantages as "rewards" with normalisation
            # disabled so they pass through _compute_grpo_advantages unchanged.
            mb_advantages = full_advantages[mb_start:mb_end]
            orig_normalize = profam_model.grpo_normalize_rewards
            orig_baseline = profam_model.grpo_reward_baseline
            profam_model.grpo_normalize_rewards = False # temporarily disable so we don't normalise twice
            profam_model.grpo_reward_baseline = "none"
            mb_loss, mb_metrics = profam_model.grpo_step_from_rewards(
              input_ids=grpo_token_data["input_ids"],
              generated_tokens=mb_tokens,
              old_per_token_lps=mb_lps,
              old_per_token_mask=mb_masks,
              rewards=mb_advantages,
              clip_ratio=cfg.grpo_clip_ratio,
              beta=cfg.grpo_beta,
            )
            profam_model.grpo_normalize_rewards = orig_normalize
            profam_model.grpo_reward_baseline = orig_baseline
            # Scale loss by micro-batch fraction for correct gradient averaging
            scaled_loss = mb_loss * (mb_end - mb_start) / n_total
            scaled_loss.backward()
            # Accumulate metrics (weighted average)
            weight = (mb_end - mb_start) / n_total
            for k, v in mb_metrics.items():
              if isinstance(v, (int, float)):
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v * weight
          rl_metrics = accum_metrics
          grad_norm = torch.nn.utils.clip_grad_norm_(profam_model.parameters(), 1.0)
          grpo_step.optimizer.step()
          rl_metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
          rl_metrics["lr"] = grpo_step.optimizer.param_groups[0]["lr"]
          rl_metrics["grpo_effective_group_size"] = n_total
          rl_metrics["grpo_cached_seqs"] = n_cached
          print(f"  GRPO step {rl_step_i}: loss={rl_metrics.get('grpo_loss', 0):.4f}, "
                f"clip_frac={rl_metrics.get('clip_fraction', 0):.3f}, "
                f"grad_norm={rl_metrics['grad_norm']:.4f}, "
                f"group={n_total} ({n_current} current + {n_cached} cached, "
                f"{n_micro} micro-batches of {mbs})")
        profam_model.eval()
      else:
        print("  GRPO skipped: fewer than 2 sequences with finite energy")

    # ---- Bradley-Terry ranking update (optional) ----
    if bt_optimizer is not None and cycle % cfg.bt_every_n_cycles == 0:
      # Add current cycle's sequences to the ranking pool
      for seq, energy in zip(gen_seqs, energies):
        if np.isfinite(energy):
          bt_pool.append((energy, seq))
      # Keep the most recent sequences, capped at pool_size.
      # This trains on the current quality frontier for fine-grained ranking.
      if len(bt_pool) > cfg.bt_pool_size:
        bt_pool = bt_pool[-cfg.bt_pool_size:]

      if len(bt_pool) >= 4:  # need at least a few pairs
        from pipeline.bradley_terry import bradley_terry_loss, score_variants_differentiable
        tok = profam_model.tokenizer

        # Compute frozen KV cache for the current prompt
        prompt_ids = grpo_token_data["input_ids"] if grpo_token_data is not None else None
        if prompt_ids is not None:
          with torch.no_grad():
            enc_out = profam_model._encoder_model(prompt_ids.to(profam_device), use_cache=True)
          frozen_kv = tuple(
            tuple(t.detach() for t in layer_kv) for layer_kv in enc_out.past_key_values
          )

          profam_model.train()
          for bt_step in range(cfg.bt_steps_per_cycle):
            # Sample a batch from the pool
            n_sample = min(cfg.bt_batch_size, len(bt_pool))
            indices = rng.choice(len(bt_pool), size=n_sample, replace=False)
            batch_seqs = [bt_pool[i][1] for i in indices]
            batch_energies = [bt_pool[i][0] for i in indices]

            # Tokenize sequences: [SEP] + seq + [SEP]
            comp_tok = tok.encode_completions(
              batch_seqs,
              bos_token=tok.sep_token,
              eos_token=tok.sep_token,
            )
            completion_ids = (
              torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
              .unsqueeze(0)
              .to(profam_device)
            )

            # Fitness = negated energy (higher = better)
            fitness = torch.tensor(
              [-e for e in batch_energies], dtype=torch.float32, device=profam_device
            )

            # Score with gradient flow
            scores = score_variants_differentiable(
              profam_model, frozen_kv, completion_ids,
              sub_batch_size=cfg.bt_sub_batch_size,
            )

            loss = bradley_terry_loss(scores, fitness)
            bt_optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
              [p for p in profam_model.parameters() if p.requires_grad], 1.0
            )
            bt_optimizer.step()

            print(f"  BT step {bt_step}: loss={loss.item():.4f}, "
                  f"grad_norm={grad_norm.item():.4f}, pool={len(bt_pool)}, batch={n_sample}")

            if wandb_run is not None:
              import wandb
              wandb.log({
                "bt/loss": loss.item(),
                "bt/grad_norm": grad_norm.item(),
                "bt/pool_size": len(bt_pool),
              }, step=cycle)

          profam_model.eval()
          del frozen_kv
          torch.cuda.empty_cache()

    # ---- Temperature adjustment ----
    # Use raw diversity metrics (computed BEFORE dedup) for temperature control.
    unique_fraction = raw_unique_fraction

    # Only adjust temperature when using profam proposals — random_mutation ignores temperature.
    if temp_bandit is None and cycle_proposal_method == "profam":
      prev_temperature = current_temperature
      current_temperature, temp_reason = adjust_temperature(
        raw_unique_fraction, raw_novel_fraction, avg_sim_to_prompt,
        current_temperature, starting_temperature,
      )
      if temp_reason:
        print(f"  Adaptive temp: {temp_reason}, "
              f"raising temperature {prev_temperature:.2f} → {current_temperature:.2f}")
      elif current_temperature != prev_temperature:
        print(f"  Adaptive temp: all unique & novel, "
              f"lowering temperature {prev_temperature:.2f} → {current_temperature:.2f}")
      cfg.profam_temperature = current_temperature

    # ---- Weights & Biases logging ----
    if wandb_run is not None:
      import wandb
      log_dict = {
        "cycle": cycle,
        "energy/all_min": float(min(energies)) if energies else float("inf"),
        "energy/all_mean": float(np.mean(energies)) if energies else float("inf"),
        "energy/elite": elite_energy,
        "energy/elite_cycle": elite_cycle,
        "similarity/to_initial": avg_sim,
        "similarity/to_prompt": avg_sim_to_prompt,
        "generation/num_sequences": len(gen_seqs),
        "generation/unique_fraction": unique_fraction,
        "generation/raw_unique_fraction": raw_unique_fraction,
        "generation/raw_novel_fraction": raw_novel_fraction,
        "generation/prompt_copies": n_prompt_copies,
        "generation/temperature": cycle_temperature if cycle_temperature is not None else current_temperature,
        "generation/mean_length": float(np.mean([len(s) for s in gen_seqs])),
        "generation/proposal_method": 1.0 if cycle_proposal_method == "profam" else 0.0,
      }
      if proposal_bandit is not None:
        for p_info in proposal_bandit.get_state_dict():
          m = p_info["method"]
          log_dict[f"bandit/{m}_p_improve"] = p_info["expected_reward"]
          log_dict[f"bandit/{m}_selected"] = p_info["times_selected"]
      # Per-energy-term breakdown (if available)
      if details:
        for key in details[0].get("energy_terms", {}):
          term_vals = [d.get("energy_terms", {}).get(key) for d in details if d.get("energy_terms")]
          finite_vals = [v for v in term_vals if v is not None and np.isfinite(v)]
          if finite_vals:
            log_dict[f"energy_terms/{key}_min"] = float(min(finite_vals))
            log_dict[f"energy_terms/{key}_mean"] = float(np.mean(finite_vals))
      # GRPO metrics (if run this cycle)
      if rl_metrics is not None:
        for k, v in rl_metrics.items():
          if isinstance(v, (int, float)) and not isinstance(v, bool):
            log_dict[f"grpo/{k}"] = v
      wandb.log(log_dict, step=cycle)

    # Build the selection pool: current cycle + up to n_memory previous cycles.
    if cfg.n_memory > 0 and memory_buffer:
      pool_ids: List[int] = []
      pool_names: List[str] = []
      pool_seqs: List[str] = []
      pool_energies: List[float] = []
      for mem_ids, mem_names, mem_seqs, mem_energies in memory_buffer:
        pool_ids.extend(mem_ids)
        pool_names.extend(mem_names)
        pool_seqs.extend(mem_seqs)
        pool_energies.extend(mem_energies)
      pool_offset = len(pool_seqs)  # index where current cycle starts in pool
      pool_ids.extend(gen_ids)
      pool_names.extend(gen_names)
      pool_seqs.extend(gen_seqs)
      pool_energies.extend(energies)
      print(
        f"  Memory pool: {len(pool_seqs)} sequences "
        f"({pool_offset} from memory + {len(gen_seqs)} current)"
      )
    else:
      pool_ids = list(gen_ids)
      pool_names = list(gen_names)
      pool_seqs = list(gen_seqs)
      pool_energies = list(energies)
      pool_offset = 0

    # --- SelectionManager branch: unified Thompson/greedy arm management ---
    thompson_cycle_state: Dict[str, Any] | None = None
    thompson_selected_arm_id: int | None = None
    thompson_progeny_reward: float | None = None

    if selection_manager is not None:
      # Get reward statistics for logging.
      reward_stats = selection_manager.get_reward_stats(details)
      print(f"  Selection [{reward_stats['reward_term']}]: "
            f"{reward_stats['n_finite']} finite, {reward_stats['n_inf']} inf "
            f"out of {reward_stats['n_total']} progeny")
      if reward_stats['n_finite'] > 0:
        print(f"    progeny {reward_stats['reward_term']} range: "
              f"[{reward_stats['min']:.4f}, {reward_stats['max']:.4f}], "
              f"mean={reward_stats['mean']:.4f}")

      # 1. Update the parent arm's posterior with the best progeny's reward.
      update_stats = selection_manager.update_parent_posterior(details)
      if update_stats is not None:
        print(f"  Thompson POSTERIOR UPDATE for parent arm {update_stats['parent_arm_id']} "
              f"({update_stats['parent_name']}):")
        if update_stats['updated']:
          print(f"    before: α={update_stats['alpha_before']:.4f}, "
                f"β={update_stats['beta_before']:.4f}")
          print(f"    best progeny: idx={update_stats['best_progeny_idx']}, "
                f"energy={update_stats['best_progeny_ipsae']:.4f}, "
                f"reward={update_stats['progeny_reward']:.4f}")
          print(f"    after:  α={update_stats['alpha_after']:.4f}, "
                f"β={update_stats['beta_after']:.4f}, "
                f"E[θ]={update_stats['expected_reward_after']:.4f}")
          thompson_progeny_reward = update_stats['progeny_reward']
        else:
          print(f"    no finite progeny — posterior unchanged")

      # 2. Update temperature bandit with the progeny reward.
      if temp_bandit is not None and cycle_temperature is not None:
        temp_reward = thompson_progeny_reward if thompson_progeny_reward is not None else 0.0
        temp_bandit.update(cycle_temperature, temp_reward)
        print(f"  Temperature bandit UPDATE: T={cycle_temperature}, "
              f"reward={temp_reward:.4f}")

      # 3. Update proposal bandit: did this cycle improve over the previous best?
      if proposal_bandit is not None:
        cycle_best = float(min(energies)) if energies else float("inf")
        improved = cycle_best < elite_energy_before_cycle
        proposal_bandit.update(cycle_proposal_method, improved)
        print(f"  Proposal bandit UPDATE: method={cycle_proposal_method}, "
              f"improved={improved} (cycle_best={cycle_best:.4f}, prev_elite={elite_energy_before_cycle:.4f})")

      # 4. Register ALL progeny as new arms (THE BUG FIX - this now happens
      #    for BOTH selection_strategy="thompson" AND "greedy" with bandit).
      reg_stats = selection_manager.register_progeny(gen_names, gen_seqs, details, cycle)
      print(f"  Thompson: registered {reg_stats['n_registered']} new arms "
            f"(total arms: {reg_stats['n_total_arms']})")

      # 5. Prune arms to maintain diversity.
      prune_stats = selection_manager.prune_arms()
      if prune_stats.get('pruned'):
        print(f"  Thompson PRUNING: {prune_stats['arms_before']} → "
              f"{prune_stats['arms_after']} arms "
              f"(max_identity={selection_manager.max_identity:.2f})")
        print(f"    retained: {prune_stats['retained_ids']}")
        print(f"    pruned: {prune_stats['pruned_ids']}")

      # 6. Select next prompt using the configured selector (greedy or thompson).
      b = selection_manager.thompson_sampler.exploit_bias
      selector_name = selection_manager.prompt_selector.method_name
      print(f"  ARM SELECTION ({selector_name}, "
            f"m_samples={selection_manager.thompson_sampler.m_samples}, "
            f"exploit_bias={b:.1f}):")
      selection_result = selection_manager.select_prompt()
      next_arm = selection_result.selected_arm
      thompson_selected_arm_id = next_arm.arm_id if next_arm else None

      print(f"  {selector_name.upper()} SELECTED → arm {next_arm.arm_id} ({next_arm.name})")
      print(f"    ipSAE_raw={next_arm.ipsae_raw:.4f} "
            f"(of {len(selection_manager.thompson_sampler.arms)} arms)")
      print(f"    times_selected={next_arm.times_selected}, "
            f"parent_arm={next_arm.parent_arm_id}, "
            f"created_cycle={next_arm.created_at_cycle}")
      print(f"  PROMPT SEQUENCE: {next_arm.sequence}")

      # 7. Build injection set from selection result.
      injected_names, injected_seqs = selection_manager.build_injection_set(selection_result)

      # Update prev_injection_best_energy for next cycle's relative reward calculation.
      prev_injection_best_energy = selection_result.selected_ipsae

      # For logging compatibility, create minimal selected_indices pointing at
      # the best generated sequence this cycle.
      selected_indices = np.array([int(np.argmin(energies))])

      # Build detailed thompson cycle state for JSON logging.
      top_arms_summary = selection_manager.get_top_arms_summary(n=10)
      thompson_cycle_state = {
        "num_arms": len(selection_manager.thompson_sampler.arms),
        "selected_arm_id": thompson_selected_arm_id,
        "selected_arm_name": next_arm.name,
        "selected_arm_expected_reward": next_arm.alpha / (next_arm.alpha + next_arm.beta_param),
        "selected_arm_alpha": next_arm.alpha,
        "selected_arm_beta": next_arm.beta_param,
        "selected_arm_times_selected": next_arm.times_selected,
        "selection_method": selector_name,
        "progeny_finite_count": reward_stats['n_finite'],
        "progeny_inf_count": reward_stats['n_inf'],
        "top_10_arms": top_arms_summary,
      }
      if cycle_temperature is not None:
        thompson_cycle_state["sampled_temperature"] = cycle_temperature
      if temp_bandit is not None:
        thompson_cycle_state["temperature_bandit"] = temp_bandit.get_state_dict()
      if proposal_bandit is not None:
        thompson_cycle_state["proposal_method"] = cycle_proposal_method
        thompson_cycle_state["proposal_bandit"] = proposal_bandit.get_state_dict()

      # Save full thompson arms state to dedicated file.
      arms_path = cfg.output_dir / "thompson_arms.json"
      with arms_path.open("w") as f:
        json.dump(selection_manager.thompson_sampler.get_state_dict(), f, indent=2)

      # Save per-cycle thompson decision log (append, one entry per cycle).
      # Reward signal is always the composite weighted total energy.
      reward_values_for_log = [
        float(d.get("energy", float("inf"))) if isinstance(d, dict) else float("inf")
        for d in details
      ]
      decision_log_path = cfg.output_dir / "thompson_decisions.jsonl"
      decision_entry: Dict[str, Any] = {
        "cycle": cycle,
        "selected_arm_id": thompson_selected_arm_id,
        "selected_arm_name": next_arm.name,
        "selected_arm_alpha": next_arm.alpha,
        "selected_arm_beta": next_arm.beta_param,
        "selected_arm_expected_reward": next_arm.alpha / (next_arm.alpha + next_arm.beta_param),
        "selection_method": selector_name,
        "progeny_reward": thompson_progeny_reward,
        "progeny_finite_count": reward_stats['n_finite'],
        "total_arms": len(selection_manager.thompson_sampler.arms),
        "progeny_ipsae_values": [
          round(v, 6) if math.isfinite(v) else None for v in reward_values_for_log
        ],
        "top_10_arms": top_arms_summary,
      }
      if cycle_temperature is not None:
        decision_entry["sampled_temperature"] = cycle_temperature
      if temp_bandit is not None:
        decision_entry["temperature_bandit"] = temp_bandit.get_state_dict()
      if proposal_bandit is not None:
        decision_entry["proposal_method"] = cycle_proposal_method
        decision_entry["proposal_bandit"] = proposal_bandit.get_state_dict()
      with decision_log_path.open("a") as f:
        f.write(json.dumps(decision_entry) + "\n")

      # Save statistics.
      swap_accepted = True
      swap_reason = f"selection_manager_{selector_name}"
      update_cycle_log(
        log_path=cycle_log_path,
        cycle_index=cycle,
        selected_indices=selected_indices,
        energies=energies,
        sequence_details=details,
        avg_similarity=avg_sim,
        avg_similarity_to_prompt=avg_sim_to_prompt,
        global_ids=gen_ids,
        thompson_state=thompson_cycle_state,
        thompson_selected_arm_id=thompson_selected_arm_id,
        thompson_progeny_reward=thompson_progeny_reward,
        proposal_method=cycle_proposal_method,
        prompt_sequences=list(all_seqs),
        raw_unique_fraction=raw_unique_fraction,
        raw_novel_fraction=raw_novel_fraction,
      )
      save_selected_structures(
        cycle_index=cycle,
        selected_indices=selected_indices,
        folding_results=folding_results,
        output_dir=cfg.output_dir,
        pool_offset=pool_offset,
        save_structures=cfg.save_structures,
      )

      # Restore original temperature on cfg so it doesn't leak.
      if temp_bandit is not None and cycle_temperature is not None:
        cfg.profam_temperature = original_temperature  # type: ignore[possibly-undefined]

    else:
      # --- Greedy selection (original path) ---

      # Step 3: probabilities via softmax(-energy / T) over the full pool.
      probs = softmax_from_energies(
        energies=pool_energies,
        temperature=cfg.softmax_temperature,
      )

      # Step 4: sample subset according to probs.
      # k is always based on the current generation size, not the pool size.
      k_inject = max(1, int(math.floor(cfg.f_inject * len(gen_seqs))))
      selected_indices = sample_subset_indices(
        num_items=len(pool_seqs),
        probs=probs,
        f_inject=cfg.f_inject,
        rng=rng,
        replace=cfg.sample_with_reinsertion,
        energies=pool_energies,
        subset_size=k_inject,
      )

      # Build candidate injection set from selected indices.
      candidate_names = [pool_names[int(i)] for i in selected_indices]
      candidate_seqs = [pool_seqs[int(i)] for i in selected_indices]
      candidate_energies = [float(pool_energies[int(i)]) for i in selected_indices]

      # --- Elitism: ensure global best sequence occupies position 0 ---
      if cfg.elitism and elite_seq is not None:
        if elite_seq in candidate_seqs:
          # Move elite to position 0.
          ei = candidate_seqs.index(elite_seq)
          if ei != 0:
            candidate_names[0], candidate_names[ei] = candidate_names[ei], candidate_names[0]
            candidate_seqs[0], candidate_seqs[ei] = candidate_seqs[ei], candidate_seqs[0]
            candidate_energies[0], candidate_energies[ei] = candidate_energies[ei], candidate_energies[0]
        else:
          # Replace worst candidate with elite.
          worst_idx = int(np.argmax(candidate_energies))
          candidate_names[worst_idx] = elite_name  # type: ignore[assignment]
          candidate_seqs[worst_idx] = elite_seq
          candidate_energies[worst_idx] = elite_energy
          # Move elite to position 0.
          if worst_idx != 0:
            candidate_names[0], candidate_names[worst_idx] = candidate_names[worst_idx], candidate_names[0]
            candidate_seqs[0], candidate_seqs[worst_idx] = candidate_seqs[worst_idx], candidate_seqs[0]
            candidate_energies[0], candidate_energies[worst_idx] = candidate_energies[worst_idx], candidate_energies[0]
        print(f"  Elitism: elite at position 0, energy={elite_energy:.4f} from cycle {elite_cycle}")

      # --- Conditional swap: only update injection set if improvement ---
      swap_accepted = True
      swap_reason = "unconditional"
      if cfg.accept_only_improvement:
        candidate_best = min(candidate_energies)
        delta = candidate_best - prev_injection_best_energy
        if delta < 0:
          swap_accepted = True
          swap_reason = "improved"
          print(f"  Swap accepted: candidate best {candidate_best:.4f} < previous {prev_injection_best_energy:.4f}")
        elif annealing_temp is not None and annealing_temp > 0:
          accept_prob = math.exp(-delta / annealing_temp)
          roll = rng.random()
          if roll < accept_prob:
            swap_accepted = True
            swap_reason = f"annealing (p={accept_prob:.3f}, roll={roll:.3f}, T={annealing_temp:.4f})"
            print(f"  Swap accepted via annealing: p={accept_prob:.3f}, T={annealing_temp:.4f}")
          else:
            swap_accepted = False
            swap_reason = f"annealing_rejected (p={accept_prob:.3f}, roll={roll:.3f}, T={annealing_temp:.4f})"
            print(f"  Swap rejected via annealing: p={accept_prob:.3f}, T={annealing_temp:.4f}")
          annealing_temp *= cfg.annealing_decay
        else:
          swap_accepted = False
          swap_reason = "no_improvement"
          print(f"  Swap rejected: candidate best {candidate_best:.4f} >= previous {prev_injection_best_energy:.4f}")

      # Save statistics and selected sequences
      update_cycle_log(
        log_path=cycle_log_path,
        cycle_index=cycle,
        selected_indices=selected_indices,
        energies=energies,
        sequence_details=details,
        avg_similarity=avg_sim,
        avg_similarity_to_prompt=avg_sim_to_prompt,
        global_ids=gen_ids,
        pool_ids=pool_ids if cfg.n_memory > 0 else None,
        pool_energies=pool_energies if cfg.n_memory > 0 else None,
        pool_names=pool_names if cfg.n_memory > 0 else None,
        pool_seqs=pool_seqs if cfg.n_memory > 0 else None,
        swap_accepted=swap_accepted if cfg.accept_only_improvement else None,
        swap_reason=swap_reason if cfg.accept_only_improvement else None,
        elite_energy=elite_energy if cfg.elitism else None,
        elite_cycle=elite_cycle if cfg.elitism else None,
        annealing_temp=annealing_temp if cfg.annealing_initial_temp is not None else None,
        proposal_method=cycle_proposal_method,
        prompt_sequences=list(all_seqs),
        raw_unique_fraction=raw_unique_fraction,
        raw_novel_fraction=raw_novel_fraction,
      )
      save_selected_structures(
        cycle_index=cycle,
        selected_indices=selected_indices,
        folding_results=folding_results,
        output_dir=cfg.output_dir,
        pool_offset=pool_offset,
        save_structures=cfg.save_structures,
      )

      # Update injection set: only on accepted swaps (and only if prompt is not frozen).
      if cfg.freeze_prompt:
        print(f"  freeze_prompt=True: prompt unchanged")
      elif swap_accepted:
        injected_names = candidate_names
        injected_seqs = candidate_seqs
        prev_injection_best_energy = min(candidate_energies)
      else:
        print(f"  Keeping previous injection set (best energy={prev_injection_best_energy:.4f})")

    # Update memory buffer with current cycle's data.
    if cfg.n_memory > 0:
      memory_buffer.append((list(gen_ids), list(gen_names), list(gen_seqs), list(energies)))
      if len(memory_buffer) > cfg.n_memory:
        memory_buffer.pop(0)

    # Periodic checkpoint: regenerate plots and push results every
    # output_frequency cycles (and always at the final cycle).
    output_freq = max(1, cfg.output_frequency)
    if cycle % output_freq == 0 or cycle == cfg.max_cycles:
      make_energy_summary_plot(log_path=cycle_log_path, output_dir=cfg.output_dir)
      if checkpoint_callback:
        checkpoint_callback(_collect_checkpoint_results(cfg))

  # After all cycles, plot summary (always, even without callback).
  make_energy_summary_plot(
    log_path=cycle_log_path,
    output_dir=cfg.output_dir,
  )

  # Finalize wandb: log summary plot and close run.
  if wandb_run is not None:
    import wandb
    plot_path = cfg.output_dir / "energy_summary.png"
    if plot_path.exists():
      wandb.log({"energy_summary": wandb.Image(str(plot_path))})
    wandb.summary["elite_energy"] = elite_energy
    wandb.summary["elite_cycle"] = elite_cycle
    wandb.summary["elite_seq"] = elite_seq
    wandb.finish()

  # Return results for HP search / programmatic use.
  try:
    with cycle_log_path.open("r") as f:
      cycle_stats = json.load(f)
    per_cycle_best = [
      cycle_stats[k].get("all_min_energy", float("inf"))
      for k in sorted(cycle_stats.keys(), key=int)
    ]
    best_energy = min(per_cycle_best) if per_cycle_best else float("inf")
  except Exception:
    per_cycle_best = []
    best_energy = float("inf")

  return {
    "best_energy": best_energy,
    "per_cycle_best": per_cycle_best,
    "elite_energy": elite_energy,
    "elite_seq": elite_seq,
    "elite_cycle": elite_cycle,
    "output_dir": str(cfg.output_dir),
  }


def _save_results_locally(
  cfg: PipelineConfig,
  results: Dict[str, Any],
  label: str = "",
) -> None:
  """Save a results dict (cycle_stats + energy_plot) to the local output dir."""
  if results and results.get("cycle_stats") is not None:
    local_stats_path = cfg.output_dir / "cycle_stats.json"
    with local_stats_path.open("w") as f:
      json.dump(results["cycle_stats"], f, indent=2)

    all_keys = sorted(results["cycle_stats"].keys(), key=int)
    latest_key = all_keys[-1] if all_keys else None
    if latest_key is not None:
      entry = results["cycle_stats"][latest_key]
      best = entry.get("best_sequence", {})
      prefix = f"[{label}] " if label else ""
      print(
        f"{prefix}"
        f"Cycle {entry['cycle']}: "
        f"avg_energy={entry.get('all_avg_energy', 'N/A'):.4f}, "
        f"min_energy={entry.get('all_min_energy', 'N/A'):.4f}, "
        f"best_seq={best.get('sequence', 'N/A')[:60]}..."
      )

  if results and results.get("energy_plot_png") is not None:
    local_plot_path = cfg.output_dir / "energy_summary.png"
    local_plot_path.write_bytes(results["energy_plot_png"])


def main(argv: Sequence[str] | None = None) -> None:
  parser = build_arg_parser()
  args = parser.parse_args(argv)

  yaml_cfg = load_yaml_config(Path(args.config)) if args.config else {}

  # Determine proposal method and random_init early to know which fields are required.
  proposal_method = (
    getattr(args, "proposal_method", None)
    or yaml_cfg.get("proposal_method", "profam")
  )
  random_init = (
    getattr(args, "random_init", None)
    or yaml_cfg.get("random_init", False)
  )
  required_fields = ["energy_config"]
  if not random_init:
    required_fields.append("initial_fasta")
  if proposal_method == "profam":
    required_fields.append("profam_checkpoint_dir")
  for required in required_fields:
    if not (required in yaml_cfg and yaml_cfg[required]) and getattr(args, required) is None:
      parser.error(
        f"--{required} must be provided either in the YAML config or as a CLI flag."
      )

  cfg = merge_config(yaml_cfg, args)

  # If requested, run the entire pipeline inside a Modal app instead of locally.
  if cfg.run_on_modal:
    try:
      import modal  # type: ignore
      from run_profam_bagel_modal_app import app as modal_app  # type: ignore
      from run_profam_bagel_modal_app import results_vol  # type: ignore
      from run_profam_bagel_modal_app import run_pipeline_modal  # type: ignore
    except ImportError as e:  # pragma: no cover - runtime error path
      raise ImportError(
        "run_on_modal is set to True, but the Modal app entrypoint "
        "'run_profam_bagel_modal_app.run_pipeline_modal' could not be imported. "
        "Ensure that the 'modal' package is installed and that the Modal app "
        "script is available on PYTHONPATH."
      ) from e

    # Convert Path objects to strings for safe serialization over the wire.
    cfg_dict: Dict[str, Any] = dict(cfg.__dict__)
    for key in ("initial_fasta", "profam_checkpoint_dir", "energy_config", "output_dir"):
      if key in cfg_dict:
        cfg_dict[key] = str(cfg_dict[key])

    # Pass the local repo root so the Modal function can remap absolute paths
    # to the /workspace mount point inside the container.
    cfg_dict["_local_repo_root"] = str(ROOT_DIR)

    # Derive a unique run ID from the output directory name.  Each run
    # gets its own namespace within the shared Modal Volume, so parallel
    # runs with different configs don't interfere with each other.
    run_id = cfg.output_dir.name
    cfg_dict["_run_id"] = run_id

    # Clean the local output directory so stale files from a previous run
    # don't get mixed in with the current run's results.
    if cfg.output_dir.exists():
      import shutil
      shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale data for *this run* from the Modal Volume before starting
    # the poller, so it doesn't download results from a previous run with
    # the same output directory.  Other runs' namespaces are left untouched.
    try:
      for entry in results_vol.listdir(f"/{run_id}"):
        results_vol.remove_file(entry.path, recursive=True)
      results_vol.commit()
    except Exception:
      pass  # Namespace may not exist yet

    # Background thread that polls the Modal Volume for new checkpoint files
    # written by the remote function, and saves them to the local output dir.
    import threading

    _poll_stop = threading.Event()
    _poll_last_manifest_len = [0]  # mutable counter shared with poller thread

    def _sync_volume_to_local() -> None:
      """Download every file listed in the Volume manifest to the local
      output directory.  Called by the poller thread and after the remote
      function completes.  Reads from the run-specific namespace on the
      Volume (``/<run_id>/...``)."""
      try:
        manifest_data = b"".join(results_vol.read_file(f"{run_id}/_manifest.json"))
        manifest = json.loads(manifest_data.decode())
      except Exception:
        return  # manifest not yet written

      if len(manifest) <= _poll_last_manifest_len[0]:
        return  # nothing new

      n_ok = 0
      n_fail = 0
      for rel_path in manifest:
        try:
          file_data = b"".join(results_vol.read_file(f"{run_id}/{rel_path}"))
          local_path = cfg.output_dir / rel_path
          local_path.parent.mkdir(parents=True, exist_ok=True)
          local_path.write_bytes(file_data)
          n_ok += 1
        except Exception as exc:
          n_fail += 1
          print(f"[sync] Failed to download {rel_path}: {exc}")

      _poll_last_manifest_len[0] = len(manifest)

      # Print a short progress message.
      try:
        stats_path = cfg.output_dir / "cycle_stats.json"
        if stats_path.is_file():
          with stats_path.open("r") as f:
            stats = json.load(f)
          num_cycles = len(stats)
          print(
            f"[checkpoint] Synced {n_ok}/{len(manifest)} files "
            f"({num_cycles} cycles) from Modal Volume"
            + (f" ({n_fail} failed)" if n_fail else "")
          )
      except Exception:
        print(
          f"[checkpoint] Synced {n_ok}/{len(manifest)} files from Modal Volume"
          + (f" ({n_fail} failed)" if n_fail else "")
        )

    def _poll_volume() -> None:
      while not _poll_stop.is_set():
        _sync_volume_to_local()
        _poll_stop.wait(30)

    poller = threading.Thread(target=_poll_volume, daemon=True)

    with modal.enable_output():
      with modal_app.run():
        poller.start()
        try:
          run_pipeline_modal.remote(cfg_dict)
        finally:
          _poll_stop.set()
          poller.join(timeout=10)

    # Final sync: the remote function does a last on_checkpoint() before
    # returning, so the Volume has the complete output directory.
    _poll_last_manifest_len[0] = 0  # force full re-download
    _sync_volume_to_local()
    print("[final] Local output directory is up to date.")

    # Clean up this run's namespace from the Volume so it doesn't
    # accumulate stale data across runs.
    try:
      for entry in results_vol.listdir(f"/{run_id}"):
        results_vol.remove_file(entry.path, recursive=True)
      results_vol.commit()
    except Exception:
      pass
  else:
    run_pipeline(cfg)


if __name__ == "__main__":
  main()

