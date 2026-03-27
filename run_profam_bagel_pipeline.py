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
profam_temperature: 0.8             # optional
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

from Bio import Align  # type: ignore
from Bio.Align import substitution_matrices  # type: ignore

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
from bagel.utils import get_atomarray_in_residue_range  # type: ignore

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
# Thompson Sampling for conditioning sequence selection
# ---------------------------------------------------------------------------


def compute_sequence_identity(seq1: str, seq2: str) -> float:
  """Compute sequence identity using global pairwise alignment with BLOSUM62.

  Returns: identity as fraction [0, 1] = identical_positions / max(len(seq1), len(seq2))

  Using max sequence length as denominator penalizes length differences
  (insertions/deletions) and gives a conservative identity measure.
  """
  if not seq1 or not seq2:
    return 0.0

  aligner = Align.PairwiseAligner()
  aligner.mode = "global"
  aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
  aligner.open_gap_score = -10
  aligner.extend_gap_score = -0.5

  alignments = aligner.align(seq1, seq2)
  if not alignments:
    return 0.0

  # Take best alignment
  best_alignment = alignments[0]
  aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]

  # Count identical positions (excluding gaps)
  identical = sum(
    1 for a, b in zip(aligned_seq1, aligned_seq2)
    if a == b and a != '-' and b != '-'
  )

  # Normalize by max original sequence length
  max_len = max(len(seq1), len(seq2))
  return identical / max_len if max_len > 0 else 0.0


@dataclass
class ThompsonArm:
  arm_id: int
  sequence: str
  name: str
  alpha: float              # Beta posterior α
  beta_param: float         # Beta posterior β
  ipsae_raw: float          # Raw ipSAE at creation
  reward: float             # clamp(-ipsae_raw, 0, 1)
  parent_arm_id: int | None
  created_at_cycle: int
  times_selected: int = 0
  total_reward_credited: float = 0.0


class ThompsonSampler:
  """Manages a pool of arms for Thompson sampling over conditioning sequences."""

  def __init__(
    self,
    m_samples: int = 1,
    exploit_bias: float = 1.0,
    rng: np.random.Generator | None = None,
  ):
    self.m_samples = max(1, m_samples)
    self.exploit_bias = max(1.0, exploit_bias)
    self.rng = rng if rng is not None else np.random.default_rng()
    self.arms: Dict[int, ThompsonArm] = {}
    self._next_arm_id = 0

  @staticmethod
  def _ipsae_to_reward(ipsae_raw: float) -> float:
    """Convert raw ipSAE (negative = better) to reward in [0, 1]."""
    return float(np.clip(-ipsae_raw, 0.0, 1.0))

  def add_arm(
    self,
    sequence: str,
    name: str,
    ipsae_raw: float,
    parent_arm_id: int | None,
    cycle: int,
  ) -> ThompsonArm:
    """Register a new arm with Beta(1 + r, 2 - r) prior from its own ipSAE."""
    reward = self._ipsae_to_reward(ipsae_raw)
    arm = ThompsonArm(
      arm_id=self._next_arm_id,
      sequence=sequence,
      name=name,
      alpha=1.0 + reward,
      beta_param=2.0 - reward,
      ipsae_raw=ipsae_raw,
      reward=reward,
      parent_arm_id=parent_arm_id,
      created_at_cycle=cycle,
    )
    self.arms[arm.arm_id] = arm
    self._next_arm_id += 1
    return arm

  def select_arm(self) -> ThompsonArm:
    """Thompson sampling: sample θ ~ Beta(α*b, β*b) per arm, pick max.

    When exploit_bias > 1, both α and β are scaled up before sampling,
    which concentrates the Beta distribution around its mean — making
    selection more greedy (exploitative).  exploit_bias=1.0 is standard
    Thompson sampling.
    """
    if not self.arms:
      raise RuntimeError("No arms registered in ThompsonSampler.")
    best_arm = None
    best_theta = -1.0
    b = self.exploit_bias
    for arm in self.arms.values():
      # Sample m times and take the max (max-seeking variant).
      thetas = self.rng.beta(arm.alpha * b, arm.beta_param * b, size=self.m_samples)
      theta = float(np.max(thetas))
      if theta > best_theta:
        best_theta = theta
        best_arm = arm
    assert best_arm is not None
    best_arm.times_selected += 1
    return best_arm

  def update_arm(self, arm_id: int, progeny_ipsae: float) -> None:
    """Update the chosen arm's posterior with the progeny's reward."""
    arm = self.arms[arm_id]
    reward = self._ipsae_to_reward(progeny_ipsae)
    arm.alpha += reward
    arm.beta_param += (1.0 - reward)
    arm.total_reward_credited += reward

  def get_state_dict(self) -> List[Dict[str, Any]]:
    """Serialize all arm states for JSON logging."""
    result = []
    for arm in self.arms.values():
      result.append({
        "arm_id": arm.arm_id,
        "sequence": arm.sequence,
        "name": arm.name,
        "alpha": arm.alpha,
        "beta_param": arm.beta_param,
        "ipsae_raw": arm.ipsae_raw,
        "reward": arm.reward,
        "parent_arm_id": arm.parent_arm_id,
        "created_at_cycle": arm.created_at_cycle,
        "times_selected": arm.times_selected,
        "total_reward_credited": arm.total_reward_credited,
      })
    return result

  def decay_posteriors(self, discount: float) -> None:
    """Apply exponential decay to all arm posteriors.

    Shrinks α and β toward the prior (1, 1) by the discount factor,
    making the sampler more responsive to recent observations.
    """
    if discount >= 1.0:
      return
    for arm in self.arms.values():
      arm.alpha = 1.0 + discount * (arm.alpha - 1.0)
      arm.beta_param = 1.0 + discount * (arm.beta_param - 1.0)

  def prune_to_top_k_diverse(
    self,
    k: int,
    max_identity: float = 0.95,
  ) -> Dict[str, Any]:
    """Keep top-K arms ensuring diversity via sequence identity threshold.

    Uses greedy selection: iterates through arms sorted by posterior mean
    (expected reward = alpha / (alpha + beta)), keeping an arm only if its
    sequence identity to all already-retained arms is below max_identity.

    Args:
      k: Maximum number of arms to retain.
      max_identity: Maximum allowed sequence identity between any two retained
                    arms. Arms more similar than this are considered redundant.
                    Default 0.95 (95% identity) allows closely related variants
                    while filtering near-duplicates.

    Returns:
      Dict with pruning statistics: arms_before, arms_after, retained_ids,
      pruned_ids, and pairwise_identities of retained arms.
    """
    arms_before = len(self.arms)
    if arms_before <= k:
      return {
        "arms_before": arms_before,
        "arms_after": arms_before,
        "retained_ids": list(self.arms.keys()),
        "pruned_ids": [],
        "pairwise_identities": [],
      }

    # Sort all arms by posterior mean (expected reward), highest first
    # Posterior mean = alpha / (alpha + beta) from Beta distribution
    sorted_arms = sorted(
      self.arms.values(),
      key=lambda a: a.alpha / (a.alpha + a.beta_param),
      reverse=True,  # higher expected reward = better
    )

    retained: List[ThompsonArm] = []
    for arm in sorted_arms:
      if len(retained) >= k:
        break

      # Check diversity: must be < max_identity to all retained arms
      is_diverse = all(
        compute_sequence_identity(arm.sequence, r.sequence) < max_identity
        for r in retained
      )
      if is_diverse:
        retained.append(arm)

    # Compute pairwise identities for logging
    pairwise_identities = []
    for i, arm_i in enumerate(retained):
      for j, arm_j in enumerate(retained):
        if i < j:
          identity = compute_sequence_identity(arm_i.sequence, arm_j.sequence)
          pairwise_identities.append({
            "arm_i": arm_i.arm_id,
            "arm_j": arm_j.arm_id,
            "identity": identity,
          })

    # Update arms dict to only contain retained arms
    retained_ids = {a.arm_id for a in retained}
    pruned_ids = [aid for aid in self.arms.keys() if aid not in retained_ids]
    self.arms = {aid: arm for aid, arm in self.arms.items() if aid in retained_ids}

    return {
      "arms_before": arms_before,
      "arms_after": len(self.arms),
      "retained_ids": list(retained_ids),
      "pruned_ids": pruned_ids,
      "pairwise_identities": pairwise_identities,
    }


class TemperatureBandit:
  """Thompson sampler over a discrete set of temperature bins."""

  def __init__(
    self,
    bins: List[float],
    exploit_bias: float = 1.0,
    rng: np.random.Generator | None = None,
  ):
    self.bins = sorted(bins)
    self.exploit_bias = max(1.0, exploit_bias)
    self.rng = rng if rng is not None else np.random.default_rng()
    # Each bin gets a Beta(1, 1) = uniform prior.
    self.alphas = {t: 1.0 for t in self.bins}
    self.betas = {t: 1.0 for t in self.bins}
    self.times_selected: Dict[float, int] = {t: 0 for t in self.bins}
    self.total_reward: Dict[float, float] = {t: 0.0 for t in self.bins}

  def select(self) -> float:
    """Thompson-sample a temperature bin."""
    best_temp = self.bins[0]
    best_theta = -1.0
    b = self.exploit_bias
    for t in self.bins:
      theta = float(self.rng.beta(self.alphas[t] * b, self.betas[t] * b))
      if theta > best_theta:
        best_theta = theta
        best_temp = t
    self.times_selected[best_temp] += 1
    return best_temp

  def update(self, temperature: float, reward: float) -> None:
    """Update the chosen bin's posterior with the observed reward."""
    self.alphas[temperature] += reward
    self.betas[temperature] += (1.0 - reward)
    self.total_reward[temperature] += reward

  def decay(self, discount: float) -> None:
    """Apply exponential decay toward the prior."""
    if discount >= 1.0:
      return
    for t in self.bins:
      self.alphas[t] = 1.0 + discount * (self.alphas[t] - 1.0)
      self.betas[t] = 1.0 + discount * (self.betas[t] - 1.0)

  def get_state_dict(self) -> List[Dict[str, Any]]:
    """Serialize state for JSON logging."""
    result = []
    for t in self.bins:
      a, b = self.alphas[t], self.betas[t]
      result.append({
        "temperature": t,
        "alpha": a,
        "beta_param": b,
        "expected_reward": a / (a + b),
        "times_selected": self.times_selected[t],
        "total_reward": self.total_reward[t],
      })
    return result


class ProposalBandit:
  """Thompson sampler over proposal methods (profam vs random_mutation)."""

  METHODS = ["profam", "random_mutation"]

  def __init__(
    self,
    exploit_bias: float = 1.0,
    rng: np.random.Generator | None = None,
  ):
    self.exploit_bias = max(1.0, exploit_bias)
    self.rng = rng if rng is not None else np.random.default_rng()
    self.alphas = {m: 1.0 for m in self.METHODS}
    self.betas = {m: 1.0 for m in self.METHODS}
    self.times_selected: Dict[str, int] = {m: 0 for m in self.METHODS}
    self.total_reward: Dict[str, float] = {m: 0.0 for m in self.METHODS}

  def select(self) -> str:
    """Thompson-sample a proposal method."""
    best_method = self.METHODS[0]
    best_theta = -1.0
    b = self.exploit_bias
    for m in self.METHODS:
      theta = float(self.rng.beta(self.alphas[m] * b, self.betas[m] * b))
      if theta > best_theta:
        best_theta = theta
        best_method = m
    self.times_selected[best_method] += 1
    return best_method

  def update(self, method: str, reward: float) -> None:
    """Update the chosen method's posterior with the observed reward."""
    self.alphas[method] += reward
    self.betas[method] += (1.0 - reward)
    self.total_reward[method] += reward

  def decay(self, discount: float) -> None:
    """Apply exponential decay toward the prior."""
    if discount >= 1.0:
      return
    for m in self.METHODS:
      self.alphas[m] = 1.0 + discount * (self.alphas[m] - 1.0)
      self.betas[m] = 1.0 + discount * (self.betas[m] - 1.0)

  def get_state_dict(self) -> List[Dict[str, Any]]:
    """Serialize state for JSON logging."""
    result = []
    for m in self.METHODS:
      a, b = self.alphas[m], self.betas[m]
      result.append({
        "method": m,
        "alpha": a,
        "beta_param": b,
        "expected_reward": a / (a + b),
        "times_selected": self.times_selected[m],
        "total_reward": self.total_reward[m],
      })
    return result


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
  thompson_reward_term: str = "ipSAE"     # energy term name for reward
  thompson_exploit_bias: float = 1.0      # >1 = more exploitation (concentrate posteriors)
  thompson_temperature_bins: List[float] | None = None  # e.g. [0.6, 0.8, 1.0, 1.3]; None = fixed temperature
  thompson_discount: float = 1.0          # per-cycle decay on posteriors (1.0 = no decay)
  thompson_proposal_bandit: bool = False   # True = Thompson bandit over proposal methods (profam vs random_mutation)
  thompson_max_arms: int = 0              # max arms to retain (0 = unlimited); prunes to top-K diverse arms
  thompson_max_identity: float = 0.95     # max sequence identity between retained arms (diversity threshold)
  deduplicate_sequences: bool = True       # skip folding for already-seen sequences, retry generation

  random_init: bool = False                # if True, generate a random initial sequence instead of reading from FASTA
  random_init_max_residues: int = 80       # max length of randomly generated initial sequence


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
    run_on_modal=bool(pick("run_on_modal", False)),
    enforce_template=bool(pick("enforce_template", True)),
    output_frequency=int(pick("output_frequency", 1)),
    sample_with_reinsertion=bool(pick("sample_with_reinsertion", True)),
    reinject_initial=bool(pick("reinject_initial", True)),
    n_memory=int(pick("n_memory", 0)),
    elitism=bool(pick("elitism", False)),
    accept_only_improvement=bool(pick("accept_only_improvement", False)),
    annealing_initial_temp=(
      None
      if pick("annealing_initial_temp", None) is None
      else float(pick("annealing_initial_temp"))
    ),
    annealing_decay=float(pick("annealing_decay", 0.95)),
    proposal_method=str(pick("proposal_method", "profam")),
    max_mutations=int(pick("max_mutations", 5)),
    freeze_prompt=bool(pick("freeze_prompt", False)),
    selection_strategy=str(pick("selection_strategy", "greedy")),
    thompson_m_samples=int(pick("thompson_m_samples", 1)),
    thompson_reward_term=str(pick("thompson_reward_term", "ipSAE")),
    thompson_exploit_bias=float(pick("thompson_exploit_bias", 1.0)),
    thompson_temperature_bins=(
      None
      if pick("thompson_temperature_bins", None) is None
      else [float(x) for x in pick("thompson_temperature_bins")]
    ),
    thompson_discount=float(pick("thompson_discount", 1.0)),
    thompson_proposal_bandit=bool(pick("thompson_proposal_bandit", False)),
    thompson_max_arms=int(pick("thompson_max_arms", 0)),
    thompson_max_identity=float(pick("thompson_max_identity", 0.95)),
    deduplicate_sequences=bool(pick("deduplicate_sequences", True)),
    random_init=bool(pick("random_init", False)),
    random_init_max_residues=int(pick("random_init_max_residues", 80)),
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
    "--thompson_reward_term",
    type=str,
    default=None,
    help=(
      "Energy term name to use as the reward signal for Thompson sampling. "
      "Default: 'ipSAE'. Must match a term name in the energy config."
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
    "--thompson_discount",
    type=float,
    default=None,
    help=(
      "Per-cycle exponential decay on all Thompson posteriors. Shrinks α and β "
      "toward the prior each cycle, making the sampler more responsive to recent "
      "observations. 1.0 = no decay (default), 0.95 = moderate forgetting."
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
  fixed_positions: Dict[int, str] | None = None,
) -> Tuple[List[str], List[str]]:
  """
  Generate sequences using ProFam's Python API.

  This calls the sampler directly (no subprocess), reusing the model loaded
  once by ``load_profam_model()``.
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

  # Convert fixed_positions to token IDs if provided.
  fixed_token_positions = None
  if fixed_positions is not None:
    fixed_token_positions = {
      int(k): model.tokenizer.convert_tokens_to_ids(v)
      for k, v in fixed_positions.items()
    }

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
    # fixed_positions is only available in newer ProFam versions.
    import inspect
    if "fixed_positions" in inspect.signature(sampler_obj.sample_seqs).parameters:
      sample_kwargs["fixed_positions"] = fixed_token_positions
    elif fixed_token_positions:
      print(
        "WARNING: fixed_positions requested but not supported by this "
        "ProFam version — ignoring constrained residues."
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

  return list(accessions), list(sequences)


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


def build_folding_oracle(energy_cfg: Dict[str, Any], force_modal: bool = False) -> FoldingOracle:
  folding_cfg = energy_cfg.get("folding_oracle", {}) or {}
  oracle_type = folding_cfg.get("type", "ESMFold")
  kwargs = folding_cfg.get("kwargs", {}) or {}
  if not isinstance(kwargs, dict):
    raise ValueError("folding_oracle.kwargs must be a dictionary.")
  if oracle_type == "ESMFold":
    if force_modal:
      # Override to make sure the folding oracle itself uses Modal,
      # regardless of what is specified in the energy config.
      kwargs["use_modal"] = True
    return ESMFold(**kwargs)
  elif oracle_type == "AlphaFast":
    # AlphaFast always runs on Modal — force_modal is implicit
    return AlphaFast(**kwargs)
  elif oracle_type == "Boltz":
    # Boltz runs locally via CLI subprocess
    return Boltz(**kwargs)
  else:
    raise ValueError(
      f"Unsupported folding oracle type: {oracle_type!r}. "
      f"Use 'ESMFold', 'AlphaFast', or 'Boltz'."
    )


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
  oracle: FoldingOracle,
  chain: bg.Chain,
  target_chains: Dict[int, "bg.Chain"] | None = None,
) -> List[bg.energies.EnergyTerm]:
  """
  Instantiate BAGEL EnergyTerm objects for a given chain, based on the
  energy YAML configuration.
  """
  energies_spec = energy_cfg.get("energies", [])
  if not isinstance(energies_spec, list):
    raise ValueError("energy_config must contain an 'energies' list.")

  terms: List[bg.energies.EnergyTerm] = []
  for entry_idx, entry in enumerate(energies_spec):
    if not isinstance(entry, dict):
      raise ValueError(f"Each energy entry must be a dict, got {entry!r}")
    etype = entry.get("type")
    if not isinstance(etype, str):
      raise ValueError(f"Energy 'type' must be a string, got {etype!r}")

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

    term = energy_cls(oracle=oracle, **kwargs)
    terms.append(term)

  return terms


def extract_fixed_residues_from_energy_config(
  energy_cfg: Dict[str, Any],
) -> Dict[int, str] | None:
  """
  Scan the energy config for TemplateMatchEnergy entries and extract a mapping
  from generated-sequence position to amino-acid character.

  The ``residues`` list in the energy config gives 0-based positions that
  refer to the same locations in both the generated sequence and the template
  chain.  For each position ``p`` in ``residues``, the amino acid at
  position ``p`` of the template chain is the identity that should be forced
  during generation.

  Returns ``None`` if no TemplateMatchEnergy is found.
  """
  energies_spec = energy_cfg.get("energies", [])
  fixed: Dict[int, str] = {}

  for entry in energies_spec:
    if not isinstance(entry, dict):
      continue
    if entry.get("type") != "TemplateMatchEnergy":
      continue

    kwargs = dict(entry.get("kwargs", {}) or {})

    # Load template structure (supports local file or PDB code download).
    atoms, _ = _load_structure_from_spec(kwargs)
    if atoms is None:
      raise ValueError(
        "TemplateMatchEnergy requires 'template_structure_path' or "
        "'pdb_code' in kwargs."
      )

    # Full amino-acid sequence of the template chain.
    template_seq = sequence_from_atomarray(atoms)

    # The `residues` specification gives 0-based positions that refer to the
    # same locations in both the generated sequence and the template chain.
    # Normalise compact range strings (e.g. "0-43") to integer lists.
    raw_spec = _normalise_residue_spec(kwargs.get("residues", {}))
    if isinstance(raw_spec, dict):
      residue_indices = raw_spec.get("GEN", [])
    else:
      residue_indices = raw_spec
    # "all" means every position in the template chain.
    if isinstance(residue_indices, str) and residue_indices.lower() == "all":
      residue_indices = list(range(len(template_seq)))
    if not isinstance(residue_indices, list):
      raise ValueError(
        "'residues' must be a dict with a 'GEN' key mapping to integer indices, "
        "a range string, or 'all'."
      )

    max_idx = max(int(i) for i in residue_indices) if residue_indices else -1
    if max_idx >= len(template_seq):
      raise ValueError(
        f"Residue index {max_idx} is out of bounds for template chain with "
        f"{len(template_seq)} residues."
      )

    for gen_pos in residue_indices:
      fixed[int(gen_pos)] = template_seq[int(gen_pos)]

  return fixed if fixed else None


def evaluate_sequences_with_bagel(
  sequences: Sequence[str],
  energy_cfg: Dict[str, Any],
  folding_oracle: FoldingOracle,
  cycle_index: int,
  cycle_dir: Path,
  enforce_template: bool = True,
) -> Tuple[List[float], List[Dict[str, Any]], List[Any]]:
  """
  For each sequence, build a single-chain BAGEL System, run the required
  oracles (folding, embedding, …), compute total weighted energy from
  configured energy terms, and — when a folding oracle was invoked — save
  the predicted structures for later export.

  The folding oracle is only called when at least one energy term requires
  a FoldingOracle; otherwise no structure prediction is performed and no
  CIF files are written.

  Returns:
    - energies: list of total energies, one per input sequence
    - details: list of dicts including per-sequence energy breakdown
    - folding_results: list of FoldingResult objects (entries are None for
      sequences where no folding oracle was needed)
  """
  from bagel.oracles import OraclesResultDict  # type: ignore
  from bagel.oracles.folding import FoldingOracle  # type: ignore

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
  # Phase 2: Batch-predict structures for all sequences at once.
  #
  # If the folding oracle supports predict_batch (e.g. Boltz) and every
  # sequence needs it, we call it once instead of N times.  This loads
  # the model once and processes all inputs sequentially, saving ~30 s
  # of model-load overhead per additional sequence.
  # ------------------------------------------------------------------

  # Check which sequences need a folding oracle.
  needs_folding = [
    any(isinstance(o, FoldingOracle) for o in d["oracles_needed"])
    for d in per_seq_data
  ]
  batch_folding_results: List[Any] = [None] * len(sequences)

  if any(needs_folding) and hasattr(folding_oracle, "predict_batch"):
    # Collect the chain-lists that need folding.
    batch_indices = [i for i, nf in enumerate(needs_folding) if nf]
    batch_chains = [per_seq_data[i]["all_chains"] for i in batch_indices]
    # Fold each individually so a single failure doesn't crash the batch.
    for i, chains in zip(batch_indices, batch_chains):
      try:
        batch_folding_results[i] = folding_oracle.predict(chains=chains)
      except Exception as exc:
        print(f"  Sequence {i}: folding failed ({exc}); will assign inf energy")
        batch_folding_results[i] = None

  # ------------------------------------------------------------------
  # Phase 3: Compute energies using the (pre-computed) oracle results.
  # ------------------------------------------------------------------
  energies: List[float] = []
  details: List[Dict[str, Any]] = []
  folding_results: List[Any] = []

  for idx, seq in enumerate(sequences):
    d = per_seq_data[idx]
    energy_terms = d["energy_terms"]

    oracles_result = OraclesResultDict()
    folding_result = None

    folding_failed = False
    for oracle in d["oracles_needed"]:
      if isinstance(oracle, FoldingOracle):
        if batch_folding_results[idx] is not None:
          # Use the pre-computed batch result.
          result = batch_folding_results[idx]
        else:
          # Folding failed for this sequence — skip energy computation.
          print(f"  Sequence {idx}: batch_folding_results is None, marking as folding_failed")
          folding_failed = True
          break
      else:
        # Non-folding oracle — call sequentially.
        try:
          result = oracle.predict(chains=d["all_chains"])
        except Exception as exc:
          print(f"  Sequence {idx}: non-folding oracle {type(oracle).__name__} failed: {exc}")
          folding_failed = True
          break
      oracles_result[oracle] = result
      if isinstance(oracle, FoldingOracle):
        folding_result = result

    if folding_failed:
      energies.append(float("inf"))
      folding_results.append(None)
      details.append({"total_energy": float("inf"), "error": "folding_failed"})
      continue

    total_energy = 0.0
    per_term: Dict[str, float] = {}
    for term in energy_terms:
      try:
        unweighted, weighted = term.compute(oracles_result=oracles_result)
        per_term[term.name] = float(unweighted)
        total_energy += float(weighted)
      except (ValueError, Exception) as exc:
        if not enforce_template:
          print(
            f"  Sequence {idx}: caught {type(exc).__name__} in {term.name}, "
            f"assigning inf energy ({exc})"
          )
          per_term[term.name] = float("inf")
          total_energy = float("inf")
          break
        else:
          raise

    energies.append(total_energy)
    folding_results.append(folding_result)
    details.append(
      {
        "index": idx,
        "sequence": seq,
        "energy": total_energy,
        "energy_terms": per_term,
      }
    )

  # Save structures for sequences where the folding oracle was called.
  if any(fr is not None for fr in folding_results):
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
# Sampling / statistics / plotting
# ---------------------------------------------------------------------------


def _pairwise_identity(seq_a: str, seq_b: str) -> float:
  """Return the fraction of identical residues from a global alignment.

  Uses Biopython's ``PairwiseAligner`` (Needleman–Wunsch) so that
  insertions and deletions are handled correctly — a single indel no longer
  shifts all downstream positions and artificially tanks the score.

  The identity is defined as::

      identity = matched_columns / alignment_length

  where *alignment_length* includes gap columns on either side.
  """
  from Bio.Align import PairwiseAligner

  if not seq_a or not seq_b:
    return 0.0

  aligner = PairwiseAligner()
  aligner.mode = "global"
  # Standard NW scoring: match +1, mismatch 0, gap open/extend penalties.
  aligner.match_score = 1.0
  aligner.mismatch_score = 0.0
  aligner.open_gap_score = -0.5
  aligner.extend_gap_score = -0.1

  # We only need the top alignment.
  alignment = aligner.align(seq_a, seq_b)[0]
  aln_a, aln_b = alignment[0], alignment[1]
  aln_len = len(aln_a)
  if aln_len == 0:
    return 0.0
  matches = sum(a == b and a != "-" for a, b in zip(aln_a, aln_b))
  return matches / max(len(seq_a), len(seq_b))


def compute_avg_sequence_similarity(
  generated_seqs: Sequence[str],
  initial_seqs: Sequence[str],
) -> float:
  """
  Compute the average sequence similarity between generated sequences and
  the initial input sequences.

  For each generated sequence the similarity to each initial sequence is
  computed via global pairwise alignment (Needleman–Wunsch) so that
  insertions and deletions are properly accounted for.  The best (maximum)
  similarity across all initial sequences is kept for each generated
  sequence, and the mean of those best-match values is returned.
  """
  if not generated_seqs or not initial_seqs:
    return 0.0

  best_sims: List[float] = []
  for gen_seq in generated_seqs:
    best = 0.0
    for init_seq in initial_seqs:
      best = max(best, _pairwise_identity(gen_seq, init_seq))
    best_sims.append(best)

  return float(np.mean(best_sims))


def extract_reward_term(
  details: Sequence[Dict[str, Any]],
  term_name: str,
) -> List[float]:
  """Extract per-sequence values for a named energy term from evaluation details.

  Returns a list parallel to ``details``. If a sequence has no valid value
  for the term (e.g. folding failure), ``float('inf')`` is returned.
  """
  values: List[float] = []
  for d in details:
    if d and isinstance(d, dict) and "energy_terms" in d:
      val = d["energy_terms"].get(term_name, float("inf"))
      values.append(float(val))
    else:
      values.append(float("inf"))
  return values


def softmax_from_energies(
  energies: Sequence[float],
  temperature: float = 1.0,
) -> np.ndarray:
  """
  Convert energies into sampling probabilities via a softmax over -energy / T.
  Lower energies correspond to higher probabilities.
  """
  if temperature <= 0:
    raise ValueError("softmax_temperature must be > 0.")
  arr = np.asarray(energies, dtype=float)
  if arr.size == 0:
    raise ValueError("Cannot compute softmax for empty energy list.")

  # Mask out inf energies (e.g. from template mismatch with enforce_template=False).
  # These get zero probability; finite energies are softmaxed normally.
  finite_mask = np.isfinite(arr)
  if not np.any(finite_mask):
    # All inf — fall back to uniform (caller should ideally retry).
    return np.ones(arr.size) / arr.size

  logits = np.full_like(arr, -np.inf)
  logits[finite_mask] = -arr[finite_mask] / float(temperature)
  logits -= np.max(logits)  # numerical stability
  exp = np.exp(logits)
  probs = exp / np.sum(exp)
  return probs


def sample_subset_indices(
  num_items: int,
  probs: np.ndarray,
  f_inject: float,
  rng: np.random.Generator,
  replace: bool = True,
  energies: Sequence[float] | None = None,
  subset_size: int | None = None,
) -> np.ndarray:
  """
  Sample a subset of indices of size floor(f_inject * num_items) according
  to probabilities ``probs``.

  Parameters
  ----------
  replace : bool
      If True (default), sample with replacement (a sequence may appear
      multiple times).  If False, sample without replacement; when the
      requested subset size exceeds the number of candidates with
      non-zero probability, fall back to returning only the index of
      the best (lowest-energy) candidate.
  energies : sequence of float, optional
      Required when ``replace=False`` so that the best candidate can be
      identified as a fallback.
  subset_size : int, optional
      If provided, overrides the ``floor(f_inject * num_items)`` calculation
      for the number of items to sample.  Used when the pool contains
      sequences from previous cycles (n_memory > 0) but the injection
      count should still be based on the current generation size.
  """
  if num_items <= 0:
    raise ValueError("num_items must be > 0.")
  k = subset_size if subset_size is not None else int(math.floor(f_inject * num_items))
  if k <= 0:
    k = 1

  if replace:
    idx = rng.choice(num_items, size=k, replace=True, p=probs)
    return np.asarray(idx, dtype=int)

  # Without replacement: the pool of drawable items is limited to those
  # with non-zero probability.
  num_nonzero = int(np.sum(probs > 0))
  if k <= num_nonzero:
    idx = rng.choice(num_items, size=k, replace=False, p=probs)
    return np.asarray(idx, dtype=int)

  # Cannot draw k unique items — fall back to the single best candidate.
  if energies is None:
    raise ValueError(
      "energies must be provided when sample_with_reinsertion=False "
      "so the best candidate can be identified as a fallback."
    )
  best_idx = int(np.argmin(energies))
  print(
    f"  Cannot sample {k} unique candidates (only {num_nonzero} have "
    f"non-zero probability); falling back to best candidate (index {best_idx})."
  )
  return np.asarray([best_idx], dtype=int)


def update_cycle_log(
  log_path: Path,
  cycle_index: int,
  selected_indices: np.ndarray,
  energies: Sequence[float],
  sequence_details: Sequence[Dict[str, Any]],
  avg_similarity: float | None = None,
  avg_similarity_to_prompt: float | None = None,
  global_ids: Sequence[int] | None = None,
  pool_ids: Sequence[int] | None = None,
  pool_energies: Sequence[float] | None = None,
  pool_names: Sequence[str] | None = None,
  pool_seqs: Sequence[str] | None = None,
  swap_accepted: bool | None = None,
  swap_reason: str | None = None,
  elite_energy: float | None = None,
  elite_cycle: int | None = None,
  annealing_temp: float | None = None,
  thompson_state: Dict[str, Any] | None = None,
  thompson_selected_arm_id: int | None = None,
  thompson_progeny_reward: float | None = None,
  proposal_method: str | None = None,
) -> None:
  """
  Append / update a JSON log keyed by cycle index.

  Parameters
  ----------
  global_ids : list of int, optional
      Global unique IDs for the current cycle's generated sequences.
      When provided, each entry in ``sequence_details`` and
      ``best_sequence`` gains an ``"id"`` field.
  pool_ids : list of int, optional
      Global IDs for the full selection pool (memory + current cycle).
      When provided, ``selected_indices`` index into this pool and the
      logged ``selected_ids`` use these global IDs.
  pool_energies : list of float, optional
      Energies for the full selection pool (parallel to ``pool_ids``).
  pool_names : list of str, optional
      Names for the full selection pool (parallel to ``pool_ids``).
  pool_seqs : list of str, optional
      Sequences for the full selection pool (parallel to ``pool_ids``).
  """
  if log_path.is_file():
    with log_path.open("r") as f:
      log_data = json.load(f)
  else:
    log_data = {}

  # Replace inf/nan with large sentinel for JSON compatibility.
  def _json_safe(v: float) -> float:
    return 1e30 if (math.isinf(v) or math.isnan(v)) else v

  # When a pool is active, selected_indices index into pool_energies.
  # Otherwise they index into this cycle's energies.
  if pool_energies is not None:
    sel_energies = [_json_safe(float(pool_energies[int(i)])) for i in selected_indices]
  else:
    sel_energies = [_json_safe(float(energies[int(i)])) for i in selected_indices]
  avg_energy = _json_safe(float(np.mean(sel_energies)))
  min_energy = _json_safe(float(np.min(sel_energies)))

  # Build selected_sequences entries.  When a pool is active, some
  # selected indices may point to sequences from past cycles for which
  # we don't have full details.  In that case, build a minimal entry.
  selected_sequences: List[Dict[str, Any]] = []
  if pool_ids is not None:
    # Pool is active — selected_indices are pool-relative.
    pool_offset = len(pool_ids) - len(energies)  # where current cycle starts
    for i in selected_indices:
      idx = int(i)
      gid = pool_ids[idx]
      if idx >= pool_offset:
        # Current cycle sequence — full details available.
        local_idx = idx - pool_offset
        entry = dict(sequence_details[local_idx])
        entry["energy"] = _json_safe(float(entry.get("energy", 0.0)))
        if "energy_terms" in entry:
          entry["energy_terms"] = {
            k: _json_safe(float(v)) for k, v in entry["energy_terms"].items()
          }
      else:
        # From memory — full details not available, but we have the
        # energy, name, and sequence from the pool.
        entry: Dict[str, Any] = {"energy": _json_safe(float(pool_energies[idx]))}  # type: ignore[index]
        if pool_seqs is not None:
          entry["sequence"] = pool_seqs[idx]
      entry["id"] = gid
      selected_sequences.append(entry)
  else:
    for i in selected_indices:
      entry = dict(sequence_details[int(i)])
      entry["energy"] = _json_safe(float(entry.get("energy", 0.0)))
      if "energy_terms" in entry:
        entry["energy_terms"] = {
          k: _json_safe(float(v)) for k, v in entry["energy_terms"].items()
        }
      if global_ids is not None:
        entry["id"] = global_ids[int(i)]
      selected_sequences.append(entry)

  # Stats over ALL generated sequences in the current cycle (not the pool).
  all_energies_safe = [_json_safe(float(e)) for e in energies]
  all_avg_energy = _json_safe(float(np.mean(all_energies_safe)))
  all_min_energy = _json_safe(float(np.min(all_energies_safe)))

  # Best sequence: the one with the lowest energy among all generated this cycle.
  best_idx = int(np.argmin(all_energies_safe))
  best_entry = dict(sequence_details[best_idx])
  best_entry["energy"] = _json_safe(float(best_entry.get("energy", 0.0)))
  if "energy_terms" in best_entry:
    best_entry["energy_terms"] = {
      k: _json_safe(float(v)) for k, v in best_entry["energy_terms"].items()
    }
  if global_ids is not None:
    best_entry["id"] = global_ids[best_idx]

  # Build selected_ids: global IDs of the selected sequences.
  if pool_ids is not None:
    selected_id_list = [pool_ids[int(i)] for i in selected_indices]
  elif global_ids is not None:
    selected_id_list = [global_ids[int(i)] for i in selected_indices]
  else:
    selected_id_list = [int(i) for i in selected_indices]

  cycle_entry: Dict[str, Any] = {
    "cycle": cycle_index,
    "num_generated": len(energies),
    "all_avg_energy": all_avg_energy,
    "all_min_energy": all_min_energy,
    "best_sequence": best_entry,
    "num_selected": len(selected_indices),
    "selected_avg_energy": avg_energy,
    "selected_min_energy": min_energy,
    "selected_ids": selected_id_list,
    "selected_sequences": selected_sequences,
  }
  if pool_ids is not None:
    cycle_entry["pool_size"] = len(pool_ids)
  if avg_similarity is not None:
    cycle_entry["all_avg_similarity"] = avg_similarity
  if avg_similarity_to_prompt is not None:
    cycle_entry["all_avg_similarity_to_prompt"] = avg_similarity_to_prompt
  if swap_accepted is not None:
    cycle_entry["swap_accepted"] = swap_accepted
  if swap_reason is not None:
    cycle_entry["swap_reason"] = swap_reason
  if elite_energy is not None:
    cycle_entry["global_elite"] = {
      "energy": _json_safe(elite_energy),
      "from_cycle": elite_cycle,
    }
  if annealing_temp is not None:
    cycle_entry["annealing_temp"] = annealing_temp
  if proposal_method is not None:
    cycle_entry["proposal_method"] = proposal_method
  if thompson_selected_arm_id is not None:
    cycle_entry["thompson_selected_arm_id"] = thompson_selected_arm_id
  if thompson_progeny_reward is not None:
    cycle_entry["thompson_progeny_reward"] = thompson_progeny_reward
  if thompson_state is not None:
    cycle_entry["thompson_num_arms"] = thompson_state.get("num_arms", 0)

  log_data[str(cycle_index)] = cycle_entry

  with log_path.open("w") as f:
    json.dump(log_data, f, indent=2)


def save_selected_structures(
  cycle_index: int,
  selected_indices: np.ndarray,
  folding_results: Sequence[Any],
  output_dir: Path,
  pool_offset: int = 0,
) -> None:
  """
  Save CIF structures for the selected subset into `sequences_cycle_<cycle>`.

  If no structures were calculated during this cycle (i.e. the folding
  oracle was not invoked because no energy term required it), this
  function is a no-op.

  Parameters
  ----------
  pool_offset : int
      When n_memory > 0, ``selected_indices`` index into the combined pool
      (memory + current cycle).  ``pool_offset`` is the index at which the
      current cycle's sequences start in the pool.  Only current-cycle
      sequences have folding results available; memory sequences are skipped.
  """
  if not any(fr is not None for fr in folding_results):
    return

  seq_dir = output_dir / f"sequences_cycle_{cycle_index}"
  seq_dir.mkdir(parents=True, exist_ok=True)

  for out_idx, seq_idx in enumerate(selected_indices):
    idx = int(seq_idx)
    if idx < pool_offset:
      # Memory sequence — no folding result available for this cycle.
      continue
    fr = folding_results[idx - pool_offset]
    if fr is not None:
      cif_path = seq_dir / f"sequence_{out_idx:04d}.cif"
      fr.to_cif(cif_path)
      if hasattr(fr, "save_attributes"):
        fr.save_attributes(seq_dir / f"sequence_{out_idx:04d}")


def append_cycle_csv(
  csv_path: Path,
  cycle_index: int,
  names: Sequence[str],
  sequences: Sequence[str],
  energies: Sequence[float],
  details: Sequence[Dict[str, Any]],
  folding_results: Sequence[Any],
  initial_seqs: Sequence[str],
  prompt_seqs: Sequence[str] | None = None,
  proposal_method: str | None = None,
) -> None:
  """Append one row per generated sequence to the all_sequences.csv file."""
  import csv

  write_header = not csv_path.is_file()

  # Collect all energy term keys from the first non-error detail entry.
  energy_term_keys: List[str] = []
  for d in details:
    if isinstance(d, dict) and "energy_terms" in d:
      energy_term_keys = sorted(d["energy_terms"].keys())
      break

  fieldnames = [
    "cycle", "proposal_method", "name", "sequence", "length", "total_energy",
  ] + energy_term_keys + [
    "ptm", "mean_plddt", "iptm", "similarity_to_initial", "similarity_to_prompt",
  ]

  with csv_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()

    for idx in range(len(sequences)):
      seq = sequences[idx]
      # Extract per-term energies.
      term_values: Dict[str, float] = {}
      if idx < len(details) and isinstance(details[idx], dict):
        term_values = details[idx].get("energy_terms", {})

      # Extract structural metrics from folding result.
      ptm_val = ""
      plddt_val = ""
      iptm_val = ""
      fr = folding_results[idx] if idx < len(folding_results) else None
      if fr is not None:
        try:
          ptm_val = float(fr.ptm[0])
        except Exception:
          pass
        try:
          plddt_val = float(fr.local_plddt[0].mean())
        except Exception:
          pass
        try:
          iptm_val = float(fr.chain_pair_iptm[0, 1])
        except Exception:
          pass

      # Similarity to initial (original prompt).
      sim = max(_pairwise_identity(seq, init_s) for init_s in initial_seqs) if initial_seqs else ""
      # Similarity to current prompt.
      sim_prompt = max(_pairwise_identity(seq, ps) for ps in prompt_seqs) if prompt_seqs else ""

      row: Dict[str, Any] = {
        "cycle": cycle_index,
        "proposal_method": proposal_method or "",
        "name": names[idx] if idx < len(names) else "",
        "sequence": seq,
        "length": len(seq),
        "total_energy": energies[idx] if idx < len(energies) else "",
      }
      for k in energy_term_keys:
        row[k] = term_values.get(k, "")
      row["ptm"] = ptm_val
      row["mean_plddt"] = plddt_val
      row["iptm"] = iptm_val
      row["similarity_to_initial"] = sim
      row["similarity_to_prompt"] = sim_prompt

      writer.writerow(row)


def make_energy_summary_plot(
  log_path: Path,
  output_dir: Path,
) -> None:
  """
  Produce a PNG plot of average and minimum energy as a function of cycle index.
  """
  try:
    import matplotlib.pyplot as plt  # type: ignore
    plt.style.use("dark_background")
  except ImportError:
    # Plotting is optional; skip gracefully if matplotlib is not available.
    print("matplotlib not available, skipping summary plot.")
    return

  if not log_path.is_file():
    print(f"No cycle log found at {log_path}, skipping summary plot.")
    return

  with log_path.open("r") as f:
    log_data = json.load(f)

  if not log_data:
    print("Cycle log is empty, nothing to plot.")
    return

  cycles = sorted(int(k) for k in log_data.keys())
  avg = [log_data[str(c)].get("all_avg_energy", log_data[str(c)].get("avg_energy")) for c in cycles]
  min_e = [log_data[str(c)].get("all_min_energy", log_data[str(c)].get("min_energy")) for c in cycles]

  # Compute cumulative minimum (global best seen so far).
  cum_min = []
  running_min = float("inf")
  for e in min_e:
    if e is not None and e < running_min:
      running_min = e
    cum_min.append(running_min if running_min != float("inf") else e)

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(cycles, avg, marker="o", color="#00bfff", label="Average energy (all generated)")
  ax.plot(cycles, min_e, marker="s", color="#00e676", label="Minimum energy (all generated)")
  ax.plot(cycles, cum_min, linestyle="--", color="#ff6b6b", linewidth=1.5, label="Global best (cumulative min)")

  # Mark rejected swaps with X markers if data available.
  rejected_cycles = []
  rejected_energies = []
  for c in cycles:
    entry = log_data[str(c)]
    if entry.get("swap_accepted") is False:
      rejected_cycles.append(c)
      rejected_energies.append(entry.get("all_min_energy", entry.get("min_energy")))
  if rejected_cycles:
    ax.scatter(rejected_cycles, rejected_energies, marker="x", color="#ff6b6b",
               s=100, zorder=5, label="Swap rejected")

  ax.set_xlabel("Cycle")
  ax.set_ylabel("Energy")
  ax.set_title("Energy & similarity trajectory over cycles")
  ax.grid(True, linestyle="--", alpha=0.4)

  # Plot sequence similarity on a twin y-axis if available.
  sim_original = [log_data[str(c)].get("all_avg_similarity") for c in cycles]
  sim_prompt = [log_data[str(c)].get("all_avg_similarity_to_prompt") for c in cycles]
  has_sim_original = any(s is not None for s in sim_original)
  has_sim_prompt = any(s is not None for s in sim_prompt)
  if has_sim_original or has_sim_prompt:
    ax2 = ax.twinx()
    if has_sim_original:
      ax2.plot(
        cycles,
        [s if s is not None else float("nan") for s in sim_original],
        marker="^",
        linestyle="--",
        color="#ffab40",
        label="Similarity to original",
      )
    if has_sim_prompt:
      ax2.plot(
        cycles,
        [s if s is not None else float("nan") for s in sim_prompt],
        marker="v",
        linestyle="--",
        color="#e040fb",
        label="Similarity to prompt",
      )
    ax2.set_ylabel("Sequence similarity")
    ax2.set_ylim(0, 1.05)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize="small")
  else:
    ax.legend()

  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "energy_summary.png"
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, facecolor="black", edgecolor="none")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------


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


def run_pipeline(
  cfg: PipelineConfig,
  force_modal_folding: bool = False,
  checkpoint_callback: Any = None,
) -> None:
  cfg.output_dir.mkdir(parents=True, exist_ok=True)

  # Save all pipeline settings to the output folder for reproducibility.
  config_snapshot = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
  config_path = cfg.output_dir / "pipeline_config.json"
  with config_path.open("w") as f:
    json.dump(config_snapshot, f, indent=2)
  print(f"Config saved to {config_path}")

  if cfg.freeze_prompt and not cfg.reinject_initial:
    print("Warning: freeze_prompt requires reinject_initial=True. Setting it.")
    cfg.reinject_initial = True

  # Load ProFam model once and reuse across all cycles (skip for non-ProFam proposals).
  # When the proposal bandit is enabled, always load ProFam since it may be selected.
  profam_model, profam_device = (None, "cpu")
  if cfg.proposal_method == "profam" or cfg.thompson_proposal_bandit:
    profam_model, profam_device = load_profam_model(cfg)
    if cfg.thompson_proposal_bandit:
      print("Proposal bandit enabled: ProFam model loaded (may also use random_mutation)")
  else:
    print(f"Proposal method: {cfg.proposal_method} (max_mutations={cfg.max_mutations})")

  # Load energy configuration & instantiate BAGEL folding oracle.
  energy_cfg = load_energy_config(cfg.energy_config)
  folding_oracle = build_folding_oracle(energy_cfg, force_modal=force_modal_folding)

  # Validate Thompson reward term exists in the energy config.
  if cfg.selection_strategy == "thompson":
    energy_term_types = [e["type"] for e in energy_cfg.get("energies", [])]
    # The term name in evaluation details is derived from the BAGEL class name
    # minus the "Energy" suffix (e.g. ipSAEEnergy → ipSAE). Check that at least
    # one term type contains the reward term name.
    reward_term_found = any(
      cfg.thompson_reward_term in t for t in energy_term_types
    )
    if not reward_term_found:
      raise ValueError(
        f"thompson_reward_term='{cfg.thompson_reward_term}' not found among "
        f"energy term types {energy_term_types}. The reward term name must "
        f"match a key in the 'energy_terms' dict produced by evaluation."
      )

  rng = np.random.default_rng(cfg.random_seed)
  cycle_log_path = cfg.output_dir / "cycle_stats.json"

  # Read initial sequences S1 from FASTA, or generate randomly.
  if cfg.random_init:
    _AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    rand_len = rng.integers(cfg.random_init_max_residues // 2, cfg.random_init_max_residues + 1)
    rand_seq = "".join(rng.choice(list(_AMINO_ACIDS), size=rand_len))
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

  # Extract fixed residue positions from the energy config when enforce_template
  # is enabled. These positions will be forced during ProFam generation.
  fixed_residues: Dict[int, str] | None = None
  if cfg.enforce_template:
    fixed_residues = extract_fixed_residues_from_energy_config(energy_cfg)
    if fixed_residues:
      print(
        f"enforce_template=True: forcing {len(fixed_residues)} residue positions "
        f"during generation."
      )
    else:
      print("enforce_template=True but no TemplateMatchEnergy found; no positions forced.")

  injected_names: List[str] = []
  injected_seqs: List[str] = []

  # Global unique ID counter and memory buffer for n_memory support.
  # Each generated sequence receives a monotonically increasing ID across
  # all cycles (e.g. cycle 1 → IDs 0-9, cycle 2 → IDs 10-19, etc.).
  next_global_id = 0
  # Memory buffer: list of (ids, names, seqs, energies) tuples, one per past cycle.
  # At most cfg.n_memory entries are kept.
  memory_buffer: List[tuple] = []

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
      exploit_bias=cfg.thompson_exploit_bias,
      rng=rng,
    )
    print(f"  Proposal bandit: methods={ProposalBandit.METHODS}")

  # Sequence deduplication cache: maps sequence string → (energy, details_dict).
  # Populated during evaluation; checked before folding to skip duplicates.
  seen_sequences: Dict[str, tuple] = {}

  # Evaluate initial seed sequence(s) to establish a baseline energy.
  # This ensures cycle 1 must improve over the seed to be accepted.
  if cfg.elitism or cfg.accept_only_improvement or cfg.selection_strategy == "thompson":
    print("=== Evaluating initial seed sequence(s) ===")
    seed_energies, seed_details, seed_folding_results = evaluate_sequences_with_bagel(
      sequences=base_initial_seqs,
      energy_cfg=energy_cfg,
      folding_oracle=folding_oracle,
      cycle_index=0,
      cycle_dir=cfg.output_dir / "cycle_000_seed",
      enforce_template=cfg.enforce_template,
    )
    seed_best_idx = int(np.argmin(seed_energies))
    seed_best_energy = float(seed_energies[seed_best_idx])
    print(f"  Seed sequence best energy: {seed_best_energy:.4f}")
    if seed_details[seed_best_idx] and isinstance(seed_details[seed_best_idx], dict):
      terms = {k: v for k, v in seed_details[seed_best_idx].items() if k != "total_energy"}
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
          {k: v for k, v in seed_details[seed_best_idx].items() if k != "total_energy"}
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

    # Register seed sequences as initial Thompson arms.
    if thompson_sampler is not None:
      print(f"  Thompson SEED REGISTRATION (m_samples={thompson_sampler.m_samples}):")
      seed_reward_values = extract_reward_term(seed_details, cfg.thompson_reward_term)
      n_seed_registered = 0
      for i, (name, seq) in enumerate(zip(base_initial_names, base_initial_seqs)):
        ipsae_val = seed_reward_values[i]
        if math.isfinite(ipsae_val):
          arm = thompson_sampler.add_arm(
            sequence=seq, name=name, ipsae_raw=ipsae_val,
            parent_arm_id=None, cycle=0,
          )
          n_seed_registered += 1
          print(f"    arm {arm.arm_id}: {name}, "
                f"{cfg.thompson_reward_term}={ipsae_val:.4f}, "
                f"reward={arm.reward:.4f}, "
                f"α={arm.alpha:.4f}, β={arm.beta_param:.4f}, "
                f"seq_len={len(seq)}")
        else:
          print(f"    SKIPPED {name}: {cfg.thompson_reward_term}=inf (folding failure)")
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
              f"({best_seed_arm.name}, ipSAE={best_seed_arm.ipsae_raw:.4f})")
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
    max_dedup_retries = 10  # cap retries to avoid infinite loops
    for attempt in range(1, max_generation_attempts + max_dedup_retries + 1):
      if cycle_proposal_method == "random_mutation":
        gen_names, gen_seqs = run_random_mutation_generation(
          seed_sequences=all_seqs,
          num_samples=cfg.profam_num_samples,
          max_mutations=cfg.max_mutations,
          rng=rng,
        )
      else:
        gen_names, gen_seqs = run_profam_generation(
          cfg=cfg,
          input_fasta=profam_input_fasta,
          cycle_dir=cycle_dir,
          model=profam_model,
          device=profam_device,
          fixed_positions=fixed_residues,
        )
      if len(gen_seqs) != cfg.profam_num_samples:
        print(
          f"Warning: expected {cfg.profam_num_samples} generated sequences, "
          f"got {len(gen_seqs)}."
        )

      # Deduplication: check if all generated sequences have been seen before.
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
          # Exhausted dedup retries — use cached results for the duplicates.
          print(f"  Dedup: exhausted {max_dedup_retries} retries, "
                f"using cached results for duplicate sequences")
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
          folding_oracle=folding_oracle,
          cycle_index=cycle,
          cycle_dir=cycle_dir,
          enforce_template=cfg.enforce_template,
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
          folding_oracle=folding_oracle,
          cycle_index=cycle,
          cycle_dir=cycle_dir,
          enforce_template=cfg.enforce_template,
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
    cycle_best_idx = int(np.argmin(energies))
    cycle_best_energy = float(energies[cycle_best_idx])
    if cycle_best_energy < elite_energy:
      elite_energy = cycle_best_energy
      elite_seq = gen_seqs[cycle_best_idx]
      elite_name = gen_names[cycle_best_idx]
      elite_cycle = cycle
      print(f"  New global elite: energy={elite_energy:.4f} (cycle {elite_cycle})")

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

    # --- Thompson sampling branch ---
    thompson_cycle_state: Dict[str, Any] | None = None
    thompson_selected_arm_id: int | None = None
    thompson_progeny_reward: float | None = None

    if cfg.selection_strategy == "thompson" and thompson_sampler is not None:
      # Extract per-sequence reward term values.
      reward_values = extract_reward_term(details, cfg.thompson_reward_term)
      n_finite = sum(1 for v in reward_values if math.isfinite(v))
      n_inf = len(reward_values) - n_finite
      print(f"  Thompson [{cfg.thompson_reward_term}]: {n_finite} finite, {n_inf} inf "
            f"out of {len(reward_values)} progeny")
      if n_finite > 0:
        finite_vals = [v for v in reward_values if math.isfinite(v)]
        print(f"    progeny {cfg.thompson_reward_term} range: "
              f"[{min(finite_vals):.4f}, {max(finite_vals):.4f}], "
              f"mean={np.mean(finite_vals):.4f}")

      # Update the parent arm's posterior with the best progeny's reward.
      # The parent arm is the one that was selected last cycle to condition on.
      # On cycle 1 the parent is the best seed arm (set during init).
      if hasattr(thompson_sampler, '_last_selected_arm_id'):
        parent_id = thompson_sampler._last_selected_arm_id
        parent_arm = thompson_sampler.arms[parent_id]
        print(f"  Thompson POSTERIOR UPDATE for parent arm {parent_id} ({parent_arm.name}):")
        print(f"    before: α={parent_arm.alpha:.4f}, β={parent_arm.beta_param:.4f}, "
              f"E[θ]={parent_arm.alpha/(parent_arm.alpha+parent_arm.beta_param):.4f}")
        # Find the best (most negative) finite reward value among progeny.
        finite_rewards = [(i, v) for i, v in enumerate(reward_values) if math.isfinite(v)]
        if finite_rewards:
          best_progeny_idx, best_progeny_ipsae = min(finite_rewards, key=lambda x: x[1])
          thompson_sampler.update_arm(parent_id, best_progeny_ipsae)
          thompson_progeny_reward = ThompsonSampler._ipsae_to_reward(best_progeny_ipsae)
          print(f"    best progeny: idx={best_progeny_idx}, "
                f"{cfg.thompson_reward_term}={best_progeny_ipsae:.4f}, "
                f"reward={thompson_progeny_reward:.4f}")
          print(f"    after:  α={parent_arm.alpha:.4f}, β={parent_arm.beta_param:.4f}, "
                f"E[θ]={parent_arm.alpha/(parent_arm.alpha+parent_arm.beta_param):.4f}")
        else:
          print(f"    no finite progeny — posterior unchanged")

      # Update temperature bandit with the same reward signal.
      if temp_bandit is not None and cycle_temperature is not None:
        # Use the best finite progeny reward for the temperature update too.
        temp_reward = thompson_progeny_reward if thompson_progeny_reward is not None else 0.0
        temp_bandit.update(cycle_temperature, temp_reward)
        print(f"  Temperature bandit UPDATE: T={cycle_temperature}, "
              f"reward={temp_reward:.4f}")

      # Update proposal bandit with the same reward signal.
      if proposal_bandit is not None:
        prop_reward = thompson_progeny_reward if thompson_progeny_reward is not None else 0.0
        proposal_bandit.update(cycle_proposal_method, prop_reward)
        print(f"  Proposal bandit UPDATE: method={cycle_proposal_method}, "
              f"reward={prop_reward:.4f}")

    # --- Proposal bandit update when running with greedy selection ---
    if proposal_bandit is not None and cfg.selection_strategy != "thompson":
      reward_values_pb = extract_reward_term(details, cfg.thompson_reward_term)
      n_finite = sum(1 for v in reward_values_pb if math.isfinite(v))
      n_inf = len(reward_values_pb) - n_finite
      finite_rewards_pb = [v for v in reward_values_pb if math.isfinite(v)]
      if finite_rewards_pb:
        best_ipsae_pb = min(finite_rewards_pb)
        prop_reward = float(np.clip(-best_ipsae_pb, 0.0, 1.0))
      else:
        prop_reward = 0.0
      proposal_bandit.update(cycle_proposal_method, prop_reward)
      if cfg.thompson_discount < 1.0:
        proposal_bandit.decay(cfg.thompson_discount)
      print(f"  Proposal bandit UPDATE: method={cycle_proposal_method}, "
            f"reward={prop_reward:.4f}")

      # Register all progeny with finite ipSAE as new arms.
      parent_arm_id_for_progeny = getattr(thompson_sampler, '_last_selected_arm_id', None)
      n_registered = 0
      for i, (name, seq, ipsae_val) in enumerate(zip(gen_names, gen_seqs, reward_values_pb)):
        if math.isfinite(ipsae_val):
          arm = thompson_sampler.add_arm(
            sequence=seq, name=name, ipsae_raw=ipsae_val,
            parent_arm_id=parent_arm_id_for_progeny, cycle=cycle,
          )
          n_registered += 1
      print(f"  Thompson: registered {n_registered} new arms "
            f"(total arms: {len(thompson_sampler.arms)})")

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

      # Select next arm via Thompson sampling — log the top candidates.
      b = thompson_sampler.exploit_bias
      print(f"  Thompson ARM SELECTION (m_samples={thompson_sampler.m_samples}, "
            f"exploit_bias={b:.1f}):")
      # Sample θ for all arms and log top-5 for transparency.
      # arm_thetas: List[tuple] = []
      # for arm in thompson_sampler.arms.values():
      #   thetas = rng.beta(arm.alpha * b, arm.beta_param * b, size=thompson_sampler.m_samples)
      #   theta = float(np.max(thetas))
      #   arm_thetas.append((arm, theta))
      # arm_thetas.sort(key=lambda x: x[1], reverse=True)
      # for rank, (arm, theta) in enumerate(arm_thetas[:5]):
      #   marker = " <<<" if rank == 0 else ""
      #   print(f"    rank {rank+1}: arm {arm.arm_id} ({arm.name}), "
      #         f"θ={theta:.4f}, α={arm.alpha:.2f}, β={arm.beta_param:.2f}, "
      #         f"E[θ]={arm.alpha/(arm.alpha+arm.beta_param):.4f}, "
      #         f"selected {arm.times_selected}x, "
      #         f"created cycle {arm.created_at_cycle}{marker}")
      # if len(arm_thetas) > 5:
      #   print(f"    ... ({len(arm_thetas) - 5} more arms)")

      # Greedy selection: pick the arm with the best (lowest) ipSAE.
      arms_by_ipsae = sorted(
        thompson_sampler.arms.values(),
        key=lambda a: a.ipsae_raw,  # lowest (most negative) is best
      )
      next_arm = arms_by_ipsae[0]
      thompson_selected_arm_id = next_arm.arm_id
      thompson_sampler._last_selected_arm_id = next_arm.arm_id  # type: ignore[attr-defined]
      next_arm.times_selected += 1  # track selection count
      print(f"  GREEDY SELECTED → arm {next_arm.arm_id} ({next_arm.name})")
      print(f"    ipSAE_raw={next_arm.ipsae_raw:.4f} (BEST of {len(thompson_sampler.arms)} arms)")
      print(f"    times_selected={next_arm.times_selected}, "
            f"parent_arm={next_arm.parent_arm_id}, "
            f"created_cycle={next_arm.created_at_cycle}")
      print(f"  PROMPT SEQUENCE: {next_arm.sequence}")

      # Set injection to the single selected arm's sequence.
      injected_names = [next_arm.name]
      injected_seqs = [next_arm.sequence]

      # For logging compatibility, create minimal selected_indices pointing at
      # the best generated sequence this cycle.
      selected_indices = np.array([int(np.argmin(energies))])

      # Build detailed thompson cycle state for JSON logging.
      # Include top-10 arms by expected reward for quick inspection.
      arms_by_expected = sorted(
        thompson_sampler.arms.values(),
        key=lambda a: a.alpha / (a.alpha + a.beta_param),
        reverse=True,
      )
      top_arms_summary = []
      for arm in arms_by_expected[:10]:
        top_arms_summary.append({
          "arm_id": arm.arm_id,
          "name": arm.name,
          "alpha": arm.alpha,
          "beta_param": arm.beta_param,
          "expected_reward": arm.alpha / (arm.alpha + arm.beta_param),
          "ipsae_raw": arm.ipsae_raw,
          "times_selected": arm.times_selected,
          "total_reward_credited": arm.total_reward_credited,
          "parent_arm_id": arm.parent_arm_id,
          "created_at_cycle": arm.created_at_cycle,
        })
      thompson_cycle_state: Dict[str, Any] = {
        "num_arms": len(thompson_sampler.arms),
        "selected_arm_id": thompson_selected_arm_id,
        "selected_arm_name": next_arm.name,
        "selected_arm_expected_reward": next_arm.alpha / (next_arm.alpha + next_arm.beta_param),
        "selected_arm_alpha": next_arm.alpha,
        "selected_arm_beta": next_arm.beta_param,
        "selected_arm_times_selected": next_arm.times_selected,
        "progeny_finite_count": n_finite,
        "progeny_inf_count": n_inf,
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
        json.dump(thompson_sampler.get_state_dict(), f, indent=2)

      # Save per-cycle thompson decision log (append, one entry per cycle).
      decision_log_path = cfg.output_dir / "thompson_decisions.jsonl"
      decision_entry: Dict[str, Any] = {
        "cycle": cycle,
        "selected_arm_id": thompson_selected_arm_id,
        "selected_arm_name": next_arm.name,
        "selected_arm_alpha": next_arm.alpha,
        "selected_arm_beta": next_arm.beta_param,
        "selected_arm_expected_reward": next_arm.alpha / (next_arm.alpha + next_arm.beta_param),
        "progeny_reward": thompson_progeny_reward,
        "progeny_finite_count": n_finite,
        "total_arms": len(thompson_sampler.arms),
        "progeny_ipsae_values": [
          round(v, 6) if math.isfinite(v) else None for v in reward_values_pb
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
      swap_reason = "thompson"
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
      )
      save_selected_structures(
        cycle_index=cycle,
        selected_indices=selected_indices,
        folding_results=folding_results,
        output_dir=cfg.output_dir,
        pool_offset=pool_offset,
      )

      # Apply discount decay to all posteriors (sequence arms + temperature + proposal).
      if cfg.thompson_discount < 1.0:
        thompson_sampler.decay_posteriors(cfg.thompson_discount)
        if temp_bandit is not None:
          temp_bandit.decay(cfg.thompson_discount)
        if proposal_bandit is not None:
          proposal_bandit.decay(cfg.thompson_discount)
        print(f"  Thompson discount: decayed posteriors by {cfg.thompson_discount}")

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
      )
      save_selected_structures(
        cycle_index=cycle,
        selected_indices=selected_indices,
        folding_results=folding_results,
        output_dir=cfg.output_dir,
        pool_offset=pool_offset,
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

