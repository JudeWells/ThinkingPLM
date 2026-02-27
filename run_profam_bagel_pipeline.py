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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from biotite.structure.io.pdb import PDBFile  # type: ignore
from biotite.structure.io.pdbx import CIFFile, get_structure  # type: ignore

try:
  import yaml  # type: ignore
except ImportError as e:  # pragma: no cover - import-time check
  raise ImportError(
    "PyYAML is required to run the pipeline. Install it with `pip install pyyaml`."
  ) from e

# BAGEL — installed via: pip install git+https://github.com/softnanolab/bagel.git
import bagel as bg  # type: ignore
from bagel.oracles import ESMFold  # type: ignore
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
# Config dataclass & CLI/YAML handling
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
  initial_fasta: Path
  profam_checkpoint_dir: Path
  profam_sampler: str = "single"
  profam_num_samples: int = 10
  profam_max_tokens: int = 8192
  profam_max_generated_length: int | None = None
  profam_temperature: float | None = None
  profam_top_p: float | None = 0.95

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

  cfg = PipelineConfig(
    initial_fasta=_to_path(pick("initial_fasta")),
    profam_checkpoint_dir=_to_path(pick("profam_checkpoint_dir")),
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
  )

  if not 0.0 < cfg.f_inject <= 1.0:
    raise ValueError(f"f_inject must be in (0, 1], got {cfg.f_inject}")
  if cfg.profam_num_samples <= 0:
    raise ValueError("profam_num_samples (N_output) must be > 0.")
  if not cfg.initial_fasta.is_file():
    raise FileNotFoundError(f"Initial FASTA not found: {cfg.initial_fasta}")
  if not cfg.profam_checkpoint_dir.is_dir():
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


def build_folding_oracle(energy_cfg: Dict[str, Any], force_modal: bool = False) -> ESMFold:
  folding_cfg = energy_cfg.get("folding_oracle", {}) or {}
  oracle_type = folding_cfg.get("type", "ESMFold")
  if oracle_type != "ESMFold":
    raise ValueError(
      f"Only ESMFold folding oracle is currently supported, got type={oracle_type!r}"
    )
  kwargs = folding_cfg.get("kwargs", {}) or {}
  if not isinstance(kwargs, dict):
    raise ValueError("folding_oracle.kwargs must be a dictionary.")
  if force_modal:
    # Override to make sure the folding oracle itself uses Modal,
    # regardless of what is specified in the energy config.
    kwargs["use_modal"] = True
  return ESMFold(**kwargs)


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
  oracle: ESMFold,
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
  folding_oracle: ESMFold,
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

  energies: List[float] = []
  details: List[Dict[str, Any]] = []
  folding_results: List[Any] = []

  for idx, seq in enumerate(sequences):
    residues = [
      bg.Residue(name=aa, chain_ID="GEN", index=i, mutable=False)
      for i, aa in enumerate(seq)
    ]
    chain = bg.Chain(residues=residues)

    # Build target chains for energy entries that have a "target" key.
    # Each target uses the chain_ID derived from the residues dict.
    target_chains_map: Dict[int, bg.Chain] = {}
    seen_targets: Dict[Tuple[str, str], bg.Chain] = {}  # de-duplicate
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

    # Determine which unique oracles are needed by the energy terms and
    # call each one.  This mirrors the approach in State.energy and ensures
    # the folding oracle is only invoked when at least one energy term
    # actually requires a predicted structure.
    #
    # When target chains are present, the oracle receives all chains so
    # that it folds the multi-chain complex.
    all_chains = [chain] + list({id(c): c for c in target_chains_map.values()}.values())
    oracles_needed = list(set(term.oracle for term in energy_terms))
    oracles_result = OraclesResultDict()
    folding_result = None
    for oracle in oracles_needed:
      result = oracle.predict(chains=all_chains)
      oracles_result[oracle] = result
      if isinstance(oracle, FoldingOracle):
        folding_result = result

    total_energy = 0.0
    per_term: Dict[str, float] = {}
    for term in energy_terms:
      try:
        unweighted, weighted = term.compute(oracles_result=oracles_result)
        per_term[term.name] = float(unweighted)
        total_energy += float(weighted)
      except ValueError as exc:
        if not enforce_template:
          print(
            f"  Sequence {idx}: caught ValueError in {term.name}, "
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
  # When no energy term required a FoldingOracle, folding_results will
  # contain only None entries and no structures need to be written.
  if any(fr is not None for fr in folding_results):
    structures_dir = cycle_dir / f"sequences_cycle_all_{cycle_index}"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for idx, fr in enumerate(folding_results):
      if fr is not None:
        cif_path = structures_dir / f"sequence_{idx:04d}.cif"
        fr.to_cif(cif_path)

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
  return matches / aln_len


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
  global_ids: Sequence[int] | None = None,
  pool_ids: Sequence[int] | None = None,
  pool_energies: Sequence[float] | None = None,
  pool_names: Sequence[str] | None = None,
  pool_seqs: Sequence[str] | None = None,
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


def make_energy_summary_plot(
  log_path: Path,
  output_dir: Path,
) -> None:
  """
  Produce a PNG plot of average and minimum energy as a function of cycle index.
  """
  try:
    import matplotlib.pyplot as plt  # type: ignore
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

  fig, ax = plt.subplots(figsize=(7, 4))
  ax.plot(cycles, avg, marker="o", label="Average energy (all generated)")
  ax.plot(cycles, min_e, marker="s", label="Minimum energy (all generated)")
  ax.set_xlabel("Cycle")
  ax.set_ylabel("Energy")
  ax.set_title("Energy & similarity trajectory over cycles")
  ax.grid(True, linestyle="--", alpha=0.4)

  # Plot sequence similarity on a twin y-axis if available.
  sim = [log_data[str(c)].get("all_avg_similarity") for c in cycles]
  if any(s is not None for s in sim):
    ax2 = ax.twinx()
    ax2.plot(
      cycles,
      [s if s is not None else float("nan") for s in sim],
      marker="^",
      linestyle="--",
      color="green",
      label="Avg sequence similarity",
    )
    ax2.set_ylabel("Sequence similarity")
    ax2.set_ylim(0, 1.05)
    # Merge legends from both axes.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize="small")
  else:
    ax.legend()

  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "energy_summary.png"
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
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

  # Load ProFam model once and reuse across all cycles.
  profam_model, profam_device = load_profam_model(cfg)

  # Load energy configuration & instantiate BAGEL folding oracle.
  energy_cfg = load_energy_config(cfg.energy_config)
  folding_oracle = build_folding_oracle(energy_cfg, force_modal=force_modal_folding)

  rng = np.random.default_rng(cfg.random_seed)
  cycle_log_path = cfg.output_dir / "cycle_stats.json"

  # Read initial sequences S1 from FASTA
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

    # Step 1 & 2: generation + evaluation, with retry logic for enforce_template=False
    max_generation_attempts = 5
    for attempt in range(1, max_generation_attempts + 1):
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
        continue
      break
    else:
      print(
        f"Warning: all {max_generation_attempts} generation attempts produced "
        f"only inf-energy sequences in cycle {cycle}. Proceeding with last batch."
      )

    # Assign global unique IDs to this cycle's sequences.
    gen_ids = list(range(next_global_id, next_global_id + len(gen_seqs)))
    next_global_id += len(gen_seqs)

    # Compute average sequence similarity to the initial sequences.
    avg_sim = compute_avg_sequence_similarity(gen_seqs, base_initial_seqs)
    print(f"  Avg sequence similarity to initial: {avg_sim:.4f}")

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

    # Save statistics and selected sequences
    update_cycle_log(
      log_path=cycle_log_path,
      cycle_index=cycle,
      selected_indices=selected_indices,
      energies=energies,
      sequence_details=details,
      avg_similarity=avg_sim,
      global_ids=gen_ids,
      pool_ids=pool_ids if cfg.n_memory > 0 else None,
      pool_energies=pool_energies if cfg.n_memory > 0 else None,
      pool_names=pool_names if cfg.n_memory > 0 else None,
      pool_seqs=pool_seqs if cfg.n_memory > 0 else None,
    )
    save_selected_structures(
      cycle_index=cycle,
      selected_indices=selected_indices,
      folding_results=folding_results,
      output_dir=cfg.output_dir,
      pool_offset=pool_offset,
    )

    # Prepare injected sequences for next cycle (from the pool).
    injected_names = [pool_names[int(i)] for i in selected_indices]
    injected_seqs = [pool_seqs[int(i)] for i in selected_indices]

    # Update memory buffer with current cycle's data.
    if cfg.n_memory > 0:
      memory_buffer.append((list(gen_ids), list(gen_names), list(gen_seqs), list(energies)))
      if len(memory_buffer) > cfg.n_memory:
        memory_buffer.pop(0)

    # Periodic checkpoint: push intermediate results every output_frequency
    # cycles (and always at the final cycle).
    output_freq = max(1, cfg.output_frequency)
    if checkpoint_callback and (cycle % output_freq == 0 or cycle == cfg.max_cycles):
      make_energy_summary_plot(log_path=cycle_log_path, output_dir=cfg.output_dir)
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

  # Ensure required values are present either in YAML or CLI.
  for required in ("initial_fasta", "profam_checkpoint_dir", "energy_config"):
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

