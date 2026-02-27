"""
Modal app entrypoint for running the full ProFam + BAGEL pipeline remotely.

This app is intended to be invoked indirectly from `run_profam_bagel_pipeline.py`
when `run_on_modal: true` is set in the YAML configuration, or directly via:

    modal run run_profam_bagel_modal_app.py::run_pipeline_modal --config path/to/config.yaml

The remote function receives a serialisable configuration dictionary derived
from `PipelineConfig` and then calls `run_pipeline(...)` inside the Modal
container, forcing BAGEL's ESMFold folding oracle to use Modal as well
(`use_modal = True`), overriding any value set in the energy config file.

Intermediate results (cycle stats, energy plots, and structure CIF files) are
synced to a persistent Modal Volume every `output_frequency` cycles.  The local
caller polls the Volume in a background thread and downloads all new files.
Because the Modal function is a regular (non-generator) function, it is
automatically reschedulable on worker preemption.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import modal


REPO_ROOT = Path(__file__).resolve().parent

# Try to mount the user's local BAGEL folding models (if present) so that
# Modal can reuse already-downloaded weights instead of fetching them again.
LOCAL_MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path.home() / ".cache/bagel/models"))

# Modal Secret containing the HuggingFace token (HF_TOKEN) for downloading
# the gated ProFam checkpoint.  Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
hf_secret = modal.Secret.from_name("huggingface-secret")

# Image definition: install BAGEL and ProFam from their GitHub repositories,
# plus any additional dependencies, then add the local pipeline files.
#
# IMPORTANT: Modal requires add_local_dir / add_local_file to be the LAST
# steps in the image build (no run_commands after them).  So all run_commands
# (pip installs, ProFam clone, checkpoint download) come first, then the
# local file overlay at the very end.
image = (
  modal.Image.debian_slim()
  # git is needed for pip install from GitHub URLs and the ProFam clone below.
  .apt_install("git")
  .pip_install(
    # BAGEL (biobagel) — includes biotite, boileroom, numpy, pandas, pydantic, matplotlib
    "biobagel[local] @ git+https://github.com/softnanolab/bagel.git",
    # Additional dependencies not pulled by the above
    "transformers>=4.49.0,<5.0.0",
    "tokenizers",
    "pytorch-lightning",
    "omegaconf",
    "rootutils",
    "datasets",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "biopython",
    "scipy",
    "scikit-learn",
    "numba",
    "modal",
    "pyyaml",
  )
  # ProFam's setup.py uses find_packages(), but several sub-packages
  # (src/sequence, src/evaluators, src/pipelines) are missing __init__.py
  # so a plain pip install skips them.  We clone the repo, add the missing
  # files, then install so that all sub-packages are included.
  .run_commands(
    "git clone --depth 1 https://github.com/alex-hh/profam.git /tmp/profam"
    " && touch /tmp/profam/src/sequence/__init__.py"
    "         /tmp/profam/src/evaluators/__init__.py"
    "         /tmp/profam/src/pipelines/__init__.py"
    " && pip install /tmp/profam"
    " && rm -rf /tmp/profam",
  )
  # Download the ProFam-1 checkpoint from HuggingFace Hub (gated repo,
  # requires HF_TOKEN).  Cached in the image layer — only downloads once.
  # Create the secret with:  modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
  .run_commands(
    "python -c \""
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('judewells/ProFam-1', "
    "local_dir='/workspace/model_checkpoints/profam-1', "
    "local_dir_use_symlinks=False)"
    "\"",
    secrets=[hf_secret],
  )
  # Add the local pipeline files (configs, scripts, energy YAMLs).
  # This MUST be the last step in the image build (Modal requirement).
  .add_local_dir(
    str(REPO_ROOT),
    remote_path="/workspace",
    ignore=[
      "profam/",
      "bagel/",
      "model_checkpoints",
      "outputs/",
      ".git/",
      "__pycache__/",
    ],
  )
)

# Optionally include cached BAGEL folding model weights so BAGEL doesn't
# re-download them.  Only mount if the directory exists AND is non-empty.
if LOCAL_MODEL_DIR.is_dir() and any(LOCAL_MODEL_DIR.iterdir()):
  image = image.add_local_dir(str(LOCAL_MODEL_DIR), remote_path="/models/bagel")

# Persistent volume for intermediate checkpoint files.  The local caller polls
# this volume for new results while the remote function is running.
results_vol = modal.Volume.from_name(
  "profam-bagel-checkpoints", create_if_missing=True
)

app = modal.App("profam-bagel-pipeline")

# Path inside the container where the volume is mounted.
_VOL_MOUNT = "/vol_results"


def _setup_remote_env() -> None:
  """Configure sys.path and working directory inside the Modal container."""
  import sys

  # The pipeline script itself lives under /workspace.
  sys.path.insert(0, "/workspace")
  os.chdir("/workspace")

  if Path("/models/bagel").exists():
    os.environ["MODEL_DIR"] = "/models/bagel"


def _reconstruct_config(cfg_dict: Dict[str, Any]):
  """
  Rebuild a PipelineConfig from the serialized dict, remapping local
  absolute paths to the Modal /workspace mount.
  """
  from run_profam_bagel_pipeline import PipelineConfig, _to_path

  data = dict(cfg_dict)
  local_repo_root = data.pop("_local_repo_root", None)
  data.pop("_run_id", None)
  for key in ("initial_fasta", "profam_checkpoint_dir", "energy_config", "output_dir"):
    if key in data and data[key] is not None:
      path_str = str(data[key])
      if local_repo_root and path_str.startswith(local_repo_root):
        path_str = "/workspace" + path_str[len(local_repo_root):]
      data[key] = _to_path(path_str)

  # Ensure we don't recursively try to launch Modal from inside Modal.
  data["run_on_modal"] = False

  return PipelineConfig(**data)


@app.function(
  image=image,
  gpu="A10G",
  timeout=60 * 60 * 24,
  volumes={_VOL_MOUNT: results_vol},
  retries=3,
)
def run_pipeline_modal(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
  """
  Remote Modal entrypoint.  Runs the full pipeline and syncs the entire
  output directory (cycle stats, energy plot, and structure CIF files) to
  a persistent Volume every ``output_frequency`` cycles.  The local caller
  polls the Volume in a background thread to download new files as they
  appear.

  Each run is namespaced within the shared Volume by a ``_run_id`` key
  (derived from the output directory name).  This allows multiple runs
  with different configs to execute in parallel without interfering.
  """
  import json

  _setup_remote_env()

  from run_profam_bagel_pipeline import run_pipeline

  cfg = _reconstruct_config(cfg_dict)

  # Each run is namespaced under its own subdirectory on the volume so
  # that parallel runs don't interfere with each other.
  run_id = cfg_dict.get("_run_id", "default")
  vol_root = Path(_VOL_MOUNT) / run_id

  # Clear stale checkpoint files from any previous run *with the same run_id*.
  import shutil
  if vol_root.exists():
    shutil.rmtree(vol_root)
  vol_root.mkdir(parents=True, exist_ok=True)
  results_vol.commit()

  def on_checkpoint(results: Dict[str, Any]) -> None:
    """Sync the entire output directory to the Volume.

    This copies every file produced so far (cycle_stats.json,
    energy_summary.png, and all structure CIF files) so the local
    poller can download them.  A ``_manifest.json`` listing all
    relative paths is written last to let the poller know exactly
    which files to fetch.
    """
    manifest: list[str] = []
    for dirpath, _, filenames in os.walk(str(cfg.output_dir)):
      for filename in filenames:
        src = Path(dirpath) / filename
        rel = str(src.relative_to(cfg.output_dir))
        dest = vol_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        manifest.append(rel)
    with open(vol_root / "_manifest.json", "w") as f:
      json.dump(manifest, f)
    results_vol.commit()

  # Force the folding oracle to use Modal (use_modal = True), regardless of
  # what is specified in the energy config.
  run_pipeline(cfg, force_modal_folding=True, checkpoint_callback=on_checkpoint)

  # Final sync so the Volume has everything (belt and suspenders).
  on_checkpoint({})
  return {}
