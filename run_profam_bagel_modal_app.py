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

# Image definition: install BAGEL and ProFam from their GitHub repositories,
# plus any additional dependencies, then add the local pipeline files.
image = (
  modal.Image.debian_slim()
  # git is needed for pip install from GitHub URLs
  .apt_install("git")
  .pip_install(
    # BAGEL (biobagel) — includes biotite, boileroom, numpy, pandas, pydantic, matplotlib
    "biobagel[local] @ git+https://github.com/softnanolab/bagel.git",
    # ProFam — includes torch, transformers, lightning, hydra-core, etc.
    "git+https://github.com/alex-hh/profam.git",
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
  # Add the local pipeline files (configs, scripts) to the container.
  # Exclude profam/ and bagel/ source trees (pip-installed above) and
  # large artefacts that aren't needed inside the container.
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

# Optionally include cached model weights so BAGEL doesn't re-download them.
if LOCAL_MODEL_DIR.exists():
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
  """
  import json

  _setup_remote_env()

  from run_profam_bagel_pipeline import run_pipeline

  cfg = _reconstruct_config(cfg_dict)

  # Clear stale checkpoint files from any previous run.
  vol_root = Path(_VOL_MOUNT)
  import shutil
  for child in vol_root.iterdir():
    if child.is_file():
      child.unlink()
    elif child.is_dir():
      shutil.rmtree(child)
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
