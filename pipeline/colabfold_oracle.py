"""
ColabFold folding oracle — wraps the ``colabfold_batch`` CLI as a BAGEL FoldingOracle.

Produces results compatible with BoltzResult field layout so that existing
energy terms (ipSAEEnergy, PTMEnergy, iPTMEnergy, PLDDTEnergy, etc.) work
out of the box.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Type

import numpy as np
import numpy.typing as npt
from pydantic import field_validator

from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from bagel.chain import Chain
from bagel.oracles.folding.base import FoldingOracle, FoldingResult

logger = logging.getLogger(__name__)

# Default ColabFold binary path
_DEFAULT_COLABFOLD_BIN = (
    "/mnt/disk2/colabfold_2025_09/localcolabfold/colabfold-conda/bin/colabfold_batch"
)


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

class ColabFoldResult(FoldingResult):
    """FoldingResult with ColabFold confidence metrics.

    Fields mirror BoltzResult so that existing energy terms work unchanged.
    """

    input_chains: list[Chain]
    structure: AtomArray                      # from PDB output
    local_plddt: npt.NDArray[np.float64]      # [1, n_residues] — 0-1 scale
    pae: npt.NDArray[np.float64]              # [1, n_residues, n_residues]
    ptm: npt.NDArray[np.float64]              # [1] — overall pTM (0-1)
    chain_pair_iptm: npt.NDArray[np.float64]  # [n_chains, n_chains]

    class Config:
        arbitrary_types_allowed = True

    @field_validator("local_plddt")
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not isinstance(v, np.ndarray):
            raise ValueError("local_plddt must be a numpy array")
        if not np.all((v >= -0.01) & (v <= 1.01)):
            raise ValueError("All values in local_plddt must be between 0 and 1")
        return np.clip(v, 0.0, 1.0)

    @field_validator("ptm")
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not isinstance(v, np.ndarray):
            raise ValueError("ptm must be a numpy array")
        return v

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix(".plddt"), self.local_plddt[0], fmt="%.6f", header="plddt")
        np.savetxt(filepath.with_suffix(".pae"), self.pae[0], fmt="%.6f", header="pae")
        np.savetxt(filepath.with_suffix(".iptm"), self.chain_pair_iptm, fmt="%.6f", header="chain_pair_iptm")


# ---------------------------------------------------------------------------
# Oracle class
# ---------------------------------------------------------------------------

class ColabFold(FoldingOracle):
    """
    ColabFold structure prediction via local CLI subprocess.

    Invokes ``colabfold_batch`` on a FASTA input and parses the resulting
    PDB structures and JSON confidence scores.

    Parameters
    ----------
    colabfold_bin : str
        Path to the ``colabfold_batch`` binary.
    num_models : int
        Number of AlphaFold2 models to run (1-5). Use 1 for speed.
    num_recycle : int
        Number of recycling steps. Use 1 for speed, 3 for accuracy.
    num_seeds : int
        Number of random seeds per model.
    rank_by : str
        Ranking metric: "iptm", "ptm", "multimer", "plddt".
    timeout : int
        Subprocess timeout in seconds.
    extra_args : list[str]
        Additional CLI arguments passed to ``colabfold_batch``.
    """

    result_class: Type[ColabFoldResult] = ColabFoldResult

    def __init__(
        self,
        colabfold_bin: str = _DEFAULT_COLABFOLD_BIN,
        num_models: int = 1,
        num_recycle: int = 3,
        num_seeds: int = 1,
        rank_by: str = "iptm",
        timeout: int = 1800,
        msa_mode: str | None = None,
        extra_args: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.colabfold_bin = colabfold_bin
        self.num_models = num_models
        self.num_recycle = num_recycle
        self.num_seeds = num_seeds
        self.rank_by = rank_by
        self.timeout = timeout
        self.msa_mode = msa_mode  # e.g. "single_sequence" to skip MSA for speed
        self.extra_args = extra_args or []
        self.config = config or {}

    # --------------------------------------------------------------------- #
    # FASTA input
    # --------------------------------------------------------------------- #

    @staticmethod
    def _chains_to_fasta(chains: list[Chain], fasta_path: Path) -> str:
        """
        Write a ColabFold-compatible FASTA file.

        For multimer prediction, chains are joined with ':' separator
        on a single line, matching ColabFold's expected format.

        Returns the FASTA header name (used to locate output files).
        """
        name = "query"
        sequences = [chain.sequence for chain in chains]
        combined = ":".join(sequences)
        fasta_path.write_text(f">{name}\n{combined}\n")
        return name

    # --------------------------------------------------------------------- #
    # Output parsing
    # --------------------------------------------------------------------- #

    @staticmethod
    def _find_best_score_file(results_dir: Path, name: str) -> Path | None:
        """Find the rank_001 score JSON file."""
        # ColabFold names: {name}_scores_rank_001_*.json
        candidates = sorted(results_dir.glob(f"{name}_scores_rank_001_*.json"))
        if candidates:
            return candidates[0]
        # Fallback: any score file
        candidates = sorted(results_dir.glob(f"{name}_scores_*.json"))
        if candidates:
            return candidates[0]
        # Last resort: any scores file
        candidates = sorted(results_dir.glob("*scores*.json"))
        return candidates[0] if candidates else None

    @staticmethod
    def _find_best_pdb_file(results_dir: Path, name: str) -> Path | None:
        """Find the rank_001 PDB structure file."""
        candidates = sorted(results_dir.glob(f"{name}_unrelaxed_rank_001_*.pdb"))
        if candidates:
            return candidates[0]
        # Fallback: any unrelaxed PDB
        candidates = sorted(results_dir.glob(f"{name}_unrelaxed_*.pdb"))
        if candidates:
            return candidates[0]
        candidates = sorted(results_dir.glob("*.pdb"))
        return candidates[0] if candidates else None

    def _parse_output(
        self,
        results_dir: Path,
        name: str,
        chains: list[Chain],
    ) -> ColabFoldResult:
        """Parse ColabFold output files into a ColabFoldResult."""

        # --- Score JSON ---
        score_file = self._find_best_score_file(results_dir, name)
        if score_file is None:
            raise FileNotFoundError(
                f"No ColabFold score file found in {results_dir}. "
                f"ColabFold may have failed. Check stderr."
            )

        with open(score_file) as f:
            scores = json.load(f)

        # --- PAE [1, N, N] ---
        pae_raw = np.array(scores["pae"], dtype=np.float64)
        pae = pae_raw[np.newaxis, :, :]  # [1, N, N]

        # --- pLDDT [1, N] — normalize from 0-100 to 0-1 ---
        plddt_raw = np.array(scores.get("plddt", []), dtype=np.float64)
        if plddt_raw.size > 0 and plddt_raw.max() > 1.0:
            plddt_raw = plddt_raw / 100.0
        local_plddt = plddt_raw[np.newaxis, :]  # [1, N]

        # --- pTM [1] ---
        ptm_val = float(scores.get("ptm", 0.0))
        ptm = np.array([ptm_val], dtype=np.float64)

        # --- chain_pair_iptm [n_chains, n_chains] ---
        # ColabFold provides a single iptm value for the entire complex.
        # Construct a chain-pair matrix: diagonal = ptm, off-diagonal = iptm.
        iptm_val = float(scores.get("iptm", 0.0))
        n_chains = len(chains)
        chain_pair_iptm = np.full((n_chains, n_chains), iptm_val, dtype=np.float64)
        np.fill_diagonal(chain_pair_iptm, ptm_val)

        # --- PDB → AtomArray structure ---
        pdb_file = self._find_best_pdb_file(results_dir, name)
        if pdb_file is None:
            raise FileNotFoundError(
                f"No ColabFold PDB file found in {results_dir}."
            )

        pdb = PDBFile.read(str(pdb_file))
        structure = pdb.get_structure(model=1)

        # Remap chain IDs to match input chain ordering
        import pandas as pd
        original_chain_ids = pd.unique(structure.chain_id)
        chain_id_map = {
            old: chain.chain_ID
            for old, chain in zip(original_chain_ids, chains)
        }
        new_chain_ids = structure.chain_id.copy()
        for i in range(len(new_chain_ids)):
            if new_chain_ids[i] in chain_id_map:
                new_chain_ids[i] = chain_id_map[new_chain_ids[i]]
        structure.chain_id = new_chain_ids

        return ColabFoldResult(
            input_chains=chains,
            structure=structure,
            local_plddt=local_plddt,
            pae=pae,
            ptm=ptm,
            chain_pair_iptm=chain_pair_iptm,
        )

    # --------------------------------------------------------------------- #
    # Main fold method
    # --------------------------------------------------------------------- #

    def fold(self, chains: list[Chain]) -> ColabFoldResult:
        """
        Fold chains using ColabFold via local CLI.

        Creates a temporary FASTA input, runs ``colabfold_batch``, parses
        the output, and returns a ColabFoldResult.
        """
        with tempfile.TemporaryDirectory(prefix="colabfold_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write FASTA input
            fasta_path = tmpdir_path / "input.fasta"
            name = self._chains_to_fasta(chains, fasta_path)

            results_dir = tmpdir_path / "results"
            results_dir.mkdir()

            # Build and run colabfold_batch command
            cmd = [
                self.colabfold_bin,
                str(fasta_path),
                str(results_dir),
                "--num-models", str(self.num_models),
                "--num-seeds", str(self.num_seeds),
                "--num-recycle", str(self.num_recycle),
                "--rank", self.rank_by,
            ]
            if self.msa_mode is not None:
                cmd.extend(["--msa-mode", self.msa_mode])
            cmd.extend(self.extra_args)

            logger.info(f"Running ColabFold: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                logger.error(f"ColabFold stderr:\n{result.stderr[-2000:]}")
                raise RuntimeError(
                    f"ColabFold prediction failed (exit code {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )
            if result.stdout:
                logger.debug(f"ColabFold stdout:\n{result.stdout[-1000:]}")

            return self._parse_output(results_dir, name, chains)

    def predict_batch(self, chains_list: list[list[Chain]]) -> list[ColabFoldResult]:
        """
        Predict structures for multiple chain-lists.

        ColabFold supports batch mode by passing a directory of FASTA files.
        This writes one FASTA per job and runs colabfold_batch once.
        """
        if len(chains_list) == 1:
            return [self.fold(chains_list[0])]

        with tempfile.TemporaryDirectory(prefix="colabfold_batch_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            input_dir = tmpdir_path / "inputs"
            input_dir.mkdir()
            results_dir = tmpdir_path / "results"
            results_dir.mkdir()

            # Write one FASTA per job
            job_names: list[str] = []
            for idx, chains in enumerate(chains_list):
                job_name = f"candidate_{idx:04d}"
                job_names.append(job_name)
                fasta_path = input_dir / f"{job_name}.fasta"
                sequences = [chain.sequence for chain in chains]
                combined = ":".join(sequences)
                fasta_path.write_text(f">{job_name}\n{combined}\n")

            # Single ColabFold invocation with input directory
            timeout = max(self.timeout, 300 * len(chains_list))
            cmd = [
                self.colabfold_bin,
                str(input_dir),
                str(results_dir),
                "--num-models", str(self.num_models),
                "--num-seeds", str(self.num_seeds),
                "--num-recycle", str(self.num_recycle),
                "--rank", self.rank_by,
            ]
            if self.msa_mode is not None:
                cmd.extend(["--msa-mode", self.msa_mode])
            cmd.extend(self.extra_args)

            logger.info(f"Running ColabFold batch ({len(chains_list)} jobs): {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.error(f"ColabFold batch stderr:\n{result.stderr[-2000:]}")
                raise RuntimeError(
                    f"ColabFold batch prediction failed (exit code {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )

            # Parse each job's output
            results: list[ColabFoldResult] = []
            for job_name, chains in zip(job_names, chains_list):
                results.append(self._parse_output(results_dir, job_name, chains))

            return results
