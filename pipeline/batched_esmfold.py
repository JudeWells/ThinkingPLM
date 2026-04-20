"""
Batched ESMFold oracle — processes multiple complexes in a single forward pass.

Subclasses BAGEL's ESMFold to override ``predict_batch`` with true batching.
Instead of N sequential forward passes (~11s each), a single padded batch
runs in ~15-20s total, giving a ~5-8x speedup for typical batch sizes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Type

import numpy as np

from bagel.chain import Chain
from bagel.constants import atom_order
from bagel.oracles.folding.esmfold import ESMFold, ESMFoldResult
from bagel.oracles.folding.utils import reindex_chains

logger = logging.getLogger(__name__)


class BatchedESMFold(ESMFold):
    """ESMFold with batched prediction support.

    When ``predict_batch`` is called with multiple chain-lists, all
    complexes are folded in a **single model forward pass** (padded to
    the longest sequence).  This avoids the ~11s model overhead per
    prediction and is typically 5-8x faster than sequential folding.

    Falls back to sequential for single predictions or when batching
    fails (e.g. OOM on very large batches).

    Parameters
    ----------
    max_batch_size : int
        Maximum number of complexes per forward pass.  If more are
        requested, they are split into sub-batches.  Default 16.
    use_modal : bool
        If True, use Modal for remote execution (batching still applies).
    config : dict
        ESMFold config dict passed to boileroom.
    """

    result_class: Type[ESMFoldResult] = ESMFoldResult

    def __init__(
        self,
        use_modal: bool = False,
        max_batch_size: int = 16,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(use_modal=use_modal, config=config or {}, **kwargs)
        self.max_batch_size = max_batch_size

    def predict_batch(
        self, chains_list: list[list[Chain]]
    ) -> list[ESMFoldResult]:
        """Fold multiple complexes in batched forward passes.

        Each element of ``chains_list`` is a list of chains for one
        complex (e.g. ``[binder_chain, target_chain]``).

        Returns one ``ESMFoldResult`` per input complex, in order.
        """
        n = len(chains_list)
        if n <= 1:
            return [self.fold(chains_list[0])] if n == 1 else []

        # Build all sequences as colon-separated strings
        all_sequences: list[str] = []
        for chains in chains_list:
            monomers = [chain.sequence for chain in chains]
            all_sequences.append(":".join(monomers))

        # Process in sub-batches if needed
        results: list[ESMFoldResult] = []
        for batch_start in range(0, n, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, n)
            batch_seqs = all_sequences[batch_start:batch_end]
            batch_chains = chains_list[batch_start:batch_end]
            bs = len(batch_seqs)

            logger.info(
                f"BatchedESMFold: folding {bs} complexes in one forward pass "
                f"(batch {batch_start//self.max_batch_size + 1})"
            )

            try:
                if self.use_modal:
                    output = self.model.fold.remote(batch_seqs)
                else:
                    output = self.model.fold.local(batch_seqs)

                # Split the batched output into individual ESMFoldResults
                for i in range(bs):
                    result = self._reduce_batched_output(
                        output, i, batch_chains[i]
                    )
                    results.append(result)

            except Exception as e:
                logger.warning(
                    f"BatchedESMFold: batch failed ({e}), "
                    f"falling back to sequential for this sub-batch"
                )
                for chains in batch_chains:
                    results.append(self.fold(chains))

        return results

    def _reduce_batched_output(
        self,
        output: Any,  # ESMFoldOutput
        batch_idx: int,
        chains: list[Chain],
    ) -> ESMFoldResult:
        """Extract a single result from a batched ESMFoldOutput.

        Parameters
        ----------
        output : ESMFoldOutput
            Batched output from boileroom ESMFold.
        batch_idx : int
            Index of this complex within the batch.
        chains : list[Chain]
            The input chains for this complex (for chain ID mapping).
        """
        # atom_array is a list of AtomArray, one per batch element.
        # When batched, ESMFold pads shorter sequences to the max length.
        # Padding residues get chain_id '@'.  Strip them before reindexing.
        import pandas as pd
        raw_atoms = output.atom_array[batch_idx]
        real_mask = raw_atoms.chain_id != "@"
        if not real_mask.all():
            raw_atoms = raw_atoms[real_mask]

        # reindex_chains expects a list of length 1.
        atoms = reindex_chains(
            [raw_atoms],
            [chain.chain_ID for chain in chains],
        )

        # plddt: (batch, residue, atom37) → extract batch_idx, take CA
        plddt = output.plddt[batch_idx]  # (residue, atom37)
        # Trim padding: padded residues have plddt == -1 or 0
        n_residues = sum(c.length for c in chains)
        local_plddt = plddt[:n_residues, atom_order["CA"]]
        local_plddt = local_plddt[np.newaxis, :]  # (1, n_residues)
        # Clamp to [0, 1] (padding artefacts)
        local_plddt = np.clip(local_plddt, 0.0, 1.0)

        # ptm: ESMFold may return a single scalar for the whole batch
        # rather than per-element values. Extract per-element if possible,
        # otherwise raise an error for batch_size > 1 since sharing a
        # single PTM across different complexes is incorrect.
        ptm = output.ptm
        ptm_np = np.asarray(ptm)
        if ptm_np.ndim >= 1 and ptm_np.shape[0] > 1:
            # Per-element PTM available — index into it
            ptm_val = float(ptm_np[batch_idx])
        elif ptm_np.ndim == 0 or (ptm_np.ndim >= 1 and ptm_np.shape[0] == 1):
            # Single scalar — only valid for batch_size == 1
            batch_size = len(output.atom_array)
            if batch_size > 1:
                raise ValueError(
                    f"ESMFold returned a single PTM scalar for a batch of "
                    f"{batch_size} complexes. Per-element PTM is not available, "
                    f"so batched prediction cannot produce correct PTM values. "
                    f"Use sequential folding or an energy config that does not "
                    f"depend on PTM."
                )
            ptm_val = float(ptm_np.item())
        else:
            ptm_val = float(ptm_np.item())
        ptm_arr = np.array([ptm_val], dtype=np.float64)

        # pae: (batch, residue, residue) → trim to n_residues
        pae = output.predicted_aligned_error[batch_idx]  # (residue, residue)
        pae = pae[:n_residues, :n_residues]
        pae = pae[np.newaxis, :, :]  # (1, n_residues, n_residues)

        return ESMFoldResult(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt.astype(np.float64),
            ptm=ptm_arr,
            pae=pae.astype(np.float64),
        )
