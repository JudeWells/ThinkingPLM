"""Utility functions for the ProFam + BAGEL pipeline.

Contains sequence similarity, energy-to-probability conversion, and
subset sampling helpers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
from Bio import Align  # type: ignore
from Bio.Align import PairwiseAligner, substitution_matrices  # type: ignore


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

    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]

    identical = sum(
        1 for a, b in zip(aligned_seq1, aligned_seq2)
        if a == b and a != '-' and b != '-'
    )

    max_len = max(len(seq1), len(seq2))
    return identical / max_len if max_len > 0 else 0.0


def _pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Return the fraction of identical residues from a global alignment.

    Uses Biopython's ``PairwiseAligner`` (Needleman-Wunsch) so that
    insertions and deletions are handled correctly — a single indel no longer
    shifts all downstream positions and artificially tanks the score.

    The identity is defined as::

        identity = matched_columns / alignment_length

    where *alignment_length* includes gap columns on either side.
    """
    if not seq_a or not seq_b:
        return 0.0

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

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
    """Compute the average sequence similarity between generated and initial sequences.

    For each generated sequence the similarity to each initial sequence is
    computed via global pairwise alignment (Needleman-Wunsch).  The best (maximum)
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
    """Convert energies into sampling probabilities via a softmax over -energy / T.

    Lower energies correspond to higher probabilities.
    """
    if temperature <= 0:
        raise ValueError("softmax_temperature must be > 0.")
    arr = np.asarray(energies, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute softmax for empty energy list.")

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
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
    """Sample a subset of indices of size floor(f_inject * num_items) according
    to probabilities ``probs``.

    Parameters
    ----------
    replace : bool
        If True (default), sample with replacement.  If False, sample without
        replacement; when the requested subset size exceeds the number of
        candidates with non-zero probability, fall back to returning only the
        index of the best (lowest-energy) candidate.
    energies : sequence of float, optional
        Required when ``replace=False`` so that the best candidate can be
        identified as a fallback.
    subset_size : int, optional
        If provided, overrides the ``floor(f_inject * num_items)`` calculation.
    """
    if num_items <= 0:
        raise ValueError("num_items must be > 0.")
    k = subset_size if subset_size is not None else int(math.floor(f_inject * num_items))
    if k <= 0:
        k = 1

    if replace:
        idx = rng.choice(num_items, size=k, replace=True, p=probs)
        return np.asarray(idx, dtype=int)

    num_nonzero = int(np.sum(probs > 0))
    if k <= num_nonzero:
        idx = rng.choice(num_items, size=k, replace=False, p=probs)
        return np.asarray(idx, dtype=int)

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


def _ipsae_d0(L: float) -> float:
    """Yang & Skolnick (2004) d0, clamped to match Dunbrack ipSAE reference."""
    L = max(L, 27.0)
    return max(1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8, 1.0)


def _ipsae_one_direction(
    pae: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    pae_cutoff: float,
) -> float:
    """Asymmetric ipSAE (source -> target). Max over source residues.

    For each source residue i:
      - partners j = {j in target : pae[i,j] < pae_cutoff}
      - n_i = |partners|; d0_i = _ipsae_d0(n_i)
      - score_i = mean_j 1 / (1 + (pae[i,j] / d0_i)^2)
    Returns max_i score_i.
    """
    if len(source_indices) == 0 or len(target_indices) == 0:
        return 0.0

    pae_cross = pae[np.ix_(source_indices, target_indices)]
    valid_mask = pae_cross < pae_cutoff
    n_partners = valid_mask.sum(axis=1).astype(np.float64)

    has_partners = n_partners > 0
    if not np.any(has_partners):
        return 0.0

    per_res_scores = np.zeros(len(source_indices), dtype=np.float64)
    for i in range(len(source_indices)):
        if not has_partners[i]:
            continue
        d0_i = _ipsae_d0(float(n_partners[i]))
        partner_pae = pae_cross[i, valid_mask[i]]
        ptm_scores = 1.0 / (1.0 + (partner_pae / d0_i) ** 2)
        per_res_scores[i] = ptm_scores.mean()

    return float(np.max(per_res_scores))


def compute_ipsae(
    pae: np.ndarray,
    chain_a_indices: np.ndarray | Sequence[int],
    chain_b_indices: np.ndarray | Sequence[int],
    pae_cutoff: float = 10.0,
) -> float:
    """Compute ipSAE (Dunbrack `ipsae_d0res` variant) from a PAE matrix.

    Reference: Dunbrack Lab, "Res ipSAE loquunt" (2025),
    https://github.com/DunbrackLab/IPSAE. Mirrors
    ``bagel.energies.ipSAEEnergy`` so scores are directly comparable to
    pipeline-time ``ipSAE`` energy values.

    Higher is better; bounded in [0, 1]. Values > 0.6 suggest likely binding.

    Parameters
    ----------
    pae : np.ndarray
        Square PAE matrix of shape [n_residues, n_residues] in Angstroms.
    chain_a_indices, chain_b_indices : array-like of int
        Residue indices (rows/cols of ``pae``) for each chain.
    pae_cutoff : float
        PAE threshold below which a residue pair is considered interface.
    """
    pae = np.asarray(pae)
    a = np.asarray(chain_a_indices, dtype=np.int_)
    b = np.asarray(chain_b_indices, dtype=np.int_)
    score_a_to_b = _ipsae_one_direction(pae, a, b, pae_cutoff)
    score_b_to_a = _ipsae_one_direction(pae, b, a, pae_cutoff)
    return max(score_a_to_b, score_b_to_a)
