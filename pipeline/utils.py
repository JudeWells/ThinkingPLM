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
