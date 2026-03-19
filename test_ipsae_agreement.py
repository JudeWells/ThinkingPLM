#!/usr/bin/env python3
"""
Test that our BAGEL ipSAE implementation agrees with the Dunbrack Lab
reference implementation (https://github.com/DunbrackLab/IPSAE).

Both implementations compute the ``ipsae_d0res`` variant:
  For each residue i in chain1, find residues j in chain2 with PAE[i,j] < cutoff.
  d0_i = calc_d0(n_valid_partners_i)
  score_i = mean(1 / (1 + (PAE[i,j] / d0_i)^2)) over valid j
  asym_score(chain1->chain2) = max(score_i)
  symmetric_score = max(asym(1->2), asym(2->1))

We test on 3 AF3 structures x 5 models each from af3_test_structures/.
"""

import json
import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Dunbrack reference implementation (extracted from ipsae.py, version 4)
# ---------------------------------------------------------------------------

def dunbrack_calc_d0_array(L):
    """
    Vectorised d0 from Dunbrack ipsae.py (version 4, Jan 2026).
    Clamps L to minimum of 26 (not 27), so d0 bottoms out at 1.0
    via the np.maximum(1.0, ...) at the end.
    """
    L = np.array(L, dtype=float)
    L = np.maximum(26, L)
    return np.maximum(1.0, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def dunbrack_ipsae_one_direction(pae_matrix, source_mask, target_mask, pae_cutoff):
    """
    Compute asymmetric ipsae_d0res (source -> target) as in Dunbrack ipsae.py.
    """
    numres = len(pae_matrix)
    source_indices = np.where(source_mask)[0]

    if len(source_indices) == 0 or not target_mask.any():
        return 0.0

    valid_pairs_matrix = np.outer(source_mask, target_mask) & (pae_matrix < pae_cutoff)
    n0res_byres = valid_pairs_matrix.sum(axis=1)
    d0res_byres = dunbrack_calc_d0_array(n0res_byres)

    per_res_scores = np.zeros(numres)
    for i in source_indices:
        valid = valid_pairs_matrix[i]
        if not valid.any():
            continue
        pae_vals = pae_matrix[i, valid]
        ptm_scores = 1.0 / (1.0 + (pae_vals / d0res_byres[i]) ** 2)
        per_res_scores[i] = ptm_scores.mean()

    return float(np.max(per_res_scores))


def dunbrack_ipsae(pae_matrix, chains, pae_cutoff=10.0):
    """Symmetric ipSAE (d0res) for each chain pair, Dunbrack reference."""
    unique_chains = list(dict.fromkeys(chains))
    chains_arr = np.array(chains)
    results = {}
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2 or (c2, c1) in results:
                continue
            m1, m2 = chains_arr == c1, chains_arr == c2
            a12 = dunbrack_ipsae_one_direction(pae_matrix, m1, m2, pae_cutoff)
            a21 = dunbrack_ipsae_one_direction(pae_matrix, m2, m1, pae_cutoff)
            results[(c1, c2)] = results[(c2, c1)] = max(a12, a21)
    return results


# ---------------------------------------------------------------------------
# BAGEL implementation (extracted from bagel/energies.py ipSAEEnergy)
# ---------------------------------------------------------------------------

def bagel_calc_d0_array(L):
    """Vectorised d0 from BAGEL ipSAEEnergy._calc_d0_array."""
    L = np.maximum(np.asarray(L, dtype=float), 27.0)
    return np.maximum(1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8, 1.0)


def bagel_ipsae_one_direction(pae, source_indices, target_indices, pae_cutoff):
    """Asymmetric ipSAE score, matching BAGEL ipSAEEnergy._ipsae_one_direction."""
    if len(source_indices) == 0 or len(target_indices) == 0:
        return 0.0

    pae_cross = pae[np.ix_(source_indices, target_indices)]
    valid_mask = pae_cross < pae_cutoff
    n_partners = valid_mask.sum(axis=1).astype(np.float64)
    has_partners = n_partners > 0
    if not np.any(has_partners):
        return 0.0

    d0_per_res = bagel_calc_d0_array(n_partners)
    per_res_scores = np.zeros(len(source_indices), dtype=np.float64)
    for i in range(len(source_indices)):
        if not has_partners[i]:
            continue
        partner_pae = pae_cross[i, valid_mask[i]]
        ptm_scores = 1.0 / (1.0 + (partner_pae / d0_per_res[i]) ** 2)
        per_res_scores[i] = ptm_scores.mean()

    return float(np.max(per_res_scores))


def bagel_ipsae(pae_matrix, chains, pae_cutoff=10.0):
    """Symmetric ipSAE for each chain pair, BAGEL logic."""
    unique_chains = list(dict.fromkeys(chains))
    chains_arr = np.array(chains)
    results = {}
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2 or (c2, c1) in results:
                continue
            i1, i2 = np.where(chains_arr == c1)[0], np.where(chains_arr == c2)[0]
            s12 = bagel_ipsae_one_direction(pae_matrix, i1, i2, pae_cutoff)
            s21 = bagel_ipsae_one_direction(pae_matrix, i2, i1, pae_cutoff)
            results[(c1, c2)] = results[(c2, c1)] = max(s12, s21)
    return results


# ---------------------------------------------------------------------------
# Load AF3 data
# ---------------------------------------------------------------------------

def load_af3_data(json_path):
    """Load PAE matrix and token chain IDs from an AF3 full_data JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data["pae"]), data["token_chain_ids"]


def find_all_af3_models(test_dir):
    """Find all (structure_name, model_index, json_path) triples."""
    models = []
    for sdir in sorted(glob.glob(os.path.join(test_dir, "fold_*"))):
        name = os.path.basename(sdir)
        for jf in sorted(glob.glob(os.path.join(sdir, "*_full_data_*.json"))):
            # Extract model index from filename like ..._full_data_3.json
            idx = os.path.basename(jf).split("_full_data_")[1].replace(".json", "")
            models.append((name, idx, jf))
    return models


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "af3_test_structures")
    models = find_all_af3_models(test_dir)

    if not models:
        print(f"ERROR: No AF3 models found in {test_dir}")
        sys.exit(1)

    pae_cutoffs = [10.0, 25.0]  # test both standard and relaxed cutoff
    all_pass = True
    atol = 1e-6
    n_tests = 0
    n_nonzero = 0

    for pae_cutoff in pae_cutoffs:
        print(f"\n{'='*110}")
        print(f"PAE cutoff = {pae_cutoff}")
        print(f"{'='*110}")
        print(f"{'Structure':<45} {'Model':<7} {'Pair':<7} {'Dunbrack':<12} {'BAGEL':<12} {'Diff':<12} {'Status'}")
        print("-" * 110)

        for name, midx, jpath in models:
            pae_matrix, chains = load_af3_data(jpath)
            d_results = dunbrack_ipsae(pae_matrix, chains, pae_cutoff)
            b_results = bagel_ipsae(pae_matrix, chains, pae_cutoff)

            unique_chains = list(dict.fromkeys(chains))
            for c1 in unique_chains:
                for c2 in unique_chains:
                    if c1 >= c2:
                        continue
                    d_val = d_results.get((c1, c2), 0.0)
                    b_val = b_results.get((c1, c2), 0.0)
                    diff = abs(d_val - b_val)
                    match = diff < atol
                    n_tests += 1
                    if d_val > 0 or b_val > 0:
                        n_nonzero += 1
                    if not match:
                        all_pass = False
                    status = "PASS" if match else f"FAIL"
                    print(f"{name:<45} {midx:<7} {c1}-{c2:<5} {d_val:<12.6f} {b_val:<12.6f} {diff:<12.2e} {status}")

    print(f"\n{'='*110}")
    print(f"Summary: {n_tests} tests, {n_nonzero} non-zero scores")
    print(f"{'='*110}")

    if all_pass:
        print("ALL TESTS PASSED: implementations agree exactly (atol=1e-6).")
    else:
        print("SOME TESTS FAILED: implementations differ.")
        print("\nKnown difference: Dunbrack clamps L to min 26 in calc_d0_array,")
        print("BAGEL clamps to min 27. This affects d0 when partner count <= 26:")
        print("  Dunbrack d0(L<=26) = 1.0")
        print("  BAGEL    d0(L<=26) = 1.039  (L clamped to 27 before formula)")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
