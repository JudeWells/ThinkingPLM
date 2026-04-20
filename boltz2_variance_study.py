#!/usr/bin/env python3
"""
Measure Boltz2 prediction variance: run 32 Boltz2 predictions per sequence
for all 16 best binder sequences and record iPTM and ipSAE per run.

Outputs:
  final_evaluation/boltz2_variance.csv — per-prediction metrics
  final_evaluation/boltz2_variance_summary.csv — per-sequence summary stats
"""

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE = Path("/mnt/disk2/ThinkingPLM")
EVAL_DIR = BASE / "final_evaluation"
CSV_IN = EVAL_DIR / "evaluation_results.csv"
CSV_OUT = EVAL_DIR / "boltz2_variance.csv"
CSV_SUMMARY = EVAL_DIR / "boltz2_variance_summary.csv"

TARGET_SEQ = (
    "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQKTLFIQCDVADQQQLRDTFRKVVDHFGR"
    "LDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANL"
    "MNSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ"
)

N_REPEATS = 32


def seq_to_chains(sequence, chain_separator=":"):
    from bagel.chain import Chain, Residue
    chain_seqs = sequence.split(chain_separator)
    chains = []
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ci, seq in enumerate(chain_seqs):
        cid = chain_ids[ci]
        residues = [Residue(name=aa, chain_ID=cid, index=j) for j, aa in enumerate(seq)]
        chains.append(Chain(residues=residues))
    return chains


def compute_ipsae_from_pae(pae, binder_len, target_len, cutoff=10.0):
    """Compute ipSAE from a PAE matrix."""
    if pae is None or pae.size == 0:
        return float("nan")
    total_len = binder_len + target_len
    if pae.shape[0] < total_len or pae.shape[1] < total_len:
        return float("nan")
    pae_bt = pae[:binder_len, binder_len:total_len]
    pae_tb = pae[binder_len:total_len, :binder_len]
    cross = np.concatenate([pae_bt.flatten(), pae_tb.flatten()])
    return -float(np.mean(cross < cutoff))


def main():
    import gc
    import torch
    from bagel.oracles.folding.boltz import Boltz

    with open(CSV_IN) as f:
        entries = list(csv.DictReader(f))
    entries.sort(key=lambda r: float(r["campaign_best_energy"]))

    print(f"Loaded {len(entries)} sequences")
    print(f"Running {N_REPEATS} Boltz2 predictions per sequence ({len(entries) * N_REPEATS} total)")

    oracle = Boltz()
    target_len = len(TARGET_SEQ)

    all_rows = []
    summaries = []

    # Resume support: load existing results
    if CSV_OUT.is_file():
        with open(CSV_OUT) as f:
            existing = list(csv.DictReader(f))
        done_keys = {(r["run_name"], int(r["repeat"])) for r in existing}
        all_rows = existing
        print(f"Resuming: {len(existing)} predictions already done")
    else:
        done_keys = set()

    for si, entry in enumerate(entries):
        name = entry["run_name"].split("/")[-1]
        binder_seq = entry["binder_seq"]
        binder_len = int(entry["binder_len"])
        complex_seq = binder_seq + ":" + TARGET_SEQ
        chains = seq_to_chains(complex_seq)
        campaign_energy = float(entry["campaign_best_energy"])

        iptms = []
        ipsaes = []

        for rep in range(N_REPEATS):
            if (name, rep) in done_keys:
                # Already computed — just collect for summary
                for r in all_rows:
                    if r["run_name"] == name and int(r["repeat"]) == rep:
                        iptms.append(float(r["iptm"]))
                        ipsaes.append(float(r["ipsae"]))
                        break
                continue

            torch.cuda.empty_cache()
            gc.collect()

            try:
                result = oracle.fold(chains)

                iptm = float("nan")
                if hasattr(result, "chain_pair_iptm") and result.chain_pair_iptm is not None:
                    arr = np.asarray(result.chain_pair_iptm)
                    if arr.size > 0:
                        iptm = float(np.mean(arr))

                pae = None
                if hasattr(result, "pae") and result.pae is not None:
                    pae = np.asarray(result.pae)
                    if pae.ndim == 3:
                        pae = pae[0]

                ipsae = compute_ipsae_from_pae(pae, binder_len, target_len)

            except Exception as e:
                print(f"    FAILED rep {rep}: {e}")
                iptm = float("nan")
                ipsae = float("nan")

            iptms.append(iptm)
            ipsaes.append(ipsae)

            row = {
                "run_name": name,
                "repeat": rep,
                "iptm": iptm,
                "ipsae": ipsae,
                "binder_len": binder_len,
                "campaign_energy": campaign_energy,
            }
            all_rows.append(row)

            # Save incrementally
            if len(all_rows) % 8 == 0 or rep == N_REPEATS - 1:
                fieldnames = ["run_name", "repeat", "iptm", "ipsae", "binder_len", "campaign_energy"]
                with open(CSV_OUT, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)

        # Summary for this sequence
        iptms_clean = [v for v in iptms if v == v]
        ipsaes_clean = [v for v in ipsaes if v == v]

        summary = {
            "run_name": name,
            "binder_len": binder_len,
            "campaign_energy": campaign_energy,
            "n_successful": len(iptms_clean),
            "iptm_mean": float(np.mean(iptms_clean)) if iptms_clean else float("nan"),
            "iptm_std": float(np.std(iptms_clean)) if iptms_clean else float("nan"),
            "iptm_min": float(np.min(iptms_clean)) if iptms_clean else float("nan"),
            "iptm_max": float(np.max(iptms_clean)) if iptms_clean else float("nan"),
            "ipsae_mean": float(np.mean(ipsaes_clean)) if ipsaes_clean else float("nan"),
            "ipsae_std": float(np.std(ipsaes_clean)) if ipsaes_clean else float("nan"),
            "ipsae_min": float(np.min(ipsaes_clean)) if ipsaes_clean else float("nan"),
            "ipsae_max": float(np.max(ipsaes_clean)) if ipsaes_clean else float("nan"),
        }
        summaries.append(summary)

        print(f"[{si+1}/{len(entries)}] {name:<40s} "
              f"iPTM={summary['iptm_mean']:.3f}+/-{summary['iptm_std']:.3f} "
              f"[{summary['iptm_min']:.3f}, {summary['iptm_max']:.3f}]  "
              f"ipSAE={summary['ipsae_mean']:.3f}+/-{summary['ipsae_std']:.3f}")

    # Final save
    fieldnames = ["run_name", "repeat", "iptm", "ipsae", "binder_len", "campaign_energy"]
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_fields = list(summaries[0].keys())
    with open(CSV_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nDone. {len(all_rows)} predictions saved to {CSV_OUT}")
    print(f"Summary: {CSV_SUMMARY}")


if __name__ == "__main__":
    main()
