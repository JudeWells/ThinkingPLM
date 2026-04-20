#!/usr/bin/env python3
"""BindCraft-style AF2 predictions for the 35 best binder sequences.

Replicates BindCraft's validation prediction protocol:
  - mk_afdesign_model(protocol="binder", use_multimer=False, num_recycles=3)
  - prep_inputs(pdb_filename=target_pdb, chain="A", binder_len=N)
  - predict(seq=binder_seq, models=[0, 1])

This uses AF2 monomer mode with the target PDB as a structural template,
which gives much stronger predictions than single-sequence multimer.

Outputs (per binder):
  af2_template_output/<safe_name>/
    model_<n>.pdb       - predicted complex
    model_<n>.npz       - {plddt, pae, ptm, iptm, target_len, binder_len}
    summary.json        - aggregated metrics across models

Run with the BindCraft conda env:
  /home/judewells/miniconda3/envs/BindCraft/bin/python run_bindcraft_style_af2.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

BASE = Path("/mnt/disk2/ThinkingPLM")
INPUT_JSON = BASE / "colabfold_input" / "best_sequences.json"
TARGET_PDBS = BASE / "target_pdbs"
OUT_DIR = BASE / "af2_template_output"

# Map our target_id -> the cleaned single-chain PDB file we built
TARGET_PDB_FILES = {
    "2GDZ": TARGET_PDBS / "2GDZ.pdb",
    "1YCR_MDM2": TARGET_PDBS / "1YCR_MDM2.pdb",
    "4OYD_epstein_barr": TARGET_PDBS / "4OYD_epstein_barr.pdb",
    "4ZQK_PD-L1": TARGET_PDBS / "4ZQK_PD-L1.pdb",
    "1TNF_TNF_alpha": TARGET_PDBS / "1TNF_TNF_alpha.pdb",
    "2VSM_nipah": TARGET_PDBS / "2VSM_nipah.pdb",
}

NUM_RECYCLES = 3
PREDICTION_MODELS = [0, 1]  # BindCraft default for use_multimer_design=True
AF_PARAMS_DIR = "/mnt/disk2/BindCraft/params"


def get_safe_name(entry):
    safe = entry["campaign"].replace("/", "_")
    return f"sc_rep3_{safe}" if entry["target_id"] == "2GDZ" else f"{entry['target_id']}_{safe}"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(INPUT_JSON) as f:
        entries = json.load(f)

    # Group by target so we compile prep_inputs once per (target_pdb, binder_len)
    # Note: binder_len varies per sequence so we still recompile per binder length.
    # Group by (target, binder_len) to amortize compilation.
    from colabdesign.af import mk_afdesign_model
    from colabdesign.shared.utils import clear_mem

    # Sort by (target_id, binder_len) so we minimize recompilations
    entries_sorted = sorted(
        entries, key=lambda e: (e["target_id"], len(e["binder_seq"]))
    )

    cur_key = None
    cur_model = None
    t_start = time.time()

    for i, entry in enumerate(entries_sorted, 1):
        target_id = entry["target_id"]
        binder_seq = entry["binder_seq"]
        binder_len = len(binder_seq)
        safe_name = get_safe_name(entry)

        sub_dir = OUT_DIR / safe_name
        sub_dir.mkdir(exist_ok=True)
        summary_path = sub_dir / "summary.json"
        if summary_path.exists():
            print(f"[{i}/{len(entries_sorted)}] SKIP {safe_name} (already done)")
            continue

        target_pdb = TARGET_PDB_FILES.get(target_id)
        if target_pdb is None or not target_pdb.exists():
            print(f"[{i}/{len(entries_sorted)}] SKIP {safe_name}: no target PDB")
            continue

        key = (target_id, binder_len)
        if key != cur_key:
            # New compilation needed
            print(f"\n[{i}/{len(entries_sorted)}] compile model for target={target_id}, binder_len={binder_len}")
            clear_mem()
            cur_model = mk_afdesign_model(
                protocol="binder",
                num_recycles=NUM_RECYCLES,
                data_dir=AF_PARAMS_DIR,
                use_multimer=False,
            )
            cur_model.prep_inputs(
                pdb_filename=str(target_pdb),
                chain="A",
                binder_len=binder_len,
                rm_target_seq=False,
                rm_target_sc=False,
            )
            cur_key = key

        target_len = cur_model._target_len
        print(f"[{i}/{len(entries_sorted)}] predict {safe_name} (binder_len={binder_len}, target_len={target_len})", flush=True)

        per_model = []
        for model_num in PREDICTION_MODELS:
            t0 = time.time()
            cur_model.predict(
                seq=binder_seq, models=[model_num], num_recycles=NUM_RECYCLES,
                verbose=False,
            )
            elapsed = time.time() - t0

            pdb_path = sub_dir / f"model_{model_num + 1}.pdb"
            cur_model.save_pdb(str(pdb_path))

            aux = cur_model.aux
            log = aux["log"]
            plddt = np.asarray(aux["plddt"])  # 0-1 scale
            pae = np.asarray(aux["pae"])      # Angstroms

            npz_path = sub_dir / f"model_{model_num + 1}.npz"
            np.savez_compressed(
                npz_path,
                plddt=plddt,
                pae=pae,
                ptm=float(log.get("ptm", float("nan"))),
                iptm=float(log.get("i_ptm", float("nan"))),
                target_len=int(target_len),
                binder_len=int(binder_len),
            )

            per_model.append({
                "model": model_num + 1,
                "ptm": float(log.get("ptm", float("nan"))),
                "iptm": float(log.get("i_ptm", float("nan"))),
                "pae": float(log.get("pae", float("nan"))),
                "i_pae": float(log.get("i_pae", float("nan"))),
                "plddt_mean": float(np.mean(plddt) * 100),
                "binder_plddt_mean": float(np.mean(plddt[target_len:]) * 100),
                "elapsed_sec": elapsed,
            })
            print(f"    model {model_num + 1}: ptm={per_model[-1]['ptm']:.3f} iptm={per_model[-1]['iptm']:.3f} binder_plddt={per_model[-1]['binder_plddt_mean']:.1f} ({elapsed:.1f}s)", flush=True)

        with open(summary_path, "w") as f:
            json.dump({
                "safe_name": safe_name,
                "target_id": target_id,
                "binder_seq": binder_seq,
                "binder_len": binder_len,
                "target_len": int(target_len),
                "models": per_model,
            }, f, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
