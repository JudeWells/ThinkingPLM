#!/usr/bin/env python3
"""BindCraft-style AF2 predictions for the 180 sampled sequences (correlation study).

Same protocol as run_bindcraft_style_af2.py but reads from
boltz2_corr_manifest.json so we get the matched 30 sequences per target.

Output structure:
  af2_template_corr_output/<name>/
    model_<n>.pdb
    model_<n>.npz
    summary.json
"""

import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

BASE = Path("/mnt/disk2/ThinkingPLM")
MANIFEST = BASE / "boltz2_corr_manifest.json"
TARGET_PDBS = BASE / "target_pdbs"
OUT_DIR = BASE / "af2_template_corr_output"

TARGET_PDB_FILES = {
    "2GDZ": TARGET_PDBS / "2GDZ.pdb",
    "1YCR_MDM2": TARGET_PDBS / "1YCR_MDM2.pdb",
    "4OYD_epstein_barr": TARGET_PDBS / "4OYD_epstein_barr.pdb",
    "4ZQK_PD-L1": TARGET_PDBS / "4ZQK_PD-L1.pdb",
    "1TNF_TNF_alpha": TARGET_PDBS / "1TNF_TNF_alpha.pdb",
    "2VSM_nipah": TARGET_PDBS / "2VSM_nipah.pdb",
}

NUM_RECYCLES = 3
PREDICTION_MODELS = [0, 1]
AF_PARAMS_DIR = "/mnt/disk2/BindCraft/params"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(MANIFEST) as f:
        entries = json.load(f)

    from colabdesign.af import mk_afdesign_model
    from colabdesign.shared.utils import clear_mem

    # Sort by (target, binder_len) to amortize compilation
    entries_sorted = sorted(entries, key=lambda e: (e["target_id"], e["binder_len"]))

    cur_key = None
    cur_model = None
    t_start = time.time()

    for i, entry in enumerate(entries_sorted, 1):
        target_id = entry["target_id"]
        binder_seq = entry["binder_seq"]
        binder_len = entry["binder_len"]
        name = entry["name"]

        sub = OUT_DIR / name
        sub.mkdir(exist_ok=True)
        summary_path = sub / "summary.json"
        if summary_path.exists():
            print(f"[{i}/{len(entries_sorted)}] SKIP {name} (already done)")
            continue

        target_pdb = TARGET_PDB_FILES.get(target_id)
        if target_pdb is None or not target_pdb.exists():
            print(f"[{i}/{len(entries_sorted)}] SKIP {name}: no target PDB")
            continue

        key = (target_id, binder_len)
        if key != cur_key:
            print(f"\n[{i}/{len(entries_sorted)}] compile model for target={target_id}, binder_len={binder_len}", flush=True)
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
        print(f"[{i}/{len(entries_sorted)}] {name} (b={binder_len}, t={target_len})", flush=True)

        per_model = []
        for model_num in PREDICTION_MODELS:
            t0 = time.time()
            cur_model.predict(
                seq=binder_seq, models=[model_num], num_recycles=NUM_RECYCLES,
                verbose=False,
            )
            elapsed = time.time() - t0

            pdb_path = sub / f"model_{model_num + 1}.pdb"
            cur_model.save_pdb(str(pdb_path))

            aux = cur_model.aux
            log = aux["log"]
            plddt = np.asarray(aux["plddt"])
            pae = np.asarray(aux["pae"])

            np.savez_compressed(
                sub / f"model_{model_num + 1}.npz",
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
                "binder_plddt_mean": float(np.mean(plddt[target_len:]) * 100),
                "elapsed_sec": elapsed,
            })

        with open(summary_path, "w") as f:
            json.dump({
                "name": name,
                "target_id": target_id,
                "sample_type": entry["sample_type"],
                "binder_seq": binder_seq,
                "binder_len": binder_len,
                "target_len": int(target_len),
                "total_energy": entry["total_energy"],
                "models": per_model,
            }, f, indent=2)

    print(f"\nDone in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
