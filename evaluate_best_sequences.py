#!/usr/bin/env python3
"""
Evaluate best sequences from boltz2 15PGDH runs with multiple structure predictors.

For each run's best sequence:
  1. Predict complex (binder + target) with ESMFold, Boltz2, AF2
  2. Predict binder monomer with ESMFold, Boltz2
  3. Compute RMSD between bound and unbound binder structures
  4. Collect all metrics (ipSAE, pLDDT, iPTM, pAE) into a CSV

Usage:
  python evaluate_best_sequences.py [--skip-af2] [--skip-boltz]
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("MODEL_DIR", os.path.expanduser("~/.cache/bagel/models"))

BASE = Path("/mnt/disk2/ThinkingPLM")
OUTPUT_DIR = BASE / "final_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SEQ = (
    "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQKTLFIQCDVADQQQLRDTFRKVVDHFGR"
    "LDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANL"
    "MNSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ"
)
TARGET_PDB = BASE / "target_pdbs" / "2GDZ.pdb"
AF_PARAMS_DIR = "/mnt/disk2/BindCraft/params"


def seq_to_chains(sequence, chain_separator=":"):
    """Convert a sequence string to a list of BAGEL Chain objects.

    Multi-chain sequences use ':' as separator. First chain gets ID 'A',
    second 'B', etc.
    """
    from bagel.chain import Chain, Residue
    chain_seqs = sequence.split(chain_separator)
    chains = []
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ci, seq in enumerate(chain_seqs):
        cid = chain_ids[ci]
        residues = [Residue(name=aa, chain_ID=cid, index=j) for j, aa in enumerate(seq)]
        chains.append(Chain(residues=residues))
    return chains


def extract_best_sequences():
    """Extract best sequence from each boltz2 run."""
    import glob
    results = []
    for pattern in [
        str(BASE / "outputs/boltz2_hairpin_15pgdh/*"),
        str(BASE / "outputs/boltz2_15pgdh_v2/*"),
    ]:
        for d in sorted(glob.glob(pattern)):
            stats = os.path.join(d, "cycle_stats.json")
            if not os.path.isfile(stats):
                continue
            try:
                data = json.load(open(stats))
            except Exception:
                continue
            name = d.replace(str(BASE / "outputs/"), "")
            entries = list(data.values())
            best_energy = float("inf")
            best_seq = None
            best_cycle = None
            for e in entries:
                bs = e.get("best_sequence", {})
                energy = bs.get("energy")
                if energy is not None and energy < best_energy:
                    best_energy = energy
                    best_seq = bs.get("sequence")
                    best_cycle = e.get("cycle")
            if best_seq:
                # Determine scaffold and method from name
                short = name.split("/", 1)[1] if "/" in name else name
                results.append({
                    "run_name": short,
                    "best_energy_boltz2_campaign": best_energy,
                    "best_cycle": best_cycle,
                    "total_cycles": len(data),
                    "binder_seq": best_seq,
                    "binder_len": len(best_seq),
                })
    return results


def compute_rmsd(atoms1, atoms2):
    """Compute RMSD between two sets of CA atoms after optimal superposition."""
    from biotite.structure import superimpose, rmsd
    if len(atoms1) != len(atoms2):
        # Truncate to shorter
        n = min(len(atoms1), len(atoms2))
        atoms1 = atoms1[:n]
        atoms2 = atoms2[:n]
    if len(atoms1) == 0:
        return float("nan")
    fitted, _ = superimpose(atoms1, atoms2)
    return float(rmsd(atoms1, fitted))


def extract_ca_atoms(structure, chain_id=None, start_res=None, end_res=None):
    """Extract CA atoms from a biotite AtomArray."""
    ca = structure[structure.atom_name == "CA"]
    if chain_id is not None:
        ca = ca[ca.chain_id == chain_id]
    if start_res is not None and end_res is not None:
        ca = ca[(ca.res_id >= start_res) & (ca.res_id <= end_res)]
    return ca


def predict_esmfold(sequence, label, output_subdir):
    """Predict structure with ESMFold. Returns (atoms, metrics_dict)."""
    from bagel.oracles.folding.esmfold import ESMFold, ESMFoldResult
    from biotite.structure.io.pdbx import CIFFile, get_structure

    oracle = ESMFold(use_modal=False)
    chains = seq_to_chains(sequence)
    result = oracle.fold(chains)

    cif_path = output_subdir / f"{label}.cif"
    result.to_cif(cif_path)
    if hasattr(result, "save_attributes"):
        result.save_attributes(output_subdir / label)

    plddt_arr = np.asarray(result.local_plddt) if hasattr(result, "local_plddt") else None
    plddt = float(np.mean(plddt_arr)) if plddt_arr is not None and plddt_arr.size > 0 else float("nan")
    ptm_arr = np.asarray(result.ptm) if hasattr(result, "ptm") else None
    ptm = float(ptm_arr.flat[0]) if ptm_arr is not None and ptm_arr.size > 0 else float("nan")

    cif = CIFFile.read(str(cif_path))
    atoms = get_structure(cif, model=1)

    return atoms, {
        "plddt": plddt,
        "ptm": ptm,
    }


def predict_boltz2(sequence, label, output_subdir):
    """Predict structure with Boltz2. Returns (atoms, metrics_dict)."""
    from bagel.oracles.folding.boltz import Boltz
    from biotite.structure.io.pdbx import CIFFile, get_structure

    oracle = Boltz()
    chains = seq_to_chains(sequence)
    result = oracle.fold(chains)

    cif_path = output_subdir / f"{label}.cif"
    result.to_cif(cif_path)
    if hasattr(result, "save_attributes"):
        result.save_attributes(output_subdir / label)

    plddt_arr = np.asarray(result.local_plddt) if hasattr(result, "local_plddt") else None
    plddt = float(np.mean(plddt_arr)) if plddt_arr is not None and plddt_arr.size > 0 else float("nan")
    ptm_arr = np.asarray(result.ptm) if hasattr(result, "ptm") else None
    ptm = float(ptm_arr.flat[0]) if ptm_arr is not None and ptm_arr.size > 0 else float("nan")
    iptm = float("nan")
    if hasattr(result, "chain_pair_iptm") and result.chain_pair_iptm is not None:
        iptm_arr = np.asarray(result.chain_pair_iptm)
        if iptm_arr.size > 0:
            iptm = float(np.mean(iptm_arr))

    cif = CIFFile.read(str(cif_path))
    atoms = get_structure(cif, model=1)

    return atoms, {
        "plddt": plddt,
        "ptm": ptm,
        "iptm": iptm,
    }


def predict_af2_complex(binder_seq, binder_len, label, output_subdir):
    """Predict binder-target complex with AF2 (BindCraft-style). Returns metrics_dict."""
    from colabdesign.af import mk_afdesign_model
    from colabdesign.shared.utils import clear_mem

    clear_mem()
    model = mk_afdesign_model(
        protocol="binder",
        num_recycles=3,
        data_dir=AF_PARAMS_DIR,
        use_multimer=False,
    )
    model.prep_inputs(
        pdb_filename=str(TARGET_PDB),
        chain="A",
        binder_len=binder_len,
        rm_target_seq=False,
        rm_target_sc=False,
    )
    target_len = model._target_len

    results_per_model = []
    for model_num in [0, 1]:
        model.predict(seq=binder_seq, models=[model_num], num_recycles=3, verbose=False)

        pdb_path = output_subdir / f"{label}_af2_model{model_num + 1}.pdb"
        model.save_pdb(str(pdb_path))

        aux = model.aux
        log = aux["log"]
        plddt = np.asarray(aux["plddt"])
        pae = np.asarray(aux["pae"])

        np.savez_compressed(
            output_subdir / f"{label}_af2_model{model_num + 1}.npz",
            plddt=plddt, pae=pae,
            ptm=float(log.get("ptm", float("nan"))),
            iptm=float(log.get("i_ptm", float("nan"))),
        )

        results_per_model.append({
            "ptm": float(log.get("ptm", float("nan"))),
            "iptm": float(log.get("i_ptm", float("nan"))),
            "binder_plddt": float(np.mean(plddt[target_len:]) * 100),
        })

    # Average across models
    avg = {}
    for key in results_per_model[0]:
        vals = [r[key] for r in results_per_model if not np.isnan(r[key])]
        avg[key] = float(np.mean(vals)) if vals else float("nan")

    clear_mem()
    return avg


def compute_ipsae(pae_matrix, binder_len, target_len, cutoff=10.0):
    """Compute ipSAE from PAE matrix."""
    if pae_matrix is None:
        return float("nan")
    # Cross-chain PAE: binder rows × target cols and vice versa
    pae_bt = pae_matrix[:binder_len, binder_len:binder_len + target_len]
    pae_tb = pae_matrix[binder_len:binder_len + target_len, :binder_len]
    cross_pae = np.concatenate([pae_bt.flatten(), pae_tb.flatten()])
    # ipSAE: fraction of interface contacts below cutoff (negated)
    contact_frac = np.mean(cross_pae < cutoff)
    return -float(contact_frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-af2", action="store_true", help="Skip AF2 predictions")
    parser.add_argument("--skip-boltz", action="store_true", help="Skip Boltz2 predictions")
    args = parser.parse_args()

    print("=== Extracting best sequences ===")
    entries = extract_best_sequences()
    entries.sort(key=lambda e: e["best_energy_boltz2_campaign"])
    print(f"Found {len(entries)} runs")
    for e in entries:
        print(f"  {e['run_name']:<45s} energy={e['best_energy_boltz2_campaign']:.4f} len={e['binder_len']}")

    csv_rows = []

    for i, entry in enumerate(entries, 1):
        name = entry["run_name"]
        binder_seq = entry["binder_seq"]
        binder_len = entry["binder_len"]
        complex_seq = binder_seq + ":" + TARGET_SEQ

        subdir = OUTPUT_DIR / name.replace("/", "_")
        subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{i}/{len(entries)}] {name} (len={binder_len}) ===")

        row = {
            "run_name": name,
            "binder_seq": binder_seq,
            "binder_len": binder_len,
            "campaign_best_energy": entry["best_energy_boltz2_campaign"],
            "best_cycle": entry["best_cycle"],
            "total_cycles": entry["total_cycles"],
        }

        # --- ESMFold ---
        print("  ESMFold complex...", flush=True)
        try:
            esm_complex_atoms, esm_complex_metrics = predict_esmfold(
                complex_seq, "esmfold_complex", subdir
            )
            row["esm_complex_plddt"] = esm_complex_metrics["plddt"]
            row["esm_complex_ptm"] = esm_complex_metrics["ptm"]
        except Exception as e:
            print(f"    ESMFold complex FAILED: {e}")
            esm_complex_atoms = None
            row["esm_complex_plddt"] = float("nan")
            row["esm_complex_ptm"] = float("nan")

        print("  ESMFold monomer...", flush=True)
        try:
            esm_mono_atoms, esm_mono_metrics = predict_esmfold(
                binder_seq, "esmfold_monomer", subdir
            )
            row["esm_mono_plddt"] = esm_mono_metrics["plddt"]
            row["esm_mono_ptm"] = esm_mono_metrics["ptm"]
        except Exception as e:
            print(f"    ESMFold monomer FAILED: {e}")
            esm_mono_atoms = None
            row["esm_mono_plddt"] = float("nan")
            row["esm_mono_ptm"] = float("nan")

        # ESMFold bound-unbound RMSD
        if esm_complex_atoms is not None and esm_mono_atoms is not None:
            try:
                bound_ca = extract_ca_atoms(esm_complex_atoms)[:binder_len]
                unbound_ca = extract_ca_atoms(esm_mono_atoms)[:binder_len]
                row["esm_rmsd_bound_unbound"] = compute_rmsd(bound_ca, unbound_ca)
            except Exception as e:
                print(f"    ESMFold RMSD FAILED: {e}")
                row["esm_rmsd_bound_unbound"] = float("nan")
        else:
            row["esm_rmsd_bound_unbound"] = float("nan")

        # --- Boltz2 ---
        if not args.skip_boltz:
            print("  Boltz2 complex...", flush=True)
            try:
                b2_complex_atoms, b2_complex_metrics = predict_boltz2(
                    complex_seq, "boltz2_complex", subdir
                )
                row["b2_complex_plddt"] = b2_complex_metrics["plddt"]
                row["b2_complex_ptm"] = b2_complex_metrics["ptm"]
                row["b2_complex_iptm"] = b2_complex_metrics["iptm"]
            except Exception as e:
                print(f"    Boltz2 complex FAILED: {e}")
                b2_complex_atoms = None
                row["b2_complex_plddt"] = float("nan")
                row["b2_complex_ptm"] = float("nan")
                row["b2_complex_iptm"] = float("nan")

            print("  Boltz2 monomer...", flush=True)
            try:
                b2_mono_atoms, b2_mono_metrics = predict_boltz2(
                    binder_seq, "boltz2_monomer", subdir
                )
                row["b2_mono_plddt"] = b2_mono_metrics["plddt"]
                row["b2_mono_ptm"] = b2_mono_metrics["ptm"]
            except Exception as e:
                print(f"    Boltz2 monomer FAILED: {e}")
                b2_mono_atoms = None
                row["b2_mono_plddt"] = float("nan")
                row["b2_mono_ptm"] = float("nan")

            # Boltz2 bound-unbound RMSD
            if b2_complex_atoms is not None and b2_mono_atoms is not None:
                try:
                    bound_ca = extract_ca_atoms(b2_complex_atoms)[:binder_len]
                    unbound_ca = extract_ca_atoms(b2_mono_atoms)[:binder_len]
                    row["b2_rmsd_bound_unbound"] = compute_rmsd(bound_ca, unbound_ca)
                except Exception as e:
                    print(f"    Boltz2 RMSD FAILED: {e}")
                    row["b2_rmsd_bound_unbound"] = float("nan")
            else:
                row["b2_rmsd_bound_unbound"] = float("nan")
        else:
            for k in ["b2_complex_plddt", "b2_complex_ptm", "b2_complex_iptm",
                       "b2_mono_plddt", "b2_mono_ptm", "b2_rmsd_bound_unbound"]:
                row[k] = float("nan")

        # --- AF2 ---
        if not args.skip_af2:
            print("  AF2 complex...", flush=True)
            try:
                af2_metrics = predict_af2_complex(
                    binder_seq, binder_len, name.replace("/", "_"), subdir
                )
                row["af2_complex_plddt"] = af2_metrics["binder_plddt"]
                row["af2_complex_ptm"] = af2_metrics["ptm"]
                row["af2_complex_iptm"] = af2_metrics["iptm"]
            except Exception as e:
                print(f"    AF2 FAILED: {e}")
                row["af2_complex_plddt"] = float("nan")
                row["af2_complex_ptm"] = float("nan")
                row["af2_complex_iptm"] = float("nan")
        else:
            for k in ["af2_complex_plddt", "af2_complex_ptm", "af2_complex_iptm"]:
                row[k] = float("nan")

        csv_rows.append(row)

        # Save intermediate CSV after each entry
        csv_path = OUTPUT_DIR / "evaluation_results.csv"
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"  Saved {len(csv_rows)} rows to {csv_path}")

    print(f"\n=== Done. {len(csv_rows)} sequences evaluated. ===")
    print(f"Results: {OUTPUT_DIR / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()
