"""Logging and persistence helpers for the ProFam + BAGEL pipeline.

Contains cycle log management, CSV export, and structure saving.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from pipeline.utils import _pairwise_identity


def update_cycle_log(
    log_path: Path,
    cycle_index: int,
    selected_indices: np.ndarray,
    energies: Sequence[float],
    sequence_details: Sequence[Dict[str, Any]],
    avg_similarity: float | None = None,
    avg_similarity_to_prompt: float | None = None,
    global_ids: Sequence[int] | None = None,
    pool_ids: Sequence[int] | None = None,
    pool_energies: Sequence[float] | None = None,
    pool_names: Sequence[str] | None = None,
    pool_seqs: Sequence[str] | None = None,
    swap_accepted: bool | None = None,
    swap_reason: str | None = None,
    elite_energy: float | None = None,
    elite_cycle: int | None = None,
    annealing_temp: float | None = None,
    thompson_state: Dict[str, Any] | None = None,
    thompson_selected_arm_id: int | None = None,
    thompson_progeny_reward: float | None = None,
    proposal_method: str | None = None,
) -> None:
    """Append / update a JSON log keyed by cycle index.

    Parameters
    ----------
    global_ids : list of int, optional
        Global unique IDs for the current cycle's generated sequences.
    pool_ids : list of int, optional
        Global IDs for the full selection pool (memory + current cycle).
    pool_energies : list of float, optional
        Energies for the full selection pool (parallel to ``pool_ids``).
    pool_names : list of str, optional
        Names for the full selection pool (parallel to ``pool_ids``).
    pool_seqs : list of str, optional
        Sequences for the full selection pool (parallel to ``pool_ids``).
    """
    if log_path.is_file():
        with log_path.open("r") as f:
            log_data = json.load(f)
    else:
        log_data = {}

    def _json_safe(v: float) -> float:
        return 1e30 if (math.isinf(v) or math.isnan(v)) else v

    if pool_energies is not None:
        sel_energies = [_json_safe(float(pool_energies[int(i)])) for i in selected_indices]
    else:
        sel_energies = [_json_safe(float(energies[int(i)])) for i in selected_indices]
    avg_energy = _json_safe(float(np.mean(sel_energies)))
    min_energy = _json_safe(float(np.min(sel_energies)))

    selected_sequences: List[Dict[str, Any]] = []
    if pool_ids is not None:
        pool_offset = len(pool_ids) - len(energies)
        for i in selected_indices:
            idx = int(i)
            gid = pool_ids[idx]
            if idx >= pool_offset:
                local_idx = idx - pool_offset
                entry = dict(sequence_details[local_idx])
                entry["energy"] = _json_safe(float(entry.get("energy", 0.0)))
                if "energy_terms" in entry:
                    entry["energy_terms"] = {
                        k: _json_safe(float(v)) for k, v in entry["energy_terms"].items()
                    }
            else:
                entry: Dict[str, Any] = {"energy": _json_safe(float(pool_energies[idx]))}  # type: ignore[index]
                if pool_seqs is not None:
                    entry["sequence"] = pool_seqs[idx]
            entry["id"] = gid
            selected_sequences.append(entry)
    else:
        for i in selected_indices:
            entry = dict(sequence_details[int(i)])
            entry["energy"] = _json_safe(float(entry.get("energy", 0.0)))
            if "energy_terms" in entry:
                entry["energy_terms"] = {
                    k: _json_safe(float(v)) for k, v in entry["energy_terms"].items()
                }
            if global_ids is not None:
                entry["id"] = global_ids[int(i)]
            selected_sequences.append(entry)

    all_energies_safe = [_json_safe(float(e)) for e in energies]
    all_avg_energy = _json_safe(float(np.mean(all_energies_safe)))
    all_min_energy = _json_safe(float(np.min(all_energies_safe)))

    best_idx = int(np.argmin(all_energies_safe))
    best_entry = dict(sequence_details[best_idx])
    best_entry["energy"] = _json_safe(float(best_entry.get("energy", 0.0)))
    if "energy_terms" in best_entry:
        best_entry["energy_terms"] = {
            k: _json_safe(float(v)) for k, v in best_entry["energy_terms"].items()
        }
    if global_ids is not None:
        best_entry["id"] = global_ids[best_idx]

    if pool_ids is not None:
        selected_id_list = [pool_ids[int(i)] for i in selected_indices]
    elif global_ids is not None:
        selected_id_list = [global_ids[int(i)] for i in selected_indices]
    else:
        selected_id_list = [int(i) for i in selected_indices]

    cycle_entry: Dict[str, Any] = {
        "cycle": cycle_index,
        "num_generated": len(energies),
        "all_avg_energy": all_avg_energy,
        "all_min_energy": all_min_energy,
        "best_sequence": best_entry,
        "num_selected": len(selected_indices),
        "selected_avg_energy": avg_energy,
        "selected_min_energy": min_energy,
        "selected_ids": selected_id_list,
        "selected_sequences": selected_sequences,
    }
    if pool_ids is not None:
        cycle_entry["pool_size"] = len(pool_ids)
    if avg_similarity is not None:
        cycle_entry["all_avg_similarity"] = avg_similarity
    if avg_similarity_to_prompt is not None:
        cycle_entry["all_avg_similarity_to_prompt"] = avg_similarity_to_prompt
    if swap_accepted is not None:
        cycle_entry["swap_accepted"] = swap_accepted
    if swap_reason is not None:
        cycle_entry["swap_reason"] = swap_reason
    if elite_energy is not None:
        cycle_entry["global_elite"] = {
            "energy": _json_safe(elite_energy),
            "from_cycle": elite_cycle,
        }
    if annealing_temp is not None:
        cycle_entry["annealing_temp"] = annealing_temp
    if proposal_method is not None:
        cycle_entry["proposal_method"] = proposal_method
    if thompson_selected_arm_id is not None:
        cycle_entry["thompson_selected_arm_id"] = thompson_selected_arm_id
    if thompson_progeny_reward is not None:
        cycle_entry["thompson_progeny_reward"] = thompson_progeny_reward
    if thompson_state is not None:
        cycle_entry["thompson_num_arms"] = thompson_state.get("num_arms", 0)

    log_data[str(cycle_index)] = cycle_entry

    with log_path.open("w") as f:
        json.dump(log_data, f, indent=2)


def save_selected_structures(
    cycle_index: int,
    selected_indices: np.ndarray,
    folding_results: Sequence[Any],
    output_dir: Path,
    pool_offset: int = 0,
) -> None:
    """Save CIF structures for the selected subset into `sequences_cycle_<cycle>`.

    Parameters
    ----------
    pool_offset : int
        When n_memory > 0, ``selected_indices`` index into the combined pool.
        ``pool_offset`` is the index at which the current cycle's sequences
        start.  Only current-cycle sequences have folding results available.
    """
    if not any(fr is not None for fr in folding_results):
        return

    seq_dir = output_dir / f"sequences_cycle_{cycle_index}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    for out_idx, seq_idx in enumerate(selected_indices):
        idx = int(seq_idx)
        if idx < pool_offset:
            continue
        fr = folding_results[idx - pool_offset]
        if fr is not None:
            cif_path = seq_dir / f"sequence_{out_idx:04d}.cif"
            fr.to_cif(cif_path)
            if hasattr(fr, "save_attributes"):
                fr.save_attributes(seq_dir / f"sequence_{out_idx:04d}")


def append_cycle_csv(
    csv_path: Path,
    cycle_index: int,
    names: Sequence[str],
    sequences: Sequence[str],
    energies: Sequence[float],
    details: Sequence[Dict[str, Any]],
    folding_results: Sequence[Any],
    initial_seqs: Sequence[str],
    prompt_seqs: Sequence[str] | None = None,
    proposal_method: str | None = None,
) -> None:
    """Append one row per generated sequence to the all_sequences.csv file."""
    import csv

    write_header = not csv_path.is_file()

    energy_term_keys: List[str] = []
    for d in details:
        if isinstance(d, dict) and "energy_terms" in d:
            energy_term_keys = sorted(d["energy_terms"].keys())
            break

    fieldnames = [
        "cycle", "proposal_method", "name", "sequence", "length", "total_energy",
    ] + energy_term_keys + [
        "ptm", "mean_plddt", "iptm", "similarity_to_initial", "similarity_to_prompt",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx in range(len(sequences)):
            seq = sequences[idx]
            term_values: Dict[str, float] = {}
            if idx < len(details) and isinstance(details[idx], dict):
                term_values = details[idx].get("energy_terms", {})

            ptm_val = ""
            plddt_val = ""
            iptm_val = ""
            fr = folding_results[idx] if idx < len(folding_results) else None
            if fr is not None:
                try:
                    ptm_val = float(fr.ptm[0])
                except Exception:
                    pass
                try:
                    plddt_val = float(fr.local_plddt[0].mean())
                except Exception:
                    pass
                try:
                    iptm_val = float(fr.chain_pair_iptm[0, 1])
                except Exception:
                    pass

            sim = max(_pairwise_identity(seq, init_s) for init_s in initial_seqs) if initial_seqs else ""
            sim_prompt = max(_pairwise_identity(seq, ps) for ps in prompt_seqs) if prompt_seqs else ""

            row: Dict[str, Any] = {
                "cycle": cycle_index,
                "proposal_method": proposal_method or "",
                "name": names[idx] if idx < len(names) else "",
                "sequence": seq,
                "length": len(seq),
                "total_energy": energies[idx] if idx < len(energies) else "",
            }
            for k in energy_term_keys:
                row[k] = term_values.get(k, "")
            row["ptm"] = ptm_val
            row["mean_plddt"] = plddt_val
            row["iptm"] = iptm_val
            row["similarity_to_initial"] = sim
            row["similarity_to_prompt"] = sim_prompt

            writer.writerow(row)
