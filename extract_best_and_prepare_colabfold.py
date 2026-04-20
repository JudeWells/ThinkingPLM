#!/usr/bin/env python3
"""Extract best sequences from completed campaigns and prepare ColabFold input."""

import csv
import json
import os
from pathlib import Path

BASE = Path("/mnt/disk2/ThinkingPLM/outputs")

# Target sequences by target ID
TARGETS = {
    "2GDZ": "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQKTLFIQCDVADQQQLRDTFRKVVDHFGRLDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ",
    "1TNF_TNF_alpha": "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLLFAESGQVYFGIIAL",
    "1YCR_MDM2": "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV",
    "2VSM_nipah": "ICLQKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH",
    "4OYD_epstein_barr": "SAYSTREILLALCIRDSRVHGNGTLHPVLELAARETPLRLSPEDTVVLRYHVLLEEIIERNSETFTETWNRFITHTEHVDLDFNSVFLEIFHRGDPSLGRALAWMAWCMHACRTLCCNQSTPYYVVDLSVRGMLEASEGLDGWIHQQGGWSTLIEDNI",
    "4ZQK_PD-L1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
}

# Minimum line count to consider a run "complete" (5400 cycles + header + seed = 5402)
MIN_LINES = 5400


def get_best_from_csv(csv_path):
    """Return the row with minimum total_energy from a CSV."""
    best = None
    best_energy = float("inf")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                energy = float(row["total_energy"])
            except (ValueError, KeyError):
                continue
            if energy < best_energy:
                best_energy = energy
                best = row
    return best


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def scan_runs(base_dir, target_id):
    """Scan a base directory for completed runs and extract best sequences."""
    results = []
    if not base_dir.exists():
        return results
    for csv_path in sorted(base_dir.rglob("all_sequences.csv")):
        n_lines = count_lines(csv_path)
        if n_lines < MIN_LINES:
            continue
        rel = csv_path.relative_to(base_dir)
        parts = list(rel.parts[:-1])  # drop "all_sequences.csv"
        best = get_best_from_csv(csv_path)
        if best is None:
            continue
        results.append({
            "campaign": "/".join(parts),
            "target_id": target_id,
            "target_seq": TARGETS[target_id],
            "binder_seq": best["sequence"],
            "total_energy": float(best["total_energy"]),
            "lis_energy": float(best.get("LIS", 0)),
            "plddt_energy": float(best.get("local_pLDDT", 0)),
            "mean_plddt": float(best["mean_plddt"]) if best.get("mean_plddt") else None,
            "iptm": float(best["iptm"]) if best.get("iptm") else None,
            "ptm": float(best["ptm"]) if best.get("ptm") else None,
            "cycle": int(best["cycle"]),
            "csv_path": str(csv_path),
        })
    return results


def main():
    all_results = []

    # Scaffold comparison rep3 (target = 2GDZ)
    sc_rep3 = BASE / "scaffold_comparison_rep3"
    all_results.extend(scan_runs(sc_rep3, "2GDZ"))

    # MT2 bench (multiple targets)
    mt2 = BASE / "mt2_bench"
    if mt2.exists():
        for target_dir in sorted(mt2.iterdir()):
            if not target_dir.is_dir():
                continue
            target_id = target_dir.name
            if target_id not in TARGETS:
                print(f"WARNING: Unknown target {target_id}, skipping")
                continue
            all_results.extend(scan_runs(target_dir, target_id))

    # Sort by target then campaign
    all_results.sort(key=lambda r: (r["target_id"], r["campaign"]))

    # Print summary
    print(f"\n{'Campaign':<55} {'Target':<20} {'Energy':>8} {'LIS':>8} {'pLDDT':>6} {'Cycle':>6}")
    print("-" * 110)
    for r in all_results:
        plddt_str = f"{r['mean_plddt']:.3f}" if r['mean_plddt'] else "N/A"
        print(f"{r['campaign']:<55} {r['target_id']:<20} {r['total_energy']:>8.4f} {r['lis_energy']:>8.4f} {plddt_str:>6} {r['cycle']:>6}")

    # Create ColabFold input directory
    cf_dir = BASE.parent / "colabfold_input"
    cf_dir.mkdir(exist_ok=True)

    # Write one FASTA per complex (binder:target format for multimer)
    for r in all_results:
        # Create a safe filename
        safe_name = r["campaign"].replace("/", "_")
        if r["target_id"] != "2GDZ":
            safe_name = f"{r['target_id']}_{safe_name}"
        else:
            safe_name = f"sc_rep3_{safe_name}"
        fasta_path = cf_dir / f"{safe_name}.fasta"
        # ColabFold multimer: sequences separated by ":"
        with open(fasta_path, "w") as f:
            f.write(f">{safe_name}\n")
            f.write(f"{r['binder_seq']}:{r['target_seq']}\n")

    # Save results JSON for HTML builder
    results_json = BASE.parent / "colabfold_input" / "best_sequences.json"
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{len(all_results)} ColabFold input FASTAs written to {cf_dir}")
    print(f"Results JSON: {results_json}")


if __name__ == "__main__":
    main()
