#!/usr/bin/env python3
"""Sample sequences per target for Boltz2 correlation study.

For each target, pool all_sequences.csv across runs, dedup, and sample:
  - 15 from the top 20% by lowest total_energy
  - 15 random from the remainder (deduped against the top selections)

Writes:
  - boltz2_corr_input/<target>_<idx>_<sampletype>.yaml  (one per sequence)
  - boltz2_corr_input/manifest.json                      (metadata for plotting)
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path("/mnt/disk2/ThinkingPLM/outputs")
OUT_DIR = Path("/mnt/disk2/ThinkingPLM/boltz2_corr_input")

# Target sequences (same as in extract_best_and_prepare_colabfold.py)
TARGETS = {
    "2GDZ": "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQKTLFIQCDVADQQQLRDTFRKVVDHFGRLDILVNNAGVNNEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLMNSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLITLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ",
    "1TNF_TNF_alpha": "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLLFAESGQVYFGIIAL",
    "1YCR_MDM2": "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV",
    "2VSM_nipah": "ICLQKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH",
    "4OYD_epstein_barr": "SAYSTREILLALCIRDSRVHGNGTLHPVLELAARETPLRLSPEDTVVLRYHVLLEEIIERNSETFTETWNRFITHTEHVDLDFNSVFLEIFHRGDPSLGRALAWMAWCMHACRTLCCNQSTPYYVVDLSVRGMLEASEGLDGWIHQQGGWSTLIEDNI",
    "4ZQK_PD-L1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
}

N_RANDOM = 15
N_TOP = 15
TOP_FRACTION = 0.20
SEED = 42


def get_csv_paths(target_id):
    if target_id == "2GDZ":
        return sorted((BASE / "scaffold_comparison_rep3").rglob("all_sequences.csv"))
    return sorted((BASE / "mt2_bench" / target_id).rglob("all_sequences.csv"))


def pool_sequences(target_id):
    """Load all (sequence, total_energy) across runs, keeping best-energy occurrence."""
    best = {}  # sequence -> (energy, source_run)
    for csv_path in get_csv_paths(target_id):
        rel_run = csv_path.parent.relative_to(BASE)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    energy = float(row["total_energy"])
                except (ValueError, TypeError):
                    continue
                seq = row["sequence"]
                if not seq:
                    continue
                prev = best.get(seq)
                if prev is None or energy < prev[0]:
                    best[seq] = (energy, str(rel_run), row.get("cycle", ""))
    return best


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    manifest = []

    for target_id in TARGETS:
        pool = pool_sequences(target_id)
        if not pool:
            print(f"WARNING: no sequences for {target_id}")
            continue

        items = [(seq, e, src, cyc) for seq, (e, src, cyc) in pool.items()]
        items.sort(key=lambda x: x[1])  # ascending energy
        n = len(items)
        n_top = max(1, int(round(n * TOP_FRACTION)))

        top_pool = items[:n_top]
        # 15 from top 20%
        top_picks = rng.sample(top_pool, min(N_TOP, len(top_pool)))
        top_seqs = {t[0] for t in top_picks}

        # 15 random from entire pool, excluding top selections
        remaining = [it for it in items if it[0] not in top_seqs]
        rand_picks = rng.sample(remaining, min(N_RANDOM, len(remaining)))

        print(f"\n=== {target_id} ===")
        print(f"  pool size (unique): {n}")
        print(f"  top 20% cutoff index: {n_top} (energy <= {items[n_top-1][1]:.4f})")
        print(f"  picked top: {len(top_picks)}, picked random: {len(rand_picks)}")

        # Write YAML + manifest entries
        all_picks = [
            (pick, "top20") for pick in top_picks
        ] + [
            (pick, "random") for pick in rand_picks
        ]
        for idx, ((seq, energy, src, cyc), stype) in enumerate(all_picks):
            name = f"{target_id}_{stype}_{idx:02d}"
            yaml = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
      msa: empty
  - protein:
      id: B
      sequence: {TARGETS[target_id]}
      msa: empty
"""
            (OUT_DIR / f"{name}.yaml").write_text(yaml)
            manifest.append({
                "name": name,
                "target_id": target_id,
                "sample_type": stype,
                "binder_seq": seq,
                "binder_len": len(seq),
                "total_energy": energy,
                "source_run": src,
                "cycle": cyc,
            })

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal inputs: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"YAMLs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
