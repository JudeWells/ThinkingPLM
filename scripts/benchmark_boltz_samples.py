"""Benchmark Boltz2 fold() wall-clock vs diffusion_samples.

Runs the Boltz oracle on the hairpin binder + 2GDZ target for a range of
``diffusion_samples`` values, one subprocess per value (so each run has a
clean warm state), and prints a comparison table.

Usage:
    python scripts/benchmark_boltz_samples.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import bagel as bg
from bagel.chain import Residue
from bagel.oracles.folding import Boltz


# Hairpin binder (82 residues, same as configs/sequences/initial_sequence_hairpin.fasta)
HAIRPIN_BINDER = (
    "NEKNWEKTLQINLVSVISGTYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVY"
    "CASKHGIVGFTRSAALAANLMNSGVR"
)

# 2GDZ (15-PGDH) target — 267 residues, matches the energy config's `target:` kwarg
TARGET_2GDZ = (
    "MAHMVNGKVALVTGAAQGIGRAFAEALLLKGAKVALVDWNLEAGVQCKAALHEQFEPQ"
    "KTLFIQCDVADQQQLRDTFRKVVDHFGRLDILVNNAGVNNEKNWEKTLQINLVSVISG"
    "TYLGLDYMSKQNGGEGGIIINMSSLAGLMPVAQQPVYCASKHGIVGFTRSAALAANLM"
    "NSGVRLNAICPGFVNTAILESIEKEENMGQYIEYKDHIKDMIKYYGILDPPLIANGLI"
    "TLIEDDALNGAIMKITTSKGIHFQDYGSKENLYFQ"
)


def make_chain(seq: str, chain_id: str) -> bg.Chain:
    residues = [
        Residue(name=aa, chain_ID=chain_id, index=i, mutable=False)
        for i, aa in enumerate(seq)
    ]
    return bg.Chain(residues=residues)


def run_one(diffusion_samples: int) -> float:
    """Fold the hairpin+2GDZ complex once, return wall-clock seconds."""
    oracle = Boltz(diffusion_samples=diffusion_samples)
    binder = make_chain(HAIRPIN_BINDER, "GEN")
    target = make_chain(TARGET_2GDZ, "B")

    t0 = time.monotonic()
    result = oracle.fold([binder, target])
    elapsed = time.monotonic() - t0
    assert result is not None and hasattr(result, "structure")
    return elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20],
        help="Values of diffusion_samples to test (default: 1 2 5 10 20)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per sample count (default: 1)",
    )
    args = p.parse_args()

    print(f"Boltz2 diffusion_samples benchmark")
    print(f"binder length: {len(HAIRPIN_BINDER)}  target length: {len(TARGET_2GDZ)}")
    print(f"sample counts: {args.samples}  repeats: {args.repeats}")
    print()

    results: dict[int, list[float]] = {n: [] for n in args.samples}
    for n in args.samples:
        for r in range(args.repeats):
            print(f"  running diffusion_samples={n} (repeat {r + 1}/{args.repeats}) ...", flush=True)
            try:
                t = run_one(n)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                continue
            results[n].append(t)
            print(f"    done in {t:.1f} s", flush=True)

    print()
    print(f"{'diffusion_samples':>18s}  {'mean (s)':>10s}  {'per-sample (s)':>16s}  {'vs n=1':>10s}")
    baseline = None
    for n in sorted(results.keys()):
        times = results[n]
        if not times:
            continue
        mean = sum(times) / len(times)
        per_sample = mean / n
        if baseline is None:
            baseline = mean
        ratio = mean / baseline
        print(f"{n:>18d}  {mean:>10.1f}  {per_sample:>16.1f}  {ratio:>10.2f}x")


if __name__ == "__main__":
    main()
