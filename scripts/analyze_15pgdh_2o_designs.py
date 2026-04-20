"""Analyze the multi_oracle_2o_15pgdh_8way design campaigns.

Goal: score each generated sequence on interpretable raw metrics, explore how
many pass various plausibility thresholds, and return diverse short-lists per
campaign so we can pick ~20 designs for wet-lab testing.

Sign conventions (verified: sum of col * weight == total_energy):
  * iPTM_b2, ipSAE_b2, ipSAE_esm, local_pLDDT_b2, local_pLDDT_esm
      stored as -raw_metric    -> raw = -col
  * cross_PAE_b2, cross_PAE_esm, SolMPNN_perplexity_esm stored as raw
  * hydrophobic_esm, globular_esm, binder_length are penalty terms
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAMPAIGN_ROOT = Path("/mnt/disk2/ThinkingPLM/outputs/multi_oracle_2o_15pgdh_8way")

RAW_METRIC_FROM_NEG_COL = [
    ("ipSAE_b2", "ipsae_b2"),
    ("ipSAE_esm", "ipsae_esm"),
    ("iPTM_b2", "iptm_b2"),
    ("local_pLDDT_b2", "plddt_b2"),
    ("local_pLDDT_esm", "plddt_esm"),
]
RAW_METRIC_COPY = [
    ("cross_PAE_b2", "cross_pae_b2"),
    ("cross_PAE_esm", "cross_pae_esm"),
    ("SolMPNN_perplexity_esm", "solmpnn_ppx"),
    ("mean_plddt", "mean_plddt_esm"),
]


def load_campaign(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["campaign"] = csv_path.parent.name
    for src, dst in RAW_METRIC_FROM_NEG_COL:
        if src in df.columns:
            df[dst] = -df[src]
    for src, dst in RAW_METRIC_COPY:
        if src in df.columns:
            df[dst] = df[src]
    return df


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.glob("*/all_sequences.csv")):
        frames.append(load_campaign(csv_path))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["proposal_method"].fillna("") != "seed"].copy()
    df = df[df["sequence"].notna() & (df["sequence"].str.len() > 0)].copy()
    return df


def _kmers(seq: str, k: int = 6) -> set[str]:
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def identity(a: str, b: str) -> float:
    """Alignment-free sequence similarity via k-mer Jaccard.

    For 80-120 aa binder sequences, 6-mer Jaccard tracks global alignment
    identity tightly and is immune to length/insertion differences that
    break positional identity.
    """
    if not a or not b:
        return 0.0
    ka, kb = _kmers(a), _kmers(b)
    union = ka | kb
    if not union:
        return 0.0
    return len(ka & kb) / len(union)


@dataclass
class Filter:
    name: str
    predicate: callable  # type: ignore[valid-type]


def apply_filter_suite(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each filter level and return a dataframe with survivor flags.

    Note on thresholds:
      * cross_PAE columns are already normalised (raw_pae / 30), so 0.27 ~= 8 A.
      * ipsae_esm is strongly bimodal (many zeros when no pairs pass the
        PAE cutoff under ESMFold), so requiring it to be "high" is very harsh.
      * Boltz2 is the more reliable cofold here (trained on complexes),
        so we make the b2 signal the primary gate and use ESMFold metrics
        as soft consistency checks.
    """
    dual = (
        (df["iptm_b2"] >= 0.70)
        & (df["ipsae_b2"] >= 0.35)
        & (df["ipsae_esm"] >= 0.30)
        & (df["plddt_b2"] >= 0.75)
        & (df["plddt_esm"] >= 0.75)
        & (df["cross_pae_b2"] <= 0.27)   # raw <~8 A
        & (df["cross_pae_esm"] <= 0.30)
        & (df["solmpnn_ppx"] <= 3.5)
        & (df["length"] >= 55)
        & (df["length"] <= 120)
    )
    b2_primary = (
        (df["iptm_b2"] >= 0.65)
        & (df["ipsae_b2"] >= 0.30)
        & (df["plddt_b2"] >= 0.70)
        & (df["cross_pae_b2"] <= 0.33)   # raw <~10 A
        & (df["plddt_esm"] >= 0.65)
        & (df["ipsae_esm"] >= 0.10)      # soft gate: esm must at least register a contact
        & (df["solmpnn_ppx"] <= 4.0)
        & (df["length"] >= 55)
        & (df["length"] <= 120)
    )
    loose = (
        (df["iptm_b2"] >= 0.55)
        & (df["ipsae_b2"] >= 0.20)
        & (df["plddt_b2"] >= 0.65)
        & (df["cross_pae_b2"] <= 0.40)   # raw <~12 A
        & (df["plddt_esm"] >= 0.60)
        & (df["solmpnn_ppx"] <= 4.5)
        & (df["length"] >= 40)
        & (df["length"] <= 130)
    )
    exploratory = (
        (df["iptm_b2"] >= 0.50)
        & (df["ipsae_b2"] >= 0.15)
        & (df["plddt_b2"] >= 0.60)
        & (df["cross_pae_b2"] <= 0.50)
        & (df["plddt_esm"] >= 0.55)
        & (df["solmpnn_ppx"] <= 5.5)
        & (df["length"] >= 40)
        & (df["length"] <= 130)
    )
    out = df.copy()
    out["dual"] = dual
    out["b2_primary"] = b2_primary
    out["loose"] = loose
    out["exploratory"] = exploratory
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for camp, sub in df.groupby("campaign"):
        rows.append(
            {
                "campaign": camp,
                "n_rows": len(sub),
                "n_dual": int(sub["dual"].sum()),
                "n_b2": int(sub["b2_primary"].sum()),
                "n_loose": int(sub["loose"].sum()),
                "n_explore": int(sub["exploratory"].sum()),
                "best_total_energy": sub["total_energy"].min(),
                "best_iptm_b2": sub["iptm_b2"].max(),
                "best_ipsae_b2": sub["ipsae_b2"].max(),
                "best_ipsae_esm": sub["ipsae_esm"].max(),
                "best_plddt_b2": sub["plddt_b2"].max(),
                "min_cross_pae_b2": sub["cross_pae_b2"].min(),
                "min_solmpnn_ppx": sub["solmpnn_ppx"].min(),
            }
        )
    return pd.DataFrame(rows).sort_values("n_b2", ascending=False)


def distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "iptm_b2",
        "ipsae_b2",
        "ipsae_esm",
        "plddt_b2",
        "plddt_esm",
        "cross_pae_b2",
        "cross_pae_esm",
        "solmpnn_ppx",
        "total_energy",
    ]
    return df[cols].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T


def greedy_diverse_pick(
    candidates: pd.DataFrame,
    max_picks: int,
    identity_cap: float,
) -> pd.DataFrame:
    """Greedy selection: best ranking subject to a Jaccard-similarity cap.

    identity_cap ~0.45 filters sequences sharing more than ~85% identity.
    identity_cap ~0.25 filters anything above ~75% identity.
    """
    if candidates.empty:
        return candidates
    ranked = candidates.sort_values("composite_rank").reset_index(drop=True)
    picked_rows: list[int] = []
    picked_seqs: list[str] = []
    picked_kmers: list[set[str]] = []
    for i, row in ranked.iterrows():
        seq = row["sequence"]
        k = _kmers(seq)
        skip = False
        for pk in picked_kmers:
            union = k | pk
            if union and len(k & pk) / len(union) > identity_cap:
                skip = True
                break
        if skip:
            continue
        picked_rows.append(i)
        picked_seqs.append(seq)
        picked_kmers.append(k)
        if len(picked_rows) >= max_picks:
            break
    return ranked.iloc[picked_rows].reset_index(drop=True)


def cluster_count(candidates: pd.DataFrame, identity_cap: float) -> int:
    """Return number of sequence clusters (single-linkage under Jaccard cap)."""
    seqs = candidates["sequence"].tolist()
    if not seqs:
        return 0
    kmers = [_kmers(s) for s in seqs]
    n = len(seqs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            u = kmers[i] | kmers[j]
            if u and len(kmers[i] & kmers[j]) / len(u) > identity_cap:
                union(i, j)
    return len({find(i) for i in range(n)})


def add_composite_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Combine key metrics into a single ranking score (lower = better)."""
    out = df.copy()
    # Normalize each metric to [0,1] range where lower=better by using ranks.
    terms = {
        "iptm_b2": False,        # higher better -> rank descending
        "ipsae_b2": False,
        "ipsae_esm": False,
        "plddt_b2": False,
        "plddt_esm": False,
        "cross_pae_b2": True,    # lower better
        "cross_pae_esm": True,
        "solmpnn_ppx": True,
    }
    rank_frame = pd.DataFrame(index=out.index)
    for col, lower_better in terms.items():
        rank_frame[col] = out[col].rank(ascending=lower_better, pct=True)
    out["composite_rank"] = rank_frame.mean(axis=1)
    return out


def select_per_campaign(
    df: pd.DataFrame,
    tier: str,
    max_per_campaign: int,
    identity_cap: float,
) -> pd.DataFrame:
    df = add_composite_rank(df)
    picks = []
    for camp, sub in df.groupby("campaign"):
        survivors = sub[sub[tier]]
        if survivors.empty:
            continue
        chosen = greedy_diverse_pick(survivors, max_per_campaign, identity_cap)
        picks.append(chosen.assign(tier_used=tier))
    if not picks:
        return pd.DataFrame()
    return pd.concat(picks, ignore_index=True)


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cols = [
        ("iptm_b2", 0.65, "higher"),
        ("ipsae_b2", 0.30, "higher"),
        ("ipsae_esm", 0.30, "higher"),
        ("plddt_b2", 0.70, "higher"),
        ("plddt_esm", 0.70, "higher"),
        ("cross_pae_b2", 0.33, "lower"),
        ("cross_pae_esm", 0.33, "lower"),
        ("solmpnn_ppx", 4.0, "lower"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (col, thr, direction) in zip(axes.flat, cols):
        ax.hist(df[col].dropna(), bins=60, color="#4C72B0", alpha=0.85)
        ax.axvline(thr, color="crimson", ls="--", lw=1.5, label=f"thr={thr} ({direction})")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Raw metric distributions — multi_oracle_2o_15pgdh_8way", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=120)
    plt.close(fig)


def build_slate_of_20(
    df: pd.DataFrame,
    target_n: int = 20,
    cluster_cap: float = 0.35,
) -> pd.DataFrame:
    """Assemble a final slate of ~target_n designs.

    Strategy (after extra ESMFold sanity gate on ipsae_esm >= 0.3):
      1. Cluster the exploratory pool by k-mer Jaccard > cluster_cap.
      2. First fill with loose-tier cluster leaders (high confidence).
      3. Then add exploratory-only cluster leaders (breadth).
      4. Finally add sibling picks from the most populous strong clusters
         (hedge against single-design prediction noise).
    """
    pool = df[df["exploratory"] & (df["ipsae_esm"] >= 0.30)].copy()
    if pool.empty:
        return pool
    pool = add_composite_rank(pool)
    pool = pool.sort_values("composite_rank").reset_index(drop=True)

    seqs = pool["sequence"].tolist()
    kmers = [_kmers(s) for s in seqs]
    n = len(seqs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            u = kmers[i] | kmers[j]
            if u and len(kmers[i] & kmers[j]) / len(u) > cluster_cap:
                pa, pb = find(i), find(j)
                if pa != pb:
                    parent[pa] = pb

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    cluster_leaders = sorted(
        clusters.items(), key=lambda kv: pool.loc[kv[1][0], "composite_rank"]
    )

    picks: list[int] = []
    cluster_ids: list[int] = []
    roles: list[str] = []

    # Pass 1 — loose-tier cluster leaders
    for root, members in cluster_leaders:
        if pool.loc[members[0], "loose"]:
            picks.append(members[0])
            cluster_ids.append(root)
            roles.append("loose-leader")
            if len(picks) >= target_n:
                break

    # Pass 2 — exploratory-only cluster leaders
    if len(picks) < target_n:
        for root, members in cluster_leaders:
            if members[0] in picks:
                continue
            picks.append(members[0])
            cluster_ids.append(root)
            roles.append("explore-leader")
            if len(picks) >= target_n:
                break

    # Pass 3 — siblings from the largest strong clusters (hedge)
    if len(picks) < target_n:
        strong_clusters = [
            (root, members)
            for root, members in cluster_leaders
            if pool.loc[members[0], "loose"] and len(members) > 1
        ]
        strong_clusters.sort(key=lambda kv: -len(kv[1]))
        idx_in_cluster = {root: 1 for root, _ in strong_clusters}
        while len(picks) < target_n and strong_clusters:
            added = False
            for root, members in strong_clusters:
                k = idx_in_cluster[root]
                if k >= len(members):
                    continue
                picks.append(members[k])
                cluster_ids.append(root)
                roles.append("hedge-sibling")
                idx_in_cluster[root] = k + 1
                added = True
                if len(picks) >= target_n:
                    break
            if not added:
                break

    result = pool.iloc[picks].copy().reset_index(drop=True)
    result["cluster_id"] = cluster_ids
    result["role"] = roles
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=CAMPAIGN_ROOT)
    ap.add_argument("--out", type=Path, default=Path("/mnt/disk2/ThinkingPLM/analysis_15pgdh_2o"))
    ap.add_argument("--max-per-campaign", type=int, default=6)
    ap.add_argument("--identity-cap", type=float, default=0.35)
    ap.add_argument("--tier", choices=["dual", "b2_primary", "loose", "exploratory"], default="loose")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading campaigns from {args.root}")
    df = load_all(args.root)
    print(f"Loaded {len(df)} rows across {df['campaign'].nunique()} campaigns")

    df = apply_filter_suite(df)

    plot_distributions(df, args.out)

    dist = distribution_report(df)
    print("\n=== Metric distribution (pooled across campaigns) ===")
    print(dist.to_string(float_format=lambda v: f"{v:.3f}"))
    dist.to_csv(args.out / "distributions.csv")

    summary = summary_table(df)
    print("\n=== Per-campaign survivor counts ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    summary.to_csv(args.out / "campaign_summary.csv", index=False)

    print("\n=== Cluster counts in survivor pool (per campaign, Jaccard > 0.35) ===")
    cluster_rows = []
    for camp, sub in df.groupby("campaign"):
        cluster_rows.append(
            {
                "campaign": camp,
                "dual_clusters": cluster_count(sub[sub["dual"]], args.identity_cap),
                "b2_clusters": cluster_count(sub[sub["b2_primary"]], args.identity_cap),
                "loose_clusters": cluster_count(sub[sub["loose"]], args.identity_cap),
                "explore_clusters": cluster_count(sub[sub["exploratory"]], args.identity_cap),
            }
        )
    cluster_df = pd.DataFrame(cluster_rows).sort_values("explore_clusters", ascending=False)
    print(cluster_df.to_string(index=False))
    cluster_df.to_csv(args.out / "cluster_counts.csv", index=False)

    print("\n=== Picks per campaign by tier ===")
    all_picks = {}
    for tier in ("dual", "b2_primary", "loose", "exploratory"):
        picks = select_per_campaign(
            df,
            tier=tier,
            max_per_campaign=args.max_per_campaign,
            identity_cap=args.identity_cap,
        )
        all_picks[tier] = picks
        n_camps = 0 if picks.empty else picks["campaign"].nunique()
        print(f"  {tier:8s}: {len(picks):3d} picks across {n_camps} campaigns")
        if not picks.empty:
            picks.to_csv(args.out / f"picks_{tier}.csv", index=False)

    final_tier = args.tier
    chosen = all_picks[final_tier]
    if chosen.empty:
        print(f"No picks survive tier '{final_tier}'")
    else:
        cols_display = [
            "campaign",
            "cycle",
            "proposal_method",
            "sequence",
            "length",
            "iptm_b2",
            "ipsae_b2",
            "ipsae_esm",
            "plddt_b2",
            "plddt_esm",
            "cross_pae_b2",
            "cross_pae_esm",
            "solmpnn_ppx",
            "total_energy",
            "composite_rank",
        ]
        chosen[cols_display].to_csv(args.out / "final_shortlist.csv", index=False)
        print(f"\nFinal short-list ({final_tier}) -> {args.out / 'final_shortlist.csv'}")
        print(chosen[cols_display].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== 20-design slate (cluster leaders + siblings of top cluster) ===")
    slate = build_slate_of_20(df, target_n=20, cluster_cap=args.identity_cap)
    if slate.empty:
        print("No exploratory survivors — no slate")
    else:
        cols_slate = [
            "cluster_id",
            "role",
            "campaign",
            "cycle",
            "length",
            "iptm_b2",
            "ipsae_b2",
            "ipsae_esm",
            "plddt_b2",
            "plddt_esm",
            "cross_pae_b2",
            "cross_pae_esm",
            "solmpnn_ppx",
            "total_energy",
            "composite_rank",
            "sequence",
        ]
        slate.to_csv(args.out / "slate_20.csv", index=False)
        print(slate[cols_slate].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nSlate -> {args.out / 'slate_20.csv'}")


if __name__ == "__main__":
    main()
