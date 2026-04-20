#!/usr/bin/env python3
"""Build HTML results viewer with 3Dmol.js visualization of ColabFold predictions.

Reads best_sequences.json + ColabFold output PDBs/scores to produce a single
self-contained HTML file with interactive 3D structure viewers and metrics.
"""

import json
import os
import glob
import numpy as np
from pathlib import Path

BASE = Path("/mnt/disk2/ThinkingPLM")
CF_OUTPUT = BASE / "colabfold_output"
CF_INPUT = BASE / "colabfold_input"
HTML_OUTPUT = BASE / "results_viewer.html"


def load_best_sequences():
    with open(CF_INPUT / "best_sequences.json") as f:
        return json.load(f)


def get_cf_name(result):
    """Build the ColabFold output name from campaign info."""
    safe_name = result["campaign"].replace("/", "_")
    if result["target_id"] != "2GDZ":
        return f"{result['target_id']}_{safe_name}"
    else:
        return f"sc_rep3_{safe_name}"


def find_pdb(cf_name):
    """Find the unrelaxed PDB file for a given ColabFold name."""
    pattern = str(CF_OUTPUT / f"{cf_name}_unrelaxed_rank_001_*.pdb")
    matches = glob.glob(pattern)
    if not matches:
        # Try relaxed
        pattern = str(CF_OUTPUT / f"{cf_name}_relaxed_rank_001_*.pdb")
        matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_scores(cf_name):
    """Find the scores JSON file."""
    pattern = str(CF_OUTPUT / f"{cf_name}_scores_rank_001_*.json")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def compute_metrics(scores_path, binder_len):
    """Compute binder pLDDT and ipSAE from ColabFold scores."""
    with open(scores_path) as f:
        scores = json.load(f)

    plddt = scores["plddt"]
    pae = np.array(scores["pae"])
    total_len = len(plddt)
    target_len = total_len - binder_len

    # Binder pLDDT (chain A = first binder_len residues)
    binder_plddt = np.mean(plddt[:binder_len])

    # ipSAE: mean PAE across the interface (binder->target and target->binder)
    # PAE[i][j] = predicted error of residue j's position when aligned on residue i
    pae_binder_to_target = pae[:binder_len, binder_len:]  # binder rows, target cols
    pae_target_to_binder = pae[binder_len:, :binder_len]  # target rows, binder cols
    ipsae = np.mean([pae_binder_to_target.mean(), pae_target_to_binder.mean()])

    return {
        "binder_plddt": float(binder_plddt),
        "target_plddt": float(np.mean(plddt[binder_len:])),
        "overall_plddt": float(np.mean(plddt)),
        "ipsae": float(ipsae),
        "ptm": scores.get("ptm"),
        "iptm": scores.get("iptm"),
        "max_pae": scores.get("max_pae"),
    }


def read_pdb(pdb_path):
    """Read PDB file contents."""
    with open(pdb_path) as f:
        return f.read()


def build_html(results_data):
    """Build the HTML page with all results."""

    # Group by target
    by_target = {}
    for entry in results_data:
        tid = entry["target_id"]
        by_target.setdefault(tid, []).append(entry)

    # Sort within each target by ipsae (lower is better)
    for tid in by_target:
        by_target[tid].sort(key=lambda x: x.get("cf_ipsae", 999))

    # Build cards HTML
    cards = []
    viewer_id = 0

    for tid in sorted(by_target.keys()):
        entries = by_target[tid]
        cards.append(f'<h2 class="target-header">Target: {tid}</h2>')
        cards.append('<div class="cards-grid">')

        for entry in entries:
            vid = f"viewer_{viewer_id}"
            viewer_id += 1

            has_structure = entry.get("pdb_data") is not None
            campaign = entry["campaign"]

            # Determine method for badge color
            method = campaign.split("/")[-1] if "/" in campaign else campaign
            badge_class = {
                "bandit_grpo": "badge-grpo",
                "bandit_bt": "badge-bt",
                "proposal_bandit": "badge-bandit",
                "random_greedy": "badge-greedy",
            }.get(method, "badge-default")

            scaffold = campaign.split("/")[0] if "/" in campaign else "unknown"

            card = f'''
            <div class="card">
                <div class="card-header">
                    <span class="scaffold">{scaffold}</span>
                    <span class="badge {badge_class}">{method}</span>
                </div>
            '''

            if has_structure:
                pdb_escaped = entry["pdb_data"].replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                card += f'''
                <div id="{vid}" class="viewer-container"></div>
                <script>
                (function() {{
                    let viewer = $3Dmol.createViewer("{vid}", {{backgroundColor: "white"}});
                    let pdbData = `{pdb_escaped}`;
                    viewer.addModel(pdbData, "pdb");
                    // Chain A (binder) in cyan cartoon
                    viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#00BCD4"}}}});
                    // Chain B (target) in light gray cartoon
                    viewer.setStyle({{chain: "B"}}, {{cartoon: {{color: "#BDBDBD"}}}});
                    viewer.zoomTo();
                    viewer.render();
                }})();
                </script>
                '''
            else:
                card += '<div class="viewer-container no-structure">ColabFold prediction pending</div>'

            # Metrics table
            cf_ipsae = entry.get("cf_ipsae")
            cf_binder_plddt = entry.get("cf_binder_plddt")
            cf_iptm = entry.get("cf_iptm")
            cf_ptm = entry.get("cf_ptm")
            pipeline_energy = entry.get("total_energy")
            pipeline_lis = entry.get("lis_energy")

            def fmt(v, decimals=2):
                return f"{v:.{decimals}f}" if v is not None else "—"

            def plddt_color(v):
                if v is None: return ""
                if v > 80: return "color: #2E7D32"
                if v > 60: return "color: #F57F17"
                return "color: #C62828"

            def ipsae_color(v):
                if v is None: return ""
                if v < 10: return "color: #2E7D32"
                if v < 20: return "color: #F57F17"
                return "color: #C62828"

            card += f'''
                <div class="metrics">
                    <table>
                        <tr><th colspan="2">ColabFold Metrics</th></tr>
                        <tr><td>ipSAE</td><td style="{ipsae_color(cf_ipsae)}">{fmt(cf_ipsae)}</td></tr>
                        <tr><td>Binder pLDDT</td><td style="{plddt_color(cf_binder_plddt)}">{fmt(cf_binder_plddt, 1)}</td></tr>
                        <tr><td>ipTM</td><td>{fmt(cf_iptm, 3)}</td></tr>
                        <tr><td>pTM</td><td>{fmt(cf_ptm, 3)}</td></tr>
                        <tr><th colspan="2">Pipeline Metrics</th></tr>
                        <tr><td>Total Energy</td><td>{fmt(pipeline_energy, 4)}</td></tr>
                        <tr><td>LIS Energy</td><td>{fmt(pipeline_lis, 4)}</td></tr>
                        <tr><td>Best at Cycle</td><td>{entry.get("cycle", "—")}</td></tr>
                    </table>
                </div>
            </div>
            '''
            cards.append(card)

        cards.append("</div>")  # close cards-grid

    cards_html = "\n".join(cards)

    # Summary stats
    n_total = len(results_data)
    n_with_struct = sum(1 for e in results_data if e.get("pdb_data"))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ProFam-BAGEL Results Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #f5f5f5;
        padding: 20px;
        color: #333;
    }}
    h1 {{
        text-align: center;
        margin-bottom: 5px;
        font-size: 1.8em;
    }}
    .subtitle {{
        text-align: center;
        color: #666;
        margin-bottom: 20px;
        font-size: 0.95em;
    }}
    .summary-bar {{
        background: white;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        gap: 30px;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 0.9em;
    }}
    .summary-bar span {{ font-weight: 600; }}
    .target-header {{
        margin: 25px 0 15px 0;
        padding: 8px 15px;
        background: #37474F;
        color: white;
        border-radius: 6px;
        font-size: 1.2em;
    }}
    .cards-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
        gap: 16px;
    }}
    .card {{
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s;
    }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .card-header {{
        padding: 10px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #eee;
    }}
    .scaffold {{
        font-weight: 600;
        font-size: 0.95em;
    }}
    .badge {{
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
        color: white;
    }}
    .badge-grpo {{ background: #1565C0; }}
    .badge-bt {{ background: #6A1B9A; }}
    .badge-bandit {{ background: #E65100; }}
    .badge-greedy {{ background: #2E7D32; }}
    .badge-default {{ background: #666; }}
    .viewer-container {{
        width: 100%;
        height: 280px;
        position: relative;
    }}
    .no-structure {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: #fafafa;
        color: #999;
        font-style: italic;
    }}
    .metrics {{
        padding: 8px 12px;
    }}
    .metrics table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85em;
    }}
    .metrics th {{
        text-align: left;
        padding: 4px 0;
        color: #666;
        font-size: 0.85em;
        border-bottom: 1px solid #eee;
    }}
    .metrics td {{
        padding: 3px 0;
    }}
    .metrics td:last-child {{
        text-align: right;
        font-family: "SF Mono", "Consolas", monospace;
        font-weight: 500;
    }}
    .legend {{
        display: flex;
        gap: 12px;
        justify-content: center;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 0.85em;
    }}
    .legend-color {{
        width: 14px;
        height: 14px;
        border-radius: 50%;
    }}
</style>
</head>
<body>
    <h1>ProFam-BAGEL: Best Binder Structures</h1>
    <p class="subtitle">ColabFold (AF2-Multimer v3) predictions of best binders from each campaign</p>

    <div class="summary-bar">
        <div>Campaigns: <span>{n_total}</span></div>
        <div>Structures predicted: <span>{n_with_struct}</span></div>
        <div>Targets: <span>{len(by_target)}</span></div>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#00BCD4"></div> Binder (chain A)</div>
        <div class="legend-item"><div class="legend-color" style="background:#BDBDBD"></div> Target (chain B)</div>
        <span style="margin: 0 10px; color: #ccc">|</span>
        <div class="legend-item"><div class="legend-color" style="background:#1565C0"></div> bandit_grpo</div>
        <div class="legend-item"><div class="legend-color" style="background:#6A1B9A"></div> bandit_bt</div>
        <div class="legend-item"><div class="legend-color" style="background:#E65100"></div> proposal_bandit</div>
        <div class="legend-item"><div class="legend-color" style="background:#2E7D32"></div> random_greedy</div>
    </div>

    {cards_html}

    <div style="text-align:center; margin-top:30px; color:#999; font-size:0.8em;">
        Generated by ProFam-BAGEL pipeline analysis
    </div>
</body>
</html>'''
    return html


def main():
    results = load_best_sequences()

    for entry in results:
        cf_name = get_cf_name(entry)
        pdb_path = find_pdb(cf_name)
        scores_path = find_scores(cf_name)

        if pdb_path and scores_path:
            binder_len = len(entry["binder_seq"])
            metrics = compute_metrics(scores_path, binder_len)
            entry["cf_ipsae"] = metrics["ipsae"]
            entry["cf_binder_plddt"] = metrics["binder_plddt"]
            entry["cf_target_plddt"] = metrics["target_plddt"]
            entry["cf_iptm"] = metrics["iptm"]
            entry["cf_ptm"] = metrics["ptm"]
            entry["pdb_data"] = read_pdb(pdb_path)
            print(f"  OK  {cf_name}: ipSAE={metrics['ipsae']:.1f}, binder_pLDDT={metrics['binder_plddt']:.1f}, ipTM={metrics['iptm']:.3f}")
        else:
            entry["pdb_data"] = None
            print(f"SKIP  {cf_name}: no ColabFold output yet")

    html = build_html(results)

    with open(HTML_OUTPUT, "w") as f:
        f.write(html)

    print(f"\nHTML written to {HTML_OUTPUT}")
    n_ok = sum(1 for e in results if e.get("pdb_data"))
    print(f"{n_ok}/{len(results)} structures included")


if __name__ == "__main__":
    main()
