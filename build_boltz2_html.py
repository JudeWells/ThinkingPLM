#!/usr/bin/env python3
"""Build HTML results viewer for Boltz2 predictions."""

import json
import glob
import importlib.util
from pathlib import Path
import numpy as np

BASE = Path("/mnt/disk2/ThinkingPLM")

# Load compute_ipsae from pipeline/utils.py without importing the pipeline
# package (which pulls in bagel-dependent modules at import time).
_utils_spec = importlib.util.spec_from_file_location(
    "pipeline_utils_standalone", BASE / "pipeline" / "utils.py"
)
_utils_mod = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_mod)
compute_ipsae = _utils_mod.compute_ipsae
BOLTZ_OUT = BASE / "boltz2_output" / "boltz_results_boltz2_input" / "predictions"
INPUT_JSON = BASE / "colabfold_input" / "best_sequences.json"
HTML_OUTPUT = BASE / "results_viewer_boltz2.html"


def load_best_sequences():
    with open(INPUT_JSON) as f:
        return json.load(f)


def get_boltz_name(result):
    safe_name = result["campaign"].replace("/", "_")
    if result["target_id"] != "2GDZ":
        return f"{result['target_id']}_{safe_name}"
    else:
        return f"sc_rep3_{safe_name}"


def find_boltz_samples(name):
    """Find all sample outputs for a given input name.

    Boltz2 with --diffusion_samples 5 creates model_0..model_4 files per input.
    Returns a list of dicts, one per sample, each with pdb/confidence/plddt/pae paths.
    """
    sub_dir = BOLTZ_OUT / name
    if not sub_dir.exists():
        return []
    samples = []
    for i in range(20):  # support up to 20 samples
        pdb = sub_dir / f"{name}_model_{i}.pdb"
        conf = sub_dir / f"confidence_{name}_model_{i}.json"
        plddt = sub_dir / f"plddt_{name}_model_{i}.npz"
        pae = sub_dir / f"pae_{name}_model_{i}.npz"
        if not (pdb.exists() and conf.exists()):
            break
        samples.append({
            "model_idx": i,
            "pdb": pdb,
            "confidence": conf,
            "plddt": plddt if plddt.exists() else None,
            "pae": pae if pae.exists() else None,
        })
    return samples


def compute_sample_metrics(paths, binder_len):
    """Compute metrics for a single Boltz2 sample."""
    with open(paths["confidence"]) as f:
        conf = json.load(f)

    binder_plddt = None
    if paths["plddt"]:
        plddt_data = np.load(paths["plddt"])
        plddt = plddt_data["plddt"]
        if plddt.max() <= 1.0:
            plddt = plddt * 100.0
        binder_plddt = float(np.mean(plddt[:binder_len]))

    ipsae = None
    if paths["pae"]:
        pae_data = np.load(paths["pae"])
        pae = pae_data["pae"]
        chain_a = np.arange(binder_len)
        chain_b = np.arange(binder_len, pae.shape[0])
        ipsae = compute_ipsae(pae, chain_a, chain_b, pae_cutoff=10.0)

    return {
        "binder_plddt": binder_plddt,
        "ipsae": ipsae,
        "ptm": conf.get("ptm"),
        "iptm": conf.get("iptm"),
        "complex_plddt": conf.get("complex_plddt"),
        "confidence_score": conf.get("confidence_score"),
    }


def aggregate_samples(samples, binder_len):
    """Compute per-sample metrics, pick best, and return mean/std stats.

    'Best' = sample with lowest ipSAE (lowest interface PAE).
    """
    per_sample = []
    for s in samples:
        m = compute_sample_metrics(s, binder_len)
        m["_paths"] = s
        per_sample.append(m)

    def safe_mean(key):
        vals = [m[key] for m in per_sample if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def safe_std(key):
        vals = [m[key] for m in per_sample if m.get(key) is not None]
        return float(np.std(vals)) if len(vals) > 1 else None

    means = {
        "binder_plddt_mean": safe_mean("binder_plddt"),
        "binder_plddt_std": safe_std("binder_plddt"),
        "ipsae_mean": safe_mean("ipsae"),
        "ipsae_std": safe_std("ipsae"),
        "iptm_mean": safe_mean("iptm"),
        "iptm_std": safe_std("iptm"),
        "ptm_mean": safe_mean("ptm"),
        "complex_plddt_mean": safe_mean("complex_plddt"),
        "confidence_mean": safe_mean("confidence_score"),
        "n_samples": len(per_sample),
    }

    # Pick the best sample by ipSAE (highest = best interface, Dunbrack variant)
    ipsae_candidates = [m for m in per_sample if m.get("ipsae") is not None]
    if ipsae_candidates:
        best = max(ipsae_candidates, key=lambda m: m["ipsae"])
    else:
        best = per_sample[0]

    return means, best


def read_pdb(pdb_path):
    with open(pdb_path) as f:
        return f.read()


def build_html(results_data):
    by_target = {}
    for entry in results_data:
        by_target.setdefault(entry["target_id"], []).append(entry)

    # Sort by ipSAE mean descending (higher = better binder)
    for tid in by_target:
        by_target[tid].sort(
            key=lambda x: x.get("boltz_ipsae_mean") if x.get("boltz_ipsae_mean") is not None else -1.0,
            reverse=True,
        )

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
            method = campaign.split("/")[-1]
            badge_class = {
                "bandit_grpo": "badge-grpo",
                "bandit_bt": "badge-bt",
                "proposal_bandit": "badge-bandit",
                "random_greedy": "badge-greedy",
            }.get(method, "badge-default")
            scaffold = campaign.split("/")[0]

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
                    viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#00BCD4"}}}});
                    viewer.setStyle({{chain: "B"}}, {{cartoon: {{color: "#BDBDBD"}}}});
                    viewer.zoomTo();
                    viewer.render();
                }})();
                </script>
                '''
            else:
                card += '<div class="viewer-container no-structure">Boltz2 prediction pending</div>'

            def fmt(v, d=2):
                return f"{v:.{d}f}" if v is not None else "—"

            def plddt_color(v):
                if v is None: return ""
                if v > 80: return "color: #2E7D32"
                if v > 60: return "color: #F57F17"
                return "color: #C62828"

            def ipsae_color(v):
                # Dunbrack ipSAE: higher is better, bounded [0, 1]
                if v is None: return ""
                if v > 0.6: return "color: #2E7D32"
                if v > 0.4: return "color: #F57F17"
                return "color: #C62828"

            def iptm_color(v):
                if v is None: return ""
                if v > 0.7: return "color: #2E7D32"
                if v > 0.5: return "color: #F57F17"
                return "color: #C62828"

            def fmt_ms(mean, std, d=2):
                if mean is None:
                    return "—"
                if std is None:
                    return f"{mean:.{d}f}"
                return f"{mean:.{d}f} ± {std:.{d}f}"

            n_s = entry.get("boltz_n_samples", 0)
            best_idx = entry.get("boltz_best_model_idx")
            best_label = f"sample {best_idx}" if best_idx is not None else "—"

            card += f'''
                <div class="metrics">
                    <table>
                        <tr><th colspan="2">Boltz2 mean of {n_s} samples</th></tr>
                        <tr><td>ipSAE (Dunbrack d0res)</td><td style="{ipsae_color(entry.get("boltz_ipsae_mean"))}">{fmt_ms(entry.get("boltz_ipsae_mean"), entry.get("boltz_ipsae_std"), 3)}</td></tr>
                        <tr><td>Binder pLDDT</td><td style="{plddt_color(entry.get("boltz_binder_plddt_mean"))}">{fmt_ms(entry.get("boltz_binder_plddt_mean"), entry.get("boltz_binder_plddt_std"), 1)}</td></tr>
                        <tr><td>ipTM</td><td style="{iptm_color(entry.get("boltz_iptm_mean"))}">{fmt_ms(entry.get("boltz_iptm_mean"), entry.get("boltz_iptm_std"), 3)}</td></tr>
                        <tr><td>pTM</td><td>{fmt(entry.get("boltz_ptm_mean"), 3)}</td></tr>
                        <tr><td>Complex pLDDT</td><td>{fmt(entry.get("boltz_complex_plddt_mean"), 3)}</td></tr>
                        <tr><td>Confidence score</td><td>{fmt(entry.get("boltz_confidence_mean"), 3)}</td></tr>
                        <tr><th colspan="2">Best sample ({best_label}) · shown above</th></tr>
                        <tr><td>ipSAE</td><td style="{ipsae_color(entry.get("boltz_best_ipsae"))}">{fmt(entry.get("boltz_best_ipsae"), 3)}</td></tr>
                        <tr><td>Binder pLDDT</td><td style="{plddt_color(entry.get("boltz_best_binder_plddt"))}">{fmt(entry.get("boltz_best_binder_plddt"), 1)}</td></tr>
                        <tr><td>ipTM</td><td style="{iptm_color(entry.get("boltz_best_iptm"))}">{fmt(entry.get("boltz_best_iptm"), 3)}</td></tr>
                        <tr><th colspan="2">Pipeline (ESMFold LIS)</th></tr>
                        <tr><td>Total Energy</td><td>{fmt(entry.get("total_energy"), 4)}</td></tr>
                        <tr><td>Best at Cycle</td><td>{entry.get("cycle", "—")}</td></tr>
                    </table>
                </div>
            </div>
            '''
            cards.append(card)
        cards.append("</div>")

    cards_html = "\n".join(cards)

    n_total = len(results_data)
    n_with_struct = sum(1 for e in results_data if e.get("pdb_data"))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ProFam-BAGEL Results — Boltz2 Predictions</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; padding: 20px; color: #333; }}
    h1 {{ text-align: center; margin-bottom: 5px; font-size: 1.8em; }}
    .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 0.95em; }}
    .summary-bar {{ background: white; border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; display: flex; gap: 30px; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.9em; }}
    .summary-bar span {{ font-weight: 600; }}
    .target-header {{ margin: 25px 0 15px 0; padding: 8px 15px; background: #37474F; color: white; border-radius: 6px; font-size: 1.2em; }}
    .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: box-shadow 0.2s; }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .card-header {{ padding: 10px 12px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; }}
    .scaffold {{ font-weight: 600; font-size: 0.95em; }}
    .badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 500; color: white; }}
    .badge-grpo {{ background: #1565C0; }}
    .badge-bt {{ background: #6A1B9A; }}
    .badge-bandit {{ background: #E65100; }}
    .badge-greedy {{ background: #2E7D32; }}
    .badge-default {{ background: #666; }}
    .viewer-container {{ width: 100%; height: 280px; position: relative; }}
    .no-structure {{ display: flex; align-items: center; justify-content: center; background: #fafafa; color: #999; font-style: italic; }}
    .metrics {{ padding: 8px 12px; }}
    .metrics table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
    .metrics th {{ text-align: left; padding: 4px 0; color: #666; font-size: 0.85em; border-bottom: 1px solid #eee; }}
    .metrics td {{ padding: 3px 0; }}
    .metrics td:last-child {{ text-align: right; font-family: "SF Mono", "Consolas", monospace; font-weight: 500; }}
    .legend {{ display: flex; gap: 12px; justify-content: center; margin-bottom: 15px; flex-wrap: wrap; }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.85em; }}
    .legend-color {{ width: 14px; height: 14px; border-radius: 50%; }}
</style>
</head>
<body>
    <h1>ProFam-BAGEL: Best Binder Structures (Boltz2)</h1>
    <p class="subtitle">Boltz2 predictions (5 diffusion samples per campaign) · sorted within each target by mean ipSAE (Dunbrack d0res, higher = better) · visualized structure is the highest-ipSAE sample</p>

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
        Generated by ProFam-BAGEL pipeline analysis · Boltz2 structure prediction
    </div>
</body>
</html>'''
    return html


def main():
    results = load_best_sequences()

    for entry in results:
        name = get_boltz_name(entry)
        samples = find_boltz_samples(name)
        if samples:
            binder_len = len(entry["binder_seq"])
            means, best = aggregate_samples(samples, binder_len)
            entry["boltz_n_samples"] = means["n_samples"]
            entry["boltz_ipsae_mean"] = means["ipsae_mean"]
            entry["boltz_ipsae_std"] = means["ipsae_std"]
            entry["boltz_binder_plddt_mean"] = means["binder_plddt_mean"]
            entry["boltz_binder_plddt_std"] = means["binder_plddt_std"]
            entry["boltz_iptm_mean"] = means["iptm_mean"]
            entry["boltz_iptm_std"] = means["iptm_std"]
            entry["boltz_ptm_mean"] = means["ptm_mean"]
            entry["boltz_complex_plddt_mean"] = means["complex_plddt_mean"]
            entry["boltz_confidence_mean"] = means["confidence_mean"]
            # Best-sample metrics (for display of the visualized structure)
            entry["boltz_best_ipsae"] = best["ipsae"]
            entry["boltz_best_binder_plddt"] = best["binder_plddt"]
            entry["boltz_best_iptm"] = best["iptm"]
            entry["boltz_best_model_idx"] = best["_paths"]["model_idx"]
            entry["pdb_data"] = read_pdb(best["_paths"]["pdb"])

            ipsae_s = f"{means['ipsae_mean']:.3f}±{means['ipsae_std']:.3f}" if means["ipsae_std"] is not None else f"{means['ipsae_mean']:.3f}" if means['ipsae_mean'] is not None else "N/A"
            plddt_s = f"{means['binder_plddt_mean']:.1f}±{means['binder_plddt_std']:.1f}" if means["binder_plddt_std"] is not None else f"{means['binder_plddt_mean']:.1f}" if means['binder_plddt_mean'] is not None else "N/A"
            iptm_s = f"{means['iptm_mean']:.3f}" if means['iptm_mean'] is not None else "N/A"
            print(f"  OK  {name} [{means['n_samples']}s]: ipSAE_mean={ipsae_s}, binder_pLDDT_mean={plddt_s}, ipTM_mean={iptm_s}")
        else:
            entry["pdb_data"] = None
            print(f"SKIP  {name}: no Boltz2 output yet")

    html = build_html(results)
    HTML_OUTPUT.write_text(html)
    n_ok = sum(1 for e in results if e.get("pdb_data"))
    print(f"\nHTML written to {HTML_OUTPUT}")
    print(f"{n_ok}/{len(results)} structures included")


if __name__ == "__main__":
    main()
