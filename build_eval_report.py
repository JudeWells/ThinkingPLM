#!/usr/bin/env python3
"""Build an HTML report with 3D structure viewers for the final evaluation results.

Uses 3Dmol.js for interactive structure viewing directly in the browser.
Reads CIF files from final_evaluation/ subdirectories and embeds them inline.
"""

import csv
from pathlib import Path

BASE = Path("/mnt/disk2/ThinkingPLM")
EVAL_DIR = BASE / "final_evaluation"
CSV_PATH = EVAL_DIR / "evaluation_results.csv"
OUTPUT_HTML = EVAL_DIR / "results_report.html"


def read_cif(path):
    if path.is_file():
        return path.read_text()
    return None


def fmt(val, decimals=4):
    try:
        v = float(val)
        if v != v:
            return "—"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return "—"


def build_html():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["campaign_best_energy"]))

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Boltz2 15PGDH Binder Evaluation</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; margin: 20px; }
  h1 { color: #00bfff; border-bottom: 2px solid #00bfff; padding-bottom: 10px; }
  h2 { color: #ffab40; margin-top: 40px; }
  h3 { color: #00e676; }
  table { border-collapse: collapse; margin: 15px 0; width: 100%; }
  th { background: #16213e; color: #00bfff; padding: 8px 12px; text-align: left;
       border: 1px solid #333; font-size: 13px; }
  td { padding: 6px 12px; border: 1px solid #333; }
  tr:nth-child(even) { background: #1f1f3a; }
  tr:hover { background: #2a2a4a; }
  .good { color: #00e676; font-weight: bold; }
  .warn { color: #ffab40; }
  .bad { color: #ff6b6b; }
  .viewer-container { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
  .viewer-box { flex: 1; min-width: 350px; max-width: 500px; }
  .viewer-box h4 { text-align: center; margin: 5px 0; font-size: 13px; color: #aaa; }
  .mol-viewer { width: 100%; height: 350px; border: 1px solid #333;
                border-radius: 4px; position: relative; }
  .seq { font-family: 'Courier New', monospace; font-size: 12px; background: #16213e;
         padding: 8px; border-radius: 4px; word-break: break-all; margin: 5px 0; }
  .card { background: #16213e; border-radius: 8px; padding: 20px; margin: 20px 0;
          border: 1px solid #333; }
  .summary-table td { padding: 4px 10px; }
  .summary-table td:nth-child(odd) { font-weight: bold; color: #aaa; }
  .rank { display: inline-block; background: #00bfff; color: #000; border-radius: 50%;
          width: 28px; height: 28px; line-height: 28px; text-align: center;
          font-weight: bold; margin-right: 8px; }
  .na { color: #666; }
</style>
</head>
<body>
<h1>Boltz2 15PGDH Binder Evaluation Report</h1>
<p>16 best sequences from Boltz2 design campaigns against 15-PGDH (PDB: 2GDZ),
re-evaluated with independent ESMFold and Boltz2 predictions.
Ranked by campaign energy (ipSAE + 0.1 pLDDT + 0.05 iPTM + length penalty).</p>

<h2>Summary Table</h2>
<table>
<tr>
  <th>#</th><th>Run</th><th>Len</th>
  <th>Campaign<br>Energy</th>
  <th>ESM<br>pLDDT</th><th>ESM<br>PTM</th><th>ESM<br>RMSD</th>
  <th>B2<br>pLDDT</th><th>B2<br>iPTM</th><th>B2<br>RMSD</th>
</tr>
""")

    for i, row in enumerate(rows, 1):
        name = row["run_name"].split("/")[-1]

        def color_plddt(val_str):
            v = fmt(val_str)
            if v == "—": return '<td class="na">—</td>'
            fv = float(val_str)
            cls = "good" if fv >= 0.85 else ("warn" if fv >= 0.7 else "bad")
            return f'<td class="{cls}">{v}</td>'

        def color_ptm(val_str):
            v = fmt(val_str)
            if v == "—": return '<td class="na">—</td>'
            fv = float(val_str)
            cls = "good" if fv >= 0.7 else ("warn" if fv >= 0.5 else "bad")
            return f'<td class="{cls}">{v}</td>'

        def color_iptm(val_str):
            v = fmt(val_str)
            if v == "—": return '<td class="na">—</td>'
            fv = float(val_str)
            cls = "good" if fv >= 0.5 else ("warn" if fv >= 0.3 else "bad")
            return f'<td class="{cls}">{v}</td>'

        def color_rmsd(val_str):
            v = fmt(val_str, 2)
            if v == "—": return '<td class="na">—</td>'
            fv = float(val_str)
            cls = "good" if fv < 2.0 else ("warn" if fv < 5.0 else "bad")
            return f'<td class="{cls}">{v}</td>'

        parts.append(f"<tr><td>{i}</td><td>{name}</td><td>{row['binder_len']}</td>"
                     f"<td>{fmt(row['campaign_best_energy'])}</td>"
                     f"{color_plddt(row['esm_complex_plddt'])}"
                     f"{color_ptm(row['esm_complex_ptm'])}"
                     f"{color_rmsd(row['esm_rmsd_bound_unbound'])}"
                     f"{color_plddt(row['b2_complex_plddt'])}"
                     f"{color_iptm(row['b2_complex_iptm'])}"
                     f"{color_rmsd(row['b2_rmsd_bound_unbound'])}"
                     f"</tr>\n")

    parts.append("</table>\n")
    parts.append("<h2>Individual Sequences</h2>\n")

    vid = 0
    for i, row in enumerate(rows, 1):
        name = row["run_name"]
        short = name.split("/")[-1]
        subdir = EVAL_DIR / name.replace("/", "_")

        parts.append(f"""
<div class="card">
<h3><span class="rank">{i}</span>{short}</h3>
<div class="seq">{row['binder_seq']}</div>
<table class="summary-table">
<tr><td>Length</td><td>{row['binder_len']} aa</td>
    <td>Campaign Energy</td><td>{fmt(row['campaign_best_energy'])}</td>
    <td>Best Cycle</td><td>{row['best_cycle']} / {row['total_cycles']}</td></tr>
<tr><td>ESM pLDDT</td><td>{fmt(row['esm_complex_plddt'])}</td>
    <td>ESM PTM</td><td>{fmt(row['esm_complex_ptm'])}</td>
    <td>ESM RMSD</td><td>{fmt(row['esm_rmsd_bound_unbound'], 2)} A</td></tr>
<tr><td>ESM Mono pLDDT</td><td>{fmt(row['esm_mono_plddt'])}</td>
    <td>ESM Mono PTM</td><td>{fmt(row['esm_mono_ptm'])}</td>
    <td></td><td></td></tr>
<tr><td>B2 pLDDT</td><td>{fmt(row['b2_complex_plddt'])}</td>
    <td>B2 iPTM</td><td>{fmt(row['b2_complex_iptm'])}</td>
    <td>B2 RMSD</td><td>{fmt(row['b2_rmsd_bound_unbound'], 2)} A</td></tr>
<tr><td>B2 Mono pLDDT</td><td>{fmt(row['b2_mono_plddt'])}</td>
    <td>B2 PTM</td><td>{fmt(row['b2_complex_ptm'])}</td>
    <td></td><td></td></tr>
</table>
<div class="viewer-container">
""")

        for title, fname in [
            ("ESMFold Complex", "esmfold_complex.cif"),
            ("ESMFold Monomer", "esmfold_monomer.cif"),
            ("Boltz2 Complex", "boltz2_complex.cif"),
            ("Boltz2 Monomer", "boltz2_monomer.cif"),
        ]:
            cif = read_cif(subdir / fname)
            if cif is None:
                parts.append(f'<div class="viewer-box"><h4>{title}</h4>'
                             '<div class="mol-viewer" style="display:flex;align-items:center;'
                             'justify-content:center;color:#666;">No structure</div></div>\n')
                continue

            v = f"v{vid}"
            vid += 1
            cif_js = cif.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
            blen = int(row["binder_len"])
            is_complex = "complex" in fname

            if is_complex:
                style_js = (f'viewer.setStyle({{resi:["1-{blen}"]}},{{cartoon:{{color:"#00bfff"}}}}); '
                            f'viewer.setStyle({{resi:["{blen+1}-9999"]}},{{cartoon:{{color:"#ff8c00",opacity:0.6}}}});')
            else:
                style_js = 'viewer.setStyle({},{cartoon:{color:"#00bfff"}});'

            parts.append(f"""<div class="viewer-box"><h4>{title}</h4>
<div id="{v}" class="mol-viewer"></div>
<script>
(function(){{ var viewer=$3Dmol.createViewer("{v}",{{backgroundColor:"#1a1a2e"}});
viewer.addModel(`{cif_js}`,"cif"); {style_js}
viewer.zoomTo(); viewer.render(); }})();
</script></div>
""")

        parts.append("</div></div>\n")

    parts.append("""
<hr>
<p style="color:#666;font-size:12px;">Boltz2 15PGDH binder evaluation report.
ESMFold + Boltz2 re-evaluation of design campaign best sequences.</p>
</body></html>""")

    OUTPUT_HTML.write_text("".join(parts))
    print(f"Written {OUTPUT_HTML} ({len(rows)} sequences, {vid} viewers)")


if __name__ == "__main__":
    build_html()
