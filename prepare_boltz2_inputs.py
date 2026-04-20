#!/usr/bin/env python3
"""Prepare Boltz2 YAML inputs from best sequences JSON."""

import json
from pathlib import Path

BASE = Path("/mnt/disk2/ThinkingPLM")
INPUT_JSON = BASE / "colabfold_input" / "best_sequences.json"
BOLTZ_DIR = BASE / "boltz2_input"


def main():
    with open(INPUT_JSON) as f:
        results = json.load(f)

    BOLTZ_DIR.mkdir(exist_ok=True)

    for r in results:
        safe_name = r["campaign"].replace("/", "_")
        if r["target_id"] != "2GDZ":
            safe_name = f"{r['target_id']}_{safe_name}"
        else:
            safe_name = f"sc_rep3_{safe_name}"

        yaml = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {r['binder_seq']}
      msa: empty
  - protein:
      id: B
      sequence: {r['target_seq']}
      msa: empty
"""
        (BOLTZ_DIR / f"{safe_name}.yaml").write_text(yaml)

    print(f"Wrote {len(results)} Boltz2 YAMLs to {BOLTZ_DIR}")


if __name__ == "__main__":
    main()
