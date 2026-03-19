#!/usr/bin/env python
"""
Generate all 28 benchmark pipeline YAML configs for the iterative
prompt-updating experiment.

Targets:
  1. 2GDZ (15-PGDH)  — scaffold + de_novo
  2. 4ZQK (PD-L1)    — scaffold
  3. 3DI2 (IL-7Rα)   — scaffold
  4. 1WWW (TrkA)     — scaffold
  5. 4LXV (HA)       — scaffold
  6. 1VPF (VEGF-A)   — scaffold

Conditions:
  - profam_update:  proposal_method=profam,  freeze_prompt=false
  - profam_frozen:  proposal_method=profam,  freeze_prompt=true
  - random_update:  proposal_method=random_mutation, freeze_prompt=false
  - random_frozen:  proposal_method=random_mutation, freeze_prompt=true
"""

from pathlib import Path

import yaml

# ── Target definitions ──────────────────────────────────────────────────────

TARGETS = {
    "2GDZ": {
        "energy_config": "configs/energy/example_energy_boltz_ipsae_2GDZ.yaml",
        "inits": {
            "scaffold": "configs/sequences/initial_sequence_hairpin.fasta",
            "denovo": "configs/sequences/initial_sequence_boltz2_de_novo.fasta",
        },
    },
    "4ZQK": {
        "energy_config": "configs/energy/energy_boltz_ipsae_4ZQK.yaml",
        "inits": {"scaffold": "configs/sequences/initial_sequence_hairpin.fasta"},
    },
    "3DI2": {
        "energy_config": "configs/energy/energy_boltz_ipsae_3DI2.yaml",
        "inits": {"scaffold": "configs/sequences/initial_sequence_hairpin.fasta"},
    },
    "1WWW": {
        "energy_config": "configs/energy/energy_boltz_ipsae_1WWW.yaml",
        "inits": {"scaffold": "configs/sequences/initial_sequence_hairpin.fasta"},
    },
    "4LXV": {
        "energy_config": "configs/energy/energy_boltz_ipsae_4LXV.yaml",
        "inits": {"scaffold": "configs/sequences/initial_sequence_hairpin.fasta"},
    },
    "1VPF": {
        "energy_config": "configs/energy/energy_boltz_ipsae_1VPF.yaml",
        "inits": {"scaffold": "configs/sequences/initial_sequence_hairpin.fasta"},
    },
}

# ── Condition definitions ───────────────────────────────────────────────────

CONDITIONS = {
    "profam_update": {
        "proposal_method": "profam",
        "freeze_prompt": False,
    },
    "profam_frozen": {
        "proposal_method": "profam",
        "freeze_prompt": True,
    },
    "random_update": {
        "proposal_method": "random_mutation",
        "freeze_prompt": False,
    },
    "random_frozen": {
        "proposal_method": "random_mutation",
        "freeze_prompt": True,
    },
}

# ── Shared run parameters ──────────────────────────────────────────────────

SHARED = {
    "profam_checkpoint_dir": ".profam_repo/model_checkpoints/profam-1",
    "profam_sampler": "single",
    "profam_num_samples": 1,
    "profam_max_tokens": 8192,
    "profam_temperature": 0.8,
    "profam_top_p": 0.95,
    "f_inject": 0.25,
    "max_cycles": 100,
    "softmax_temperature": 0.01,
    "run_on_modal": True,
    "output_frequency": 1,
    "enforce_template": False,
    "sample_with_reinsertion": False,
    "reinject_initial": True,
    "n_memory": 0,
    "elitism": True,
    "accept_only_improvement": True,
    "max_mutations": 5,
}

# ── Deterministic seed per (target, init, condition) ────────────────────────

BASE_SEED = 100000


def make_seed(target: str, init: str, condition: str) -> int:
    """Deterministic seed from a hash of the run identifier."""
    key = f"{target}_{init}_{condition}"
    return BASE_SEED + (hash(key) % 900000)


# ── Generate configs ────────────────────────────────────────────────────────

def main():
    out_dir = Path("configs/pipelines")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for pdb_id, target_info in TARGETS.items():
        for init_name, init_fasta in target_info["inits"].items():
            for cond_name, cond_params in CONDITIONS.items():
                fname = f"bench_{pdb_id}_{init_name}_{cond_name}.yaml"
                output_path = out_dir / fname

                cfg = dict(SHARED)
                cfg["initial_fasta"] = init_fasta
                cfg["energy_config"] = target_info["energy_config"]
                cfg["output_dir"] = f"outputs/bench/{pdb_id}/{init_name}/{cond_name}"
                cfg["random_seed"] = make_seed(pdb_id, init_name, cond_name)
                cfg.update(cond_params)

                # Add profam_max_generated_length as null
                cfg["profam_max_generated_length"] = None

                # Comment header
                header = (
                    f"## Benchmark: {pdb_id} / {init_name} / {cond_name}\n"
                    f"##\n"
                    f"## Target: {pdb_id}, Init: {init_name}, Condition: {cond_name}\n"
                    f"## proposal_method: {cond_params['proposal_method']}, "
                    f"freeze_prompt: {cond_params['freeze_prompt']}\n"
                    f"\n"
                )

                with output_path.open("w") as f:
                    f.write(header)
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

                generated.append(str(output_path))

    print(f"Generated {len(generated)} benchmark configs:")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
