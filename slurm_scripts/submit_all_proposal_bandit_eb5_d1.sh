#!/bin/bash
# Submit all proposal bandit benchmark jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/run_bench_2GDZ_15PGDH_3helix_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2GDZ_15PGDH_4D5_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2GDZ_15PGDH_ankyrin_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2GDZ_15PGDH_nanobody_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2VSM_nipah_3helix_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2VSM_nipah_4D5_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2VSM_nipah_ankyrin_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_2VSM_nipah_nanobody_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4OYD_epstein_barr_3helix_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4OYD_epstein_barr_4D5_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4OYD_epstein_barr_ankyrin_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4OYD_epstein_barr_nanobody_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4ZQK_PD-L1_3helix_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4ZQK_PD-L1_4D5_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4ZQK_PD-L1_ankyrin_proposal_bandit_eb5_d1.sh"
sbatch "$SCRIPT_DIR/run_bench_4ZQK_PD-L1_nanobody_proposal_bandit_eb5_d1.sh"

echo "Submitted all proposal bandit benchmark jobs"
