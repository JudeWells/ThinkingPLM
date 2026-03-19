#!/bin/bash
# Test all 28 benchmark configs with 2 cycles each.
# Logs per-config output to outputs/bench/test_logs/

set -e

LOG_DIR="outputs/bench/test_logs"
mkdir -p "$LOG_DIR"

CONFIGS=($(ls configs/pipelines/bench_*.yaml | sort))
TOTAL=${#CONFIGS[@]}
PASS=0
FAIL=0
FAILED_LIST=""

echo "Running $TOTAL benchmark configs (2 cycles each)..."
echo ""

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    log_file="$LOG_DIR/${name}.log"
    idx=$((i + 1))

    echo -n "[$idx/$TOTAL] $name ... "

    if conda run -n profam_bagel python run_profam_bagel_pipeline.py \
        --config "$cfg" --max_cycles 2 \
        > "$log_file" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (see $log_file)"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST  $name\n"
    fi
done

echo ""
echo "============================="
echo "Results: $PASS passed, $FAIL failed out of $TOTAL"
if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Failed configs:"
    echo -e "$FAILED_LIST"
fi
