#!/bin/bash
# Backfill free GPUs with remaining mt2 nipah configs.
# Designed to run via cron every 5 minutes.
# Logs to /mnt/disk2/ThinkingPLM/backfill.log

LOGFILE="/mnt/disk2/ThinkingPLM/backfill.log"
SSH_KEY="$HOME/.ssh/gpu-ml-key.pem"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5"

# Remaining nipah configs to launch (in order)
CONFIGS=(
  "2VSM_nipah_4D5_proposal_bandit"
  "2VSM_nipah_4D5_random_greedy"
  "2VSM_nipah_ankyrin_bandit_grpo"
  "2VSM_nipah_ankyrin_proposal_bandit"
  "2VSM_nipah_ankyrin_random_greedy"
  "2VSM_nipah_nanobody_bandit_grpo"
  "2VSM_nipah_nanobody_proposal_bandit"
  "2VSM_nipah_nanobody_random_greedy"
  "2VSM_nipah_random_init_bandit_grpo"
  "2VSM_nipah_random_init_proposal_bandit"
  "2VSM_nipah_random_init_random_greedy"
)

# Track what we've already launched
LAUNCHED_FILE="/mnt/disk2/ThinkingPLM/backfill_launched.txt"
touch "$LAUNCHED_FILE"

# Nodes to check
NODES="3.14.255.102 18.191.140.159 3.147.71.187 3.148.107.140 3.12.148.62"

echo "$(date): Backfill check starting" >> "$LOGFILE"

# Find next config to launch
NEXT_CONFIG=""
for cfg in "${CONFIGS[@]}"; do
  if ! grep -q "^$cfg$" "$LAUNCHED_FILE"; then
    NEXT_CONFIG="$cfg"
    break
  fi
done

if [ -z "$NEXT_CONFIG" ]; then
  echo "$(date): All configs launched, removing cron job" >> "$LOGFILE"
  crontab -l 2>/dev/null | grep -v "backfill_mt2.sh" | crontab -
  exit 0
fi

# Check each node for free GPUs
for ip in $NODES; do
  FREE_GPUS=$(ssh $SSH_OPTS ubuntu@$ip 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | grep "0 MiB" | cut -d, -f1 | tr -d " "' 2>/dev/null)

  # Only launch ONE job per node per check to avoid double-booking a GPU
  # that's still loading the model from a previous launch
  gpu=$(echo "$FREE_GPUS" | head -1)
  if [ -n "$gpu" ] && [ -n "$NEXT_CONFIG" ]; then
    echo "$(date): Launching $NEXT_CONFIG on $ip GPU$gpu" >> "$LOGFILE"
    ssh -f $SSH_OPTS ubuntu@$ip "cd ~/ThinkingPLM && source ~/miniconda3/bin/activate profam_bagel && CUDA_VISIBLE_DEVICES=$gpu nohup python run_profam_bagel_pipeline.py --config configs/pipelines/multi_target_bench_mt2/${NEXT_CONFIG}.yaml > outputs/mt2_${NEXT_CONFIG}.log 2>&1 &" </dev/null 2>/dev/null

    echo "$NEXT_CONFIG" >> "$LAUNCHED_FILE"
    echo "$(date): Launched $NEXT_CONFIG on $ip GPU$gpu" >> "$LOGFILE"

    # Get next config
    NEXT_CONFIG=""
    for cfg in "${CONFIGS[@]}"; do
      if ! grep -q "^$cfg$" "$LAUNCHED_FILE"; then
        NEXT_CONFIG="$cfg"
        break
      fi
    done
  fi
done

if [ -z "$NEXT_CONFIG" ]; then
  echo "$(date): All configs now launched" >> "$LOGFILE"
fi

echo "$(date): Backfill check done" >> "$LOGFILE"
