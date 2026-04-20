#!/bin/bash
# Rsync outputs from remote GPU nodes to local.
# Excludes large structure files (.cif, .pae, .plddt) to keep sync fast.
# Logs to /mnt/disk2/ThinkingPLM/sync_outputs.log.

LOGFILE="/mnt/disk2/ThinkingPLM/sync_outputs.log"
SSH_KEY="$HOME/.ssh/gpu-ml-key.pem"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
LOCAL_DIR="/mnt/disk2/ThinkingPLM/outputs/"

NODES="3.133.117.251"

echo "$(date): sync starting" >> "$LOGFILE"

for ip in $NODES; do
  rsync -az --timeout=60 \
    --exclude='*.cif' --exclude='*.pae' --exclude='*.plddt' \
    -e "ssh $SSH_OPTS" \
    ubuntu@$ip:~/ThinkingPLM/outputs/ "$LOCAL_DIR" >> "$LOGFILE" 2>&1
  rc=$?
  echo "$(date): $ip rsync exit $rc" >> "$LOGFILE"
done

echo "$(date): sync done" >> "$LOGFILE"
