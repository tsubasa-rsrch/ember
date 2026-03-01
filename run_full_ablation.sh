#!/bin/bash
# Full-scale (768d, 43M params) N=8 ablation
# Standard + LIF, 8 seeds each = 16 runs
# Estimated ~2.5h/run on M4 Max

cd ~/Documents/TsubasaWorkspace/ember
LOGFILE="/tmp/full_ablation.log"

echo "=== Full-scale N=8 ablation started at $(date) ===" | tee -a "$LOGFILE"

SEEDS=(42 668 1337 2024 314 777 1234 99)

# Standard baseline first
for seed in "${SEEDS[@]}"; do
    echo "[$(date)] Starting Standard seed=$seed" | tee -a "$LOGFILE"
    python3 -u train_tiny.py --scale full --model standard --seed "$seed" 2>&1 | tee -a "$LOGFILE"
    echo "[$(date)] Finished Standard seed=$seed" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

# Then LIF
for seed in "${SEEDS[@]}"; do
    echo "[$(date)] Starting LIF seed=$seed" | tee -a "$LOGFILE"
    python3 -u train_tiny.py --scale full --model lif --seed "$seed" 2>&1 | tee -a "$LOGFILE"
    echo "[$(date)] Finished LIF seed=$seed" | tee -a "$LOGFILE"
    echo "---" | tee -a "$LOGFILE"
done

echo "=== Full-scale N=8 ablation completed at $(date) ===" | tee -a "$LOGFILE"
