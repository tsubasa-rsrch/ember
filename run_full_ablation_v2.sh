#!/bin/bash
# Full-scale N=8 ablation: Standard vs LIF using train.py (correct config)
# Config: 6L/6H/384d, ~10.65M params, BLOCK_SIZE=256, 5000 iters
# Each --compare run trains Standard + LIF sequentially (~28min each on MPS)
# Total: 8 seeds × 2 models × ~28min = ~7.5 hours

set -e

cd ~/Documents/TsubasaWorkspace/ember

SEEDS=(42 668 1337 2024 314 777 1234 99)

echo "=============================================="
echo "Full-scale N=8 ablation (train.py --compare)"
echo "Config: 6L/6H/384d, BLOCK_SIZE=256, 5000 iters"
echo "Seeds: ${SEEDS[@]}"
echo "Started: $(date)"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Seed $seed starting at $(date)"
    python3 -u train.py --compare --seed $seed 2>&1
    echo ">>> Seed $seed completed at $(date)"
    echo ""
done

echo "=============================================="
echo "All seeds completed at $(date)"
echo "=============================================="
