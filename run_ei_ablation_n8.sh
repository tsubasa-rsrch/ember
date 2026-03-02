#!/bin/bash
# E/I Balance N≥8 Validation
# Runs Head-persist + Refrac+Head for 7 seeds (1337 already done)
# Uses existing N=8 Standard baselines
# Estimated: ~10 hours total

set -e

SEEDS=(42 668 2024 314 777 1234 99)
LOGFILE="/tmp/ei_ablation_n8.log"

cd "$(dirname "$0")"

echo "============================================================" > "$LOGFILE"
echo "E/I BALANCE N>=8 VALIDATION" >> "$LOGFILE"
echo "Started: $(date)" >> "$LOGFILE"
echo "Seeds: ${SEEDS[*]}" >> "$LOGFILE"
echo "Conditions: Head-persist, Refrac+Head" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"

COMPLETED=0
TOTAL=$((${#SEEDS[@]} * 2))

for SEED in "${SEEDS[@]}"; do
    echo "" >> "$LOGFILE"
    echo "========== SEED $SEED ==========" >> "$LOGFILE"

    # Condition 1: Head-persist (LIF + head persistent state)
    echo ">>> [$SEED] Head-persist starting at $(date)" >> "$LOGFILE"
    python3 -u train.py --head-persistent --seed "$SEED" >> "$LOGFILE" 2>&1
    COMPLETED=$((COMPLETED + 1))
    echo ">>> [$SEED] Head-persist DONE ($COMPLETED/$TOTAL) at $(date)" >> "$LOGFILE"

    # Condition 2: Refrac+Head (LIF refractory + head persistent)
    echo ">>> [$SEED] Refrac+Head starting at $(date)" >> "$LOGFILE"
    python3 -u train.py --refractory --head-persistent --seed "$SEED" >> "$LOGFILE" 2>&1
    COMPLETED=$((COMPLETED + 1))
    echo ">>> [$SEED] Refrac+Head DONE ($COMPLETED/$TOTAL) at $(date)" >> "$LOGFILE"

    echo ">>> SEED $SEED complete. Progress: $COMPLETED/$TOTAL" >> "$LOGFILE"
done

echo "" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"
echo "ALL DONE at $(date)" >> "$LOGFILE"
echo "$COMPLETED/$TOTAL conditions completed" >> "$LOGFILE"
echo "============================================================" >> "$LOGFILE"
