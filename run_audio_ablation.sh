#!/bin/bash
# Audio Liquid Ember 3-seed ablation (6 runs total)
# Base (CfC-only) vs LIF (CfC+LIF) x 3 seeds
# 2026-02-19 — Tsubasa

cd "$(dirname "$0")"

EPOCHS=15
BATCH=64

echo "=== Audio Liquid Ember Ablation ==="
echo "Epochs: $EPOCHS | Batch: $BATCH"
echo "Started: $(date)"
echo ""

for SEED in 42 668 1337; do
    echo "--- Base seed=$SEED ---"
    python3 train_audio.py --no_lif --seed $SEED --epochs $EPOCHS --batch_size $BATCH 2>&1 | tee out/audio_base_s${SEED}.log
    echo ""

    echo "--- LIF seed=$SEED ---"
    python3 train_audio.py --use_lif --seed $SEED --epochs $EPOCHS --batch_size $BATCH 2>&1 | tee out/audio_lif_s${SEED}.log
    echo ""
done

echo "=== All done: $(date) ==="
echo ""

# Summary
echo "=== RESULTS SUMMARY ==="
for f in out/audio_*.log; do
    tag=$(basename "$f" .log)
    best=$(grep "Best val accuracy" "$f" | tail -1)
    test=$(grep "Test accuracy" "$f" | tail -1)
    echo "  $tag: $best | $test"
done
