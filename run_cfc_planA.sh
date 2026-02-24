#!/bin/bash
# Plan A: CfC experiments for Paper 1
# xs 3-seed (CfC-only + CfC+LIF) + wide 1-seed (CfC-only + CfC+LIF)
# Estimated: ~13 hours total
# Started: $(date)

set -e
cd ~/Documents/TsubasaWorkspace/ember

echo "=========================================="
echo " Plan A: CfC Experiments"
echo " Started: $(date)"
echo "=========================================="

# Phase 1: xs 3-seed — ~46 minutes (23min × 2 models)
echo ""
echo ">>> Phase 1: xs 3-seed CfC-only"
python3 train_tiny.py --model cfc_base --scale xs --ablation
echo ">>> Phase 1: xs 3-seed CfC+LIF"
python3 train_tiny.py --model cfc --scale xs --ablation

# Phase 2: wide 1-seed — ~12.2 hours (6.1h × 2 models)
echo ""
echo ">>> Phase 2: wide 1-seed CfC-only"
python3 train_tiny.py --model cfc_base --scale wide --seed 42
echo ">>> Phase 2: wide 1-seed CfC+LIF"
python3 train_tiny.py --model cfc --scale wide --seed 42

echo ""
echo "=========================================="
echo " Plan A COMPLETE: $(date)"
echo "=========================================="
