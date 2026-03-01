#!/usr/bin/env python3
"""
Statistical analysis of Audio N=6 ablation (Base vs LIF).

Loads all available audio checkpoints and runs:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d effect size
  - Per-seed comparison table
  - Std reduction ratio

Usage:
    python3 analyze_audio_stats.py          # All available seeds
    python3 analyze_audio_stats.py --n6     # Only if all 6 seeds present

2026-02-28 — Tsubasa
"""

import os
import sys
import numpy as np
import torch
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')

ALL_SEEDS = [42, 668, 1337, 2024, 314, 777]


def load_results():
    """Load val_acc from all available audio checkpoints."""
    base_accs = {}
    lif_accs = {}

    for seed in ALL_SEEDS:
        for cond, store in [('base', base_accs), ('lif', lif_accs)]:
            path = os.path.join(OUT_DIR, f'audio_{cond}_s{seed}.pt')
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                store[seed] = {
                    'val_acc': ckpt.get('val_acc', 0),
                    'val_loss': ckpt.get('val_loss', 0),
                    'epoch': ckpt.get('epoch', 0),
                }
    return base_accs, lif_accs


def cohen_d(x, y):
    """Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0


def main():
    require_n6 = '--n6' in sys.argv

    base_accs, lif_accs = load_results()

    # Find seeds with both conditions
    paired_seeds = sorted(set(base_accs.keys()) & set(lif_accs.keys()))

    if require_n6 and len(paired_seeds) < 6:
        print(f"Only {len(paired_seeds)}/6 seeds available. Waiting for more.")
        print(f"Available: {paired_seeds}")
        missing_base = [s for s in ALL_SEEDS if s not in base_accs]
        missing_lif = [s for s in ALL_SEEDS if s not in lif_accs]
        if missing_base:
            print(f"Missing base: {missing_base}")
        if missing_lif:
            print(f"Missing LIF: {missing_lif}")
        return

    if len(paired_seeds) < 2:
        print("Need at least 2 paired seeds for analysis.")
        return

    print("=" * 70)
    print(f"AUDIO ABLATION STATISTICAL ANALYSIS (N={len(paired_seeds)})")
    print("=" * 70)

    # Per-seed table
    print(f"\n{'Seed':>6} {'Base Acc':>10} {'LIF Acc':>10} {'Delta':>10} {'Winner':>8}")
    print("-" * 50)

    base_vals = []
    lif_vals = []
    lif_wins = 0

    for seed in paired_seeds:
        b = base_accs[seed]['val_acc']
        l = lif_accs[seed]['val_acc']
        delta = l - b
        winner = 'LIF' if l > b else 'Base' if b > l else 'Tie'
        if l > b:
            lif_wins += 1
        print(f"{seed:>6} {b:>10.4f} {l:>10.4f} {delta:>+10.4f} {winner:>8}")
        base_vals.append(b)
        lif_vals.append(l)

    base_arr = np.array(base_vals)
    lif_arr = np.array(lif_vals)

    # Summary statistics
    print(f"\n{'':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    print(f"{'Base':>6} {base_arr.mean():>10.4f} {base_arr.std(ddof=1):>10.4f} {base_arr.min():>10.4f} {base_arr.max():>10.4f}")
    print(f"{'LIF':>6} {lif_arr.mean():>10.4f} {lif_arr.std(ddof=1):>10.4f} {lif_arr.min():>10.4f} {lif_arr.max():>10.4f}")

    # Delta
    delta_pct = (lif_arr.mean() - base_arr.mean()) / base_arr.mean() * 100
    std_ratio = base_arr.std(ddof=1) / lif_arr.std(ddof=1) if lif_arr.std(ddof=1) > 0 else float('inf')

    print(f"\nMean Delta: {lif_arr.mean() - base_arr.mean():+.4f} ({delta_pct:+.2f}%)")
    print(f"Std Reduction: {std_ratio:.1f}x (Base {base_arr.std(ddof=1):.4f} → LIF {lif_arr.std(ddof=1):.4f})")
    print(f"LIF wins: {lif_wins}/{len(paired_seeds)}")

    # Statistical tests
    print(f"\n--- Statistical Tests ---")

    if len(paired_seeds) >= 3:
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(lif_arr, base_arr)
        print(f"Paired t-test:     t={t_stat:.3f}, p={t_pval:.4f} {'*' if t_pval < 0.05 else ''}")

        # Wilcoxon signed-rank (needs N>=6 ideally, but can run with N>=3)
        try:
            w_stat, w_pval = stats.wilcoxon(lif_arr - base_arr)
            print(f"Wilcoxon:          W={w_stat:.3f}, p={w_pval:.4f} {'*' if w_pval < 0.05 else ''}")
        except ValueError as e:
            print(f"Wilcoxon:          N/A ({e})")

        # Cohen's d
        d = cohen_d(lif_vals, base_vals)
        d_interp = 'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
        print(f"Cohen's d:         d={d:.3f} ({d_interp})")

        # 95% CI for mean difference
        diff = lif_arr - base_arr
        ci_low = diff.mean() - stats.t.ppf(0.975, len(diff)-1) * diff.std(ddof=1) / np.sqrt(len(diff))
        ci_high = diff.mean() + stats.t.ppf(0.975, len(diff)-1) * diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"95% CI (diff):     [{ci_low:.4f}, {ci_high:.4f}]")
    else:
        print("Need N>=3 for statistical tests.")

    # LaTeX table row (for paper)
    print(f"\n--- LaTeX Table Row ---")
    print(f"Audio & {base_arr.mean():.4f}$\\pm${base_arr.std(ddof=1):.4f} "
          f"& {lif_arr.mean():.4f}$\\pm${lif_arr.std(ddof=1):.4f} "
          f"& {delta_pct:+.2f}\\% & {std_ratio:.1f}$\\times$ \\\\")

    # Special note on seed 1337
    if 1337 in paired_seeds:
        b1337 = base_accs[1337]['val_acc']
        l1337 = lif_accs[1337]['val_acc']
        print(f"\n--- Seed 1337 (Catastrophic Failure Case) ---")
        print(f"Base: {b1337:.4f}, LIF: {l1337:.4f}, Delta: {l1337 - b1337:+.4f}")
        other_base = [base_accs[s]['val_acc'] for s in paired_seeds if s != 1337]
        if other_base:
            other_mean = np.mean(other_base)
            gap = other_mean - b1337
            print(f"Base s1337 gap from other seeds mean: {gap:.4f} ({gap/other_mean*100:.1f}%)")
            print(f"LIF prevented catastrophic failure: {l1337:.4f} ≈ other seeds mean {np.mean([lif_accs[s]['val_acc'] for s in paired_seeds if s != 1337]):.4f}")


if __name__ == '__main__':
    main()
