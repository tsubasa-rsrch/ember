#!/usr/bin/env python3
"""Analyze full-scale N=8 ablation results (Standard vs LIF)."""

import json
import glob
import numpy as np
from scipy import stats

def load_results():
    """Load all full-scale results from the ablation run."""
    # Match both old (tiny_) and new (full_) format result files
    files = sorted(glob.glob('results/full_*.json') + glob.glob('results/tiny_20260301_*.json'))

    standard = {}
    lif = {}

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        items = data if isinstance(data, list) else [data]
        for d in items:
            model = d.get('model_type', '')
            seed = d.get('seed')
            val = d.get('best_val_loss')
            params = d.get('n_params', 0)

            if val is None or seed is None:
                continue

            if model == 'standard':
                standard[seed] = val
            elif model == 'lif':
                lif[seed] = val

    return standard, lif

def analyze(standard, lif):
    """Run statistical analysis."""
    seeds = sorted(set(standard.keys()) & set(lif.keys()))

    if len(seeds) < 2:
        print(f"Only {len(seeds)} paired seeds found. Need at least 2 for analysis.")
        print(f"Standard seeds: {sorted(standard.keys())}")
        print(f"LIF seeds: {sorted(lif.keys())}")
        return

    std_vals = np.array([standard[s] for s in seeds])
    lif_vals = np.array([lif[s] for s in seeds])
    diffs = lif_vals - std_vals

    print("=" * 70)
    print(f"FULL-SCALE N={len(seeds)} ABLATION ANALYSIS")
    print(f"Config: 6L/6H/384d, ~10.65M params, 5000 iterations")
    print("=" * 70)

    print(f"\nSeeds: {seeds}")
    print(f"\nPer-seed results:")
    print(f"{'Seed':>8} {'Standard':>10} {'LIF':>10} {'Delta':>10} {'Winner':>8}")
    print("-" * 50)
    lif_wins = 0
    for s in seeds:
        d = lif[s] - standard[s]
        pct = d / standard[s] * 100
        w = "LIF" if d < 0 else ("Std" if d > 0 else "Tie")
        if d < 0:
            lif_wins += 1
        print(f"{s:>8} {standard[s]:>10.4f} {lif[s]:>10.4f} {pct:>+9.2f}% {w:>8}")

    std_mean = np.mean(std_vals)
    std_std = np.std(std_vals, ddof=1)
    lif_mean = np.mean(lif_vals)
    lif_std = np.std(lif_vals, ddof=1)
    delta_pct = (lif_mean - std_mean) / std_mean * 100
    var_ratio = std_std / lif_std if lif_std > 0 else float('inf')

    print(f"\n{'Summary':}")
    print(f"  Standard: {std_mean:.4f} ± {std_std:.4f}")
    print(f"  LIF:      {lif_mean:.4f} ± {lif_std:.4f}")
    print(f"  Delta:    {delta_pct:+.2f}%")
    print(f"  Var reduction: {var_ratio:.1f}×")
    print(f"  LIF wins: {lif_wins}/{len(seeds)}")

    # Paired t-test
    t_stat, p_t = stats.ttest_rel(std_vals, lif_vals)
    print(f"\nPaired t-test: t = {t_stat:.3f}, p = {p_t:.4f}")

    # Wilcoxon signed-rank
    try:
        w_stat, p_w = stats.wilcoxon(diffs)
        print(f"Wilcoxon signed-rank: W = {w_stat:.1f}, p = {p_w:.4f}")
    except ValueError as e:
        print(f"Wilcoxon: {e}")

    # Cohen's d (paired)
    d_mean = np.mean(diffs)
    d_std = np.std(diffs, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0
    print(f"Cohen's d (paired): {cohens_d:.2f}")

    # For LaTeX table
    print(f"\n{'='*70}")
    print("FOR PAPER (LaTeX):")
    print(f"Standard: ${std_mean:.4f} \\pm {std_std:.4f}$")
    print(f"LIF:      ${lif_mean:.4f} \\pm {lif_std:.4f}$")
    print(f"Delta:    ${delta_pct:+.2f}\\%$")
    print(f"Seeds:    {lif_wins}/{len(seeds)} LIF wins")
    print(f"Stats:    $t = {t_stat:.2f}$, $p = {p_t:.3f}$, $d = {cohens_d:.2f}$")
    print(f"Var:      {var_ratio:.1f}× reduction")


if __name__ == '__main__':
    standard, lif = load_results()

    n_std = len(standard)
    n_lif = len(lif)
    print(f"Found: {n_std} Standard results, {n_lif} LIF results")

    if n_std == 0 and n_lif == 0:
        print("No results yet. Runs still in progress.")
    else:
        analyze(standard, lif)
