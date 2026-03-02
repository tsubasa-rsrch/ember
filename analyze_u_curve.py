#!/usr/bin/env python3
"""
Comprehensive U-Curve Analysis: LIF gating effect vs. embedding width.

Loads all 5-scale 3-seed (xs/small/medium/mid/wide) + Full N=8 results.
Generates a 4-panel publication figure and prints a detailed summary table.

Key finding: LIF helps at small (128d) and large (320d+) widths,
but hurts in the critical zone (192d-256d) — a U-curve in delta%.

2026-03-02 — Tsubasa
"""

import os
import json
import glob
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/results")
FIGURES_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/figures")
THRESHOLD_FILE = os.path.join(RESULTS_DIR, "threshold_3seed.json")

# ---------------------------------------------------------------------------
# Scale definitions
# ---------------------------------------------------------------------------
SCALE_CONFIG = {
    'xs':     {'n_layer': 2, 'n_head': 4, 'n_embd': 128, 'order': 0},
    'small':  {'n_layer': 4, 'n_head': 4, 'n_embd': 192, 'order': 1},
    'medium': {'n_layer': 6, 'n_head': 8, 'n_embd': 256, 'order': 2},
    'mid':    {'n_layer': 6, 'n_head': 8, 'n_embd': 320, 'order': 3},
    'wide':   {'n_layer': 6, 'n_head': 8, 'n_embd': 384, 'order': 4},
}

FULL_CONFIG = {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'order': 5}


def infer_scale(n_params):
    if n_params < 600_000:
        return 'xs'
    elif n_params < 3_000_000:
        return 'small'
    elif n_params < 6_000_000:
        return 'medium'
    elif n_params < 9_000_000:
        return 'mid'
    elif n_params < 15_000_000:
        return 'wide'
    else:
        return 'large'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tiny_results():
    """Load all tiny_*.json results, deduplicate to best per (scale, model, seed)."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "tiny_*.json")))
    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        items = data if isinstance(data, list) else [data]
        all_results.extend(items)

    # Filter: only runs with sufficient training (>= 2800 iters)
    best = {}
    for r in all_results:
        if not r.get('history'):
            continue
        max_iter = max(h['iter'] for h in r['history'])
        if max_iter < 2800:
            continue
        scale = infer_scale(r['n_params'])
        mt = r['model_type']
        seed = r['seed']
        key = (scale, mt, seed)
        if key not in best or r['best_val_loss'] < best[key]['best_val_loss']:
            best[key] = r
            best[key]['_scale'] = scale
    return best


def load_full_results():
    """Load full_*.json (N=8 ablation at 384d, 6L/6H)."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "full_*.json")))
    standard, lif = {}, {}
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        items = data if isinstance(data, list) else [data]
        for d in items:
            mt = d.get('model_type', '')
            seed = d.get('seed')
            val = d.get('best_val_loss')
            if val is None or seed is None:
                continue
            if mt == 'standard':
                standard[seed] = val
            elif mt == 'lif':
                lif[seed] = val
    return standard, lif


def load_threshold_data():
    """Load threshold_3seed.json if available."""
    if not os.path.exists(THRESHOLD_FILE):
        return None
    with open(THRESHOLD_FILE) as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_scale_stats(best_runs, target_scales=None):
    """Compute per-scale statistics for standard vs LIF."""
    if target_scales is None:
        target_scales = ['xs', 'small', 'medium', 'mid', 'wide']

    results = []
    for scale in target_scales:
        cfg = SCALE_CONFIG.get(scale, {})
        width = cfg.get('n_embd', 0)
        n_layer = cfg.get('n_layer', 0)
        n_head = cfg.get('n_head', 0)

        # Collect all seeds for this scale
        seeds_std = sorted([k[2] for k in best_runs if k[0] == scale and k[1] == 'standard'])
        seeds_lif = sorted([k[2] for k in best_runs if k[0] == scale and k[1] == 'lif'])
        paired_seeds = sorted(set(seeds_std) & set(seeds_lif))

        if len(paired_seeds) < 2:
            continue

        std_vals = np.array([best_runs[(scale, 'standard', s)]['best_val_loss'] for s in paired_seeds])
        lif_vals = np.array([best_runs[(scale, 'lif', s)]['best_val_loss'] for s in paired_seeds])
        n_params_std = best_runs[(scale, 'standard', paired_seeds[0])]['n_params']
        n_params_lif = best_runs[(scale, 'lif', paired_seeds[0])]['n_params']
        lif_param_overhead = n_params_lif - n_params_std

        # Delta % per seed
        delta_pcts = (lif_vals - std_vals) / std_vals * 100
        mean_delta_pct = np.mean(delta_pcts)
        std_delta_pct = np.std(delta_pcts, ddof=1) if len(delta_pcts) > 1 else 0.0

        # Variance ratio (Std_std / LIF_std)
        std_stdev = np.std(std_vals, ddof=1)
        lif_stdev = np.std(lif_vals, ddof=1)
        var_ratio = std_stdev / lif_stdev if lif_stdev > 0 else float('inf')

        # Effect size (Cohen's d, paired)
        diffs = lif_vals - std_vals
        d_std = np.std(diffs, ddof=1) if len(diffs) > 1 else 1e-8
        cohens_d = np.mean(diffs) / d_std if d_std > 0 else 0.0

        # Paired t-test (if enough seeds)
        t_stat, p_val = (np.nan, np.nan)
        if len(paired_seeds) >= 3:
            t_stat, p_val = stats.ttest_rel(std_vals, lif_vals)

        # LIF win rate
        lif_wins = int(np.sum(lif_vals < std_vals))

        results.append({
            'scale': scale,
            'width': width,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_seeds': len(paired_seeds),
            'seeds': paired_seeds,
            'n_params_std': n_params_std,
            'n_params_lif': n_params_lif,
            'lif_overhead': lif_param_overhead,
            'std_mean': np.mean(std_vals),
            'std_stdev': std_stdev,
            'lif_mean': np.mean(lif_vals),
            'lif_stdev': lif_stdev,
            'mean_delta_pct': mean_delta_pct,
            'std_delta_pct': std_delta_pct,
            'seed_deltas': delta_pcts.tolist(),
            'var_ratio': var_ratio,
            'cohens_d': cohens_d,
            't_stat': t_stat,
            'p_val': p_val,
            'lif_wins': lif_wins,
        })

    return results


def compute_full_stats(std_dict, lif_dict):
    """Compute stats for the Full N=8 ablation."""
    paired = sorted(set(std_dict.keys()) & set(lif_dict.keys()))
    if len(paired) < 2:
        return None

    std_vals = np.array([std_dict[s] for s in paired])
    lif_vals = np.array([lif_dict[s] for s in paired])

    delta_pcts = (lif_vals - std_vals) / std_vals * 100
    diffs = lif_vals - std_vals
    std_stdev = np.std(std_vals, ddof=1)
    lif_stdev = np.std(lif_vals, ddof=1)

    t_stat, p_val = stats.ttest_rel(std_vals, lif_vals)
    d_std = np.std(diffs, ddof=1)
    cohens_d = np.mean(diffs) / d_std if d_std > 0 else 0.0

    return {
        'scale': 'full_N8',
        'width': 384,
        'n_layer': 6,
        'n_head': 6,
        'n_seeds': len(paired),
        'seeds': paired,
        'n_params_std': 10745088,
        'n_params_lif': 10745196,
        'lif_overhead': 108,
        'std_mean': np.mean(std_vals),
        'std_stdev': std_stdev,
        'lif_mean': np.mean(lif_vals),
        'lif_stdev': lif_stdev,
        'mean_delta_pct': np.mean(delta_pcts),
        'std_delta_pct': np.std(delta_pcts, ddof=1),
        'seed_deltas': delta_pcts.tolist(),
        'var_ratio': std_stdev / lif_stdev if lif_stdev > 0 else float('inf'),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_val': p_val,
        'lif_wins': int(np.sum(lif_vals < std_vals)),
    }


def analyze_thresholds(threshold_data):
    """Analyze threshold statistics from the XS 3-seed threshold file."""
    if threshold_data is None:
        return None

    stats_per_seed = {}
    for seed_str, seed_data in threshold_data.items():
        all_thresholds = []
        for param_name, values in seed_data['params'].items():
            if 'threshold' in param_name:
                all_thresholds.extend(values)
        stats_per_seed[seed_str] = {
            'mean': np.mean(all_thresholds),
            'std': np.std(all_thresholds),
            'abs_mean': np.mean(np.abs(all_thresholds)),
            'max': np.max(all_thresholds),
            'min': np.min(all_thresholds),
            'n_near_zero': sum(1 for t in all_thresholds if abs(t) < 0.01),
            'n_active': sum(1 for t in all_thresholds if abs(t) > 0.1),
            'total': len(all_thresholds),
        }
    return stats_per_seed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_4panel(scale_stats, full_stat, threshold_stats):
    """Generate the 4-panel publication figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LIF Gating: Width-Dependent U-Curve Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Color palette
    BLUE = '#1565C0'
    LIGHT_BLUE = '#90CAF9'
    RED = '#D32F2F'
    GREEN = '#388E3C'
    ORANGE = '#F57C00'
    PURPLE = '#7B1FA2'
    GRAY = '#616161'

    # ---- Panel 1: Delta% vs Width (the U-curve) ----
    ax = axes[0, 0]
    widths = [s['width'] for s in scale_stats]
    mean_deltas = [s['mean_delta_pct'] for s in scale_stats]
    std_deltas = [s['std_delta_pct'] for s in scale_stats]
    n_seeds_list = [s['n_seeds'] for s in scale_stats]

    # Error bars = SEM
    sems = [sd / np.sqrt(n) for sd, n in zip(std_deltas, n_seeds_list)]

    # Individual seed scatter
    for s in scale_stats:
        for d in s['seed_deltas']:
            ax.scatter(s['width'], d, color=LIGHT_BLUE, s=30, zorder=3, alpha=0.6, edgecolors='none')

    # Mean with SEM error bars
    ax.errorbar(widths, mean_deltas, yerr=sems, color=BLUE, linewidth=2.5,
                marker='o', markersize=9, capsize=5, capthick=2, zorder=5,
                label=f'Tiny 3-seed mean')

    # Full-scale N=8 point (if available)
    if full_stat:
        full_sem = full_stat['std_delta_pct'] / np.sqrt(full_stat['n_seeds'])
        ax.errorbar([full_stat['width'] + 8], [full_stat['mean_delta_pct']], yerr=[full_sem],
                    color=RED, linewidth=2, marker='D', markersize=9, capsize=5, capthick=2,
                    zorder=5, label=f'Full N={full_stat["n_seeds"]} (6L/6H/384d)')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1)

    # Shade improvement vs degradation
    ax.fill_between([100, 450], 0, -1.5, alpha=0.04, color='green')
    ax.fill_between([100, 450], 0, 1.5, alpha=0.04, color='red')

    # Annotations
    for s in scale_stats:
        label = f"{s['scale'].upper()}\n{s['n_layer']}L/{s['n_head']}H"
        offset_y = 14 if s['mean_delta_pct'] >= 0 else -20
        ax.annotate(f"{label}\n{s['mean_delta_pct']:.2f}%",
                    (s['width'], s['mean_delta_pct']),
                    textcoords='offset points', xytext=(0, offset_y),
                    ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_title('Panel A: LIF Effect vs. Embedding Width', fontsize=12, fontweight='bold')
    ax.set_xlabel('Embedding Dimension (d)', fontsize=11)
    ax.set_ylabel('Val Loss Delta (%, negative = better)', fontsize=11)
    ax.set_xlim(100, 420)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # ---- Panel 2: Variance Ratio vs Width ----
    ax = axes[0, 1]
    var_ratios = [s['var_ratio'] for s in scale_stats]

    ax.bar(widths, var_ratios, width=30, color=ORANGE, alpha=0.7, edgecolor='gray', zorder=3,
           label='Tiny 3-seed')

    if full_stat:
        ax.bar(full_stat['width'] + 35, full_stat['var_ratio'], width=30,
               color=RED, alpha=0.7, edgecolor='gray', zorder=3,
               label=f'Full N={full_stat["n_seeds"]}')

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1,
               label='Var ratio = 1 (no effect)')

    for s in scale_stats:
        ax.annotate(f"{s['var_ratio']:.2f}x",
                    (s['width'], s['var_ratio']),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9, fontweight='bold')
    if full_stat:
        ax.annotate(f"{full_stat['var_ratio']:.2f}x",
                    (full_stat['width'] + 35, full_stat['var_ratio']),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9, fontweight='bold', color=RED)

    ax.set_title('Panel B: Std(Standard) / Std(LIF) by Width', fontsize=12, fontweight='bold')
    ax.set_xlabel('Embedding Dimension (d)', fontsize=11)
    ax.set_ylabel('Variance Ratio (>1 = LIF more stable)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # ---- Panel 3: LIF Parameter Count vs Width ----
    ax = axes[1, 0]

    lif_overheads = [s['lif_overhead'] for s in scale_stats]
    total_params = [s['n_params_std'] for s in scale_stats]
    overhead_pct = [s['lif_overhead'] / s['n_params_std'] * 100 for s in scale_stats]

    # Twin axes: absolute count + percentage
    bars = ax.bar(widths, lif_overheads, width=30, color=PURPLE, alpha=0.7, edgecolor='gray', zorder=3)

    for s, ovh, pct in zip(scale_stats, lif_overheads, overhead_pct):
        ax.annotate(f"{ovh}\n({pct:.3f}%)",
                    (s['width'], ovh),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9, fontweight='bold')

    if full_stat:
        ax.bar(full_stat['width'] + 35, full_stat['lif_overhead'], width=30,
               color=RED, alpha=0.7, edgecolor='gray', zorder=3)
        pct_full = full_stat['lif_overhead'] / full_stat['n_params_std'] * 100
        ax.annotate(f"{full_stat['lif_overhead']}\n({pct_full:.4f}%)",
                    (full_stat['width'] + 35, full_stat['lif_overhead']),
                    textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9, fontweight='bold', color=RED)

    # Formula annotation
    ax.text(0.02, 0.95, 'LIF params = n_layer * n_head * 3\n(threshold, leak, steepness per head)',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Panel C: LIF Parameter Overhead by Width', fontsize=12, fontweight='bold')
    ax.set_xlabel('Embedding Dimension (d)', fontsize=11)
    ax.set_ylabel('LIF Parameters Added', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    # ---- Panel 4: Threshold Statistics (XS only) OR Effect Size ----
    ax = axes[1, 1]

    if threshold_stats:
        # Plot threshold distribution from the xs 3-seed data
        seeds = sorted(threshold_stats.keys())
        x_pos = np.arange(len(seeds))
        abs_means = [threshold_stats[s]['abs_mean'] for s in seeds]
        n_near_zero = [threshold_stats[s]['n_near_zero'] for s in seeds]
        n_active = [threshold_stats[s]['n_active'] for s in seeds]
        totals = [threshold_stats[s]['total'] for s in seeds]

        bar_w = 0.3
        ax.bar(x_pos - bar_w, abs_means, bar_w, color=BLUE, alpha=0.7, label='|threshold| mean')
        ax.bar(x_pos, [na/t*100 for na, t in zip(n_active, totals)], bar_w,
               color=GREEN, alpha=0.7, label='% active (|t|>0.1)')
        ax.bar(x_pos + bar_w, [nz/t*100 for nz, t in zip(n_near_zero, totals)], bar_w,
               color=GRAY, alpha=0.7, label='% near-zero (|t|<0.01)')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'seed {s}' for s in seeds])
        ax.set_title('Panel D: Threshold Statistics (XS, 2L/4H)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value / Percentage', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2, axis='y')

        # Detail annotation
        detail_lines = []
        for s in seeds:
            ts = threshold_stats[s]
            detail_lines.append(f"seed {s}: {ts['n_active']}/{ts['total']} active, "
                               f"range [{ts['min']:.3f}, {ts['max']:.3f}]")
        ax.text(0.02, 0.15, '\n'.join(detail_lines),
                transform=ax.transAxes, fontsize=7, va='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        # Fallback: Effect size (Cohen's d) across scales
        cohens_ds = [s['cohens_d'] for s in scale_stats]
        colors_bar = [GREEN if d < 0 else RED for d in cohens_ds]

        ax.bar(widths, cohens_ds, width=30, color=colors_bar, alpha=0.7, edgecolor='gray', zorder=3)

        if full_stat:
            c = GREEN if full_stat['cohens_d'] < 0 else RED
            ax.bar(full_stat['width'] + 35, full_stat['cohens_d'], width=30,
                   color=c, alpha=0.7, edgecolor='gray', zorder=3)

        for s in scale_stats:
            ax.annotate(f"{s['cohens_d']:.2f}",
                        (s['width'], s['cohens_d']),
                        textcoords='offset points',
                        xytext=(0, 6 if s['cohens_d'] >= 0 else -14),
                        ha='center', fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title('Panel D: Effect Size (Cohen\'s d, paired)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Embedding Dimension (d)', fontsize=11)
        ax.set_ylabel("Cohen's d (negative = LIF better)", fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, 'u_curve_analysis.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(scale_stats, full_stat):
    """Print comprehensive summary table."""
    all_stats = scale_stats.copy()
    if full_stat:
        all_stats.append(full_stat)

    print("=" * 120)
    print("EMBER LIF WIDTH U-CURVE: COMPREHENSIVE ANALYSIS")
    print("=" * 120)

    # Header
    print(f"\n{'Scale':<10} {'Width':>5} {'Arch':>8} {'N':>3} {'Std Mean':>10} {'LIF Mean':>10} "
          f"{'Delta%':>8} {'SEM':>7} {'VarRatio':>9} {'Cohen_d':>8} {'t-stat':>7} {'p-val':>7} "
          f"{'LIF Wins':>9} {'LIF Params':>10}")
    print("-" * 120)

    for s in all_stats:
        arch = f"{s['n_layer']}L/{s['n_head']}H"
        sem = s['std_delta_pct'] / np.sqrt(s['n_seeds']) if s['n_seeds'] > 0 else 0
        p_str = f"{s['p_val']:.4f}" if not np.isnan(s['p_val']) else "  N/A"
        t_str = f"{s['t_stat']:.3f}" if not np.isnan(s['t_stat']) else "  N/A"

        print(f"{s['scale']:<10} {s['width']:>5}d {arch:>8} {s['n_seeds']:>3} "
              f"{s['std_mean']:>10.6f} {s['lif_mean']:>10.6f} "
              f"{s['mean_delta_pct']:>+7.3f}% {sem:>6.3f}% "
              f"{s['var_ratio']:>8.2f}x {s['cohens_d']:>+7.2f} "
              f"{t_str:>7} {p_str:>7} "
              f"{s['lif_wins']}/{s['n_seeds']:>2}      "
              f"{s['lif_overhead']:>10}")

    # Summary statistics
    print("\n" + "=" * 120)
    print("INTERPRETATION:")
    print("-" * 120)

    # Find the valley and edges
    tiny_stats = [s for s in all_stats if s['scale'] != 'full_N8']
    if tiny_stats:
        valley = max(tiny_stats, key=lambda s: s['mean_delta_pct'])
        best = min(tiny_stats, key=lambda s: s['mean_delta_pct'])
        print(f"  Worst LIF effect (U-curve valley): {valley['scale'].upper()} ({valley['width']}d) = {valley['mean_delta_pct']:+.3f}%")
        print(f"  Best LIF effect:                   {best['scale'].upper()} ({best['width']}d) = {best['mean_delta_pct']:+.3f}%")
        print(f"  Valley-to-best spread:             {valley['mean_delta_pct'] - best['mean_delta_pct']:.3f} percentage points")

    # Critical width zone
    hurting = [s for s in tiny_stats if s['mean_delta_pct'] > 0]
    helping = [s for s in tiny_stats if s['mean_delta_pct'] < 0]
    if hurting and helping:
        hurt_widths = sorted([s['width'] for s in hurting])
        help_widths = sorted([s['width'] for s in helping])
        print(f"\n  LIF HURTS at widths: {hurt_widths} (under-parameterized for LIF overhead?)")
        print(f"  LIF HELPS at widths: {help_widths} (sufficient capacity for gating benefit)")
        print(f"  Critical zone boundary: ~{min(hurt_widths)}d to ~{max(hurt_widths)}d")

    # Variance stabilization
    stabilizing = [s for s in tiny_stats if s['var_ratio'] > 1.0]
    if stabilizing:
        print(f"\n  Variance stabilization (ratio > 1.0): {[s['scale'].upper() for s in stabilizing]}")

    # Statistical significance
    sig = [s for s in all_stats if not np.isnan(s['p_val']) and s['p_val'] < 0.05]
    if sig:
        pairs = [(s['scale'].upper(), f"p={s['p_val']:.4f}") for s in sig]
        print(f"\n  Statistically significant (p < 0.05): {pairs}")
    else:
        print(f"\n  No scale reaches p < 0.05 (3-seed limited power; Full N=8 needed)")

    # Per-seed detail
    print("\n" + "=" * 120)
    print("PER-SEED DETAIL:")
    print("-" * 120)
    for s in all_stats:
        print(f"  {s['scale']:<10} seeds={s['seeds']}  deltas={[f'{d:+.3f}%' for d in s['seed_deltas']]}")

    # LaTeX table
    print("\n" + "=" * 120)
    print("LaTeX TABLE (for paper):")
    print("-" * 120)
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(r"Scale & Width & Arch & $N$ & Standard & LIF & $\Delta\%$ & Var Ratio & Cohen's $d$ \\")
    print(r"\midrule")
    for s in all_stats:
        arch = f"{s['n_layer']}L/{s['n_head']}H"
        sem = s['std_delta_pct'] / np.sqrt(s['n_seeds'])
        sig_mark = "*" if (not np.isnan(s['p_val']) and s['p_val'] < 0.05) else ""
        print(f"{s['scale'].upper()} & {s['width']}d & {arch} & {s['n_seeds']} & "
              f"${s['std_mean']:.4f} \\pm {s['std_stdev']:.4f}$ & "
              f"${s['lif_mean']:.4f} \\pm {s['lif_stdev']:.4f}$ & "
              f"${s['mean_delta_pct']:+.2f}\\%{sig_mark}$ & "
              f"${s['var_ratio']:.2f}\\times$ & "
              f"${s['cohens_d']:+.2f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading tiny-scale results...")
    best_runs = load_tiny_results()
    print(f"  Loaded {len(best_runs)} best runs")

    print("Loading full-scale N=8 results...")
    full_std, full_lif = load_full_results()
    print(f"  Found {len(full_std)} Standard, {len(full_lif)} LIF seeds")

    print("Loading threshold data...")
    threshold_data = load_threshold_data()
    threshold_stats = analyze_thresholds(threshold_data)
    if threshold_stats:
        print(f"  Loaded threshold data for seeds: {list(threshold_stats.keys())}")
    else:
        print("  No threshold data found")

    # Compute stats
    scale_stats = compute_scale_stats(best_runs)
    full_stat = compute_full_stats(full_std, full_lif)

    # Print summary
    print_summary(scale_stats, full_stat)

    # Generate figure
    print("\nGenerating 4-panel figure...")
    plot_4panel(scale_stats, full_stat, threshold_stats)

    print("\nDone!")


if __name__ == '__main__':
    main()
