"""
Plot Ember-Tiny training curves for cross-scale comparison.
Generates publication-quality figures for Paper 1.

Usage:
    python3 plot_tiny_curves.py              # Plot all available scales
    python3 plot_tiny_curves.py --scale xs   # Plot single scale
"""

import os
import json
import glob
import argparse
import numpy as np

RESULTS_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/results")
FIGURES_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/figures")


def load_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "tiny_*.json")))
    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        all_results.extend(data)
    return all_results


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


def get_best_runs(results, min_iters=2800):
    """Get best run per (scale, model_type, seed), filtered by min iters."""
    best = {}
    for r in results:
        if not r.get('history'):
            continue
        max_iter = max(h['iter'] for h in r['history'])
        if max_iter < min_iters:
            continue
        scale = infer_scale(r['n_params'])
        key = (scale, r['model_type'], r['seed'])
        if key not in best or r['best_val_loss'] < best[key]['best_val_loss']:
            r['scale'] = scale
            best[key] = r
    return best


def plot_scale(best_runs, scale, ax=None):
    """Plot training curves for a single scale."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {'standard': '#2196F3', 'lif': '#FF5722', 'cfc': '#4CAF50'}
    labels = {'standard': 'Standard', 'lif': 'Ember (LIF)', 'cfc': 'Liquid Ember (CfC+LIF)'}

    seeds = sorted(set(k[2] for k in best_runs if k[0] == scale))

    for mt in ['standard', 'lif', 'cfc']:
        all_iters = []
        all_vals = []

        for seed in seeds:
            key = (scale, mt, seed)
            if key not in best_runs:
                continue
            hist = best_runs[key]['history']
            iters = [h['iter'] for h in hist]
            vals = [h['val_loss'] for h in hist]
            all_iters.append(iters)
            all_vals.append(vals)

            # Light individual seed lines
            ax.plot(iters, vals, color=colors[mt], alpha=0.15, linewidth=0.8)

        if not all_vals:
            continue

        # Mean line
        common_iters = all_iters[0]  # assume all have same iters
        mean_vals = np.mean(all_vals, axis=0)
        ax.plot(common_iters, mean_vals, color=colors[mt], linewidth=2.0,
                label=f"{labels[mt]} (mean)", marker='o', markersize=3)

    n_layers = {'xs': 2, 'small': 4, 'medium': 6, 'mid': 6, 'wide': 6}.get(scale, '?')
    n_params = None
    for k, v in best_runs.items():
        if k[0] == scale and k[1] == 'standard':
            n_params = v['n_params']
            break

    n_embd = {'xs': 128, 'small': 192, 'medium': 256, 'mid': 320, 'wide': 384}.get(scale, '?')
    title = f"Scale: {scale.upper()} ({n_layers}L/{n_embd}d, {n_params/1e6:.2f}M)" if n_params else f"Scale: {scale.upper()}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Training Iteration', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_delta(best_runs, ax=None):
    """Plot LIF-Standard delta across scales."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    SCALE_ORDER = {'xs': 0, 'small': 1, 'medium': 2, 'mid': 3, 'wide': 4, 'large': 5}
    SCALE_COLORS = {'xs': '#2196F3', 'small': '#FF9800', 'medium': '#9C27B0',
                    'mid': '#E91E63', 'wide': '#4CAF50', 'large': '#607D8B'}
    SCALE_LAYERS = {'xs': 2, 'small': 4, 'medium': 6, 'mid': 6, 'wide': 6}

    scales = sorted(set(k[0] for k in best_runs), key=lambda s: SCALE_ORDER.get(s, 99))
    seeds = sorted(set(k[2] for k in best_runs))

    for scale in scales:
        n_layers = SCALE_LAYERS.get(scale, '?')

        for seed in seeds:
            std_key = (scale, 'standard', seed)
            lif_key = (scale, 'lif', seed)
            if std_key not in best_runs or lif_key not in best_runs:
                continue

            std_hist = best_runs[std_key]['history']
            lif_hist = best_runs[lif_key]['history']

            iters = [h['iter'] for h in std_hist]
            std_vals = [h['val_loss'] for h in std_hist]
            lif_vals = [h['val_loss'] for h in lif_hist]

            # Delta = LIF - Standard (negative = LIF better)
            delta = [l - s for l, s in zip(lif_vals, std_vals)]
            ax.plot(iters, delta, alpha=0.3, linewidth=0.8,
                    color=SCALE_COLORS.get(scale, 'gray'))

        # Mean delta
        all_deltas = []
        for seed in seeds:
            std_key = (scale, 'standard', seed)
            lif_key = (scale, 'lif', seed)
            if std_key in best_runs and lif_key in best_runs:
                std_vals = [h['val_loss'] for h in best_runs[std_key]['history']]
                lif_vals = [h['val_loss'] for h in best_runs[lif_key]['history']]
                all_deltas.append([l - s for l, s in zip(lif_vals, std_vals)])

        if all_deltas:
            mean_delta = np.mean(all_deltas, axis=0)
            iters = [h['iter'] for h in best_runs[(scale, 'standard', seeds[0])]['history']]
            ax.plot(iters, mean_delta, linewidth=2.5,
                    color=SCALE_COLORS.get(scale, 'gray'),
                    label=f"{scale.upper()} ({n_layers}L)", marker='o', markersize=3)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('LIF Effect Over Training (negative = LIF better)', fontsize=13)
    ax.set_xlabel('Training Iteration', fontsize=11)
    ax.set_ylabel('Val Loss Delta (LIF - Standard)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return ax


def plot_u_curve(best_runs, ax=None):
    """Plot final LIF delta% vs embedding width — the paper's key figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    SCALE_ORDER = {'xs': 0, 'small': 1, 'medium': 2, 'mid': 3, 'wide': 4, 'large': 5}
    WIDTHS = {'xs': 128, 'small': 192, 'medium': 256, 'mid': 320, 'wide': 384}

    scales = sorted(set(k[0] for k in best_runs), key=lambda s: SCALE_ORDER.get(s, 99))
    seeds = sorted(set(k[2] for k in best_runs))

    widths = []
    mean_deltas = []
    all_seed_deltas = []  # for scatter

    for scale in scales:
        if scale not in WIDTHS:
            continue
        seed_deltas = []
        for seed in seeds:
            std_key = (scale, 'standard', seed)
            lif_key = (scale, 'lif', seed)
            if std_key in best_runs and lif_key in best_runs:
                std_val = best_runs[std_key]['best_val_loss']
                lif_val = best_runs[lif_key]['best_val_loss']
                delta_pct = (lif_val - std_val) / std_val * 100
                seed_deltas.append(delta_pct)

        if seed_deltas:
            w = WIDTHS[scale]
            widths.append(w)
            mean_deltas.append(np.mean(seed_deltas))
            all_seed_deltas.append((w, seed_deltas))

    # Individual seeds as scatter
    for w, deltas in all_seed_deltas:
        ax.scatter([w] * len(deltas), deltas, color='#90CAF9', s=40, zorder=3, alpha=0.7)

    # Mean line
    ax.plot(widths, mean_deltas, color='#1565C0', linewidth=2.5, marker='o',
            markersize=8, zorder=4, label='Mean LIF effect (%)')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Shade regions
    ax.axvspan(0, 260, alpha=0.05, color='red', label='LIF hurts (under-parameterized)')
    ax.axvspan(260, 500, alpha=0.05, color='green', label='LIF helps (sufficient width)')

    # Annotations
    for w, d in zip(widths, mean_deltas):
        label = {128: 'XS\n2L', 192: 'S\n4L', 256: 'M\n6L', 320: 'Mid\n6L', 384: 'W\n6L'}.get(w, '')
        ax.annotate(f'{label}\n{d:.2f}%', (w, d), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=9, fontweight='bold')

    ax.set_title('LIF Effect vs. Embedding Width (U-Curve)', fontsize=14)
    ax.set_xlabel('Embedding Dimension (d)', fontsize=12)
    ax.set_ylabel('Val Loss Delta (%, negative = LIF better)', fontsize=12)
    ax.set_xlim(100, 420)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    return ax


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, default=None)
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No results found!")
        return

    best_runs = get_best_runs(results)
    scales = sorted(set(k[0] for k in best_runs),
                    key=lambda s: {'xs': 0, 'small': 1, 'medium': 2, 'mid': 3, 'wide': 4, 'large': 5}.get(s, 99))

    if args.scale:
        scales = [args.scale]

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Plot individual scales
    n_scales = len(scales)
    if n_scales >= 2:
        fig, axes = plt.subplots(1, n_scales, figsize=(7 * n_scales, 5))
        if n_scales == 1:
            axes = [axes]
        for i, scale in enumerate(scales):
            plot_scale(best_runs, scale, ax=axes[i])
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'tiny_training_curves.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR}/tiny_training_curves.png")
        plt.close()
    elif n_scales == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_scale(best_runs, scales[0], ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f'tiny_curves_{scales[0]}.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR}/tiny_curves_{scales[0]}.png")
        plt.close()

    # Delta plot (only if multiple scales with both std and lif)
    scales_with_both = [s for s in scales
                        if any(k[0] == s and k[1] == 'standard' for k in best_runs)
                        and any(k[0] == s and k[1] == 'lif' for k in best_runs)]
    if scales_with_both:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_delta(best_runs, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'tiny_lif_delta.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR}/tiny_lif_delta.png")
        plt.close()

    # U-curve: final delta% vs width (paper main figure)
    if len(scales_with_both) >= 3:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_u_curve(best_runs, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'tiny_u_curve.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR}/tiny_u_curve.png")
        plt.close()

    print("Done!")


if __name__ == '__main__':
    main()
