"""
Generate figures for Ember Paper v1.

Usage:
    python plot_paper_figures.py [--fig N]

Figures:
    1. Loss curves: Standard vs LIF crossover (CfC, seed 668)
    2. Cross-backbone depth hierarchy comparison
    3. Parameter efficiency: val_loss vs extra params
    4. 3-seed summary bar chart (Transformer ablation)

2026-02-19 — Tsubasa
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Paper style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colors
C_STD = '#2196F3'   # Blue
C_LIF = '#FF5722'   # Deep Orange
C_FIXED = '#9E9E9E' # Grey
C_REFRAC = '#4CAF50' # Green
C_QWEN = '#9C27B0'  # Purple


def fig1_crossover():
    """Loss curves showing the critical period crossover at iter ~1600."""

    # CfC Seed 668 data (from DESIGN.md)
    iters_668 = [200, 1000, 1400, 1600, 2400, 2600, 2800]
    base_668 = [1.9892, 1.5984, 1.5518, 1.5223, 1.4889, 1.4809, 1.4757]
    lif_668 = [1.9916, 1.6032, 1.5534, 1.5214, 1.4868, 1.4770, 1.4747]

    # CfC Seed 1337 data (from DESIGN.md)
    iters_1337 = [800, 1600, 2400, 2800]
    base_1337 = [1.6418, 1.5348, 1.4938, 1.4826]
    lif_1337 = [1.6493, 1.5348, 1.4933, 1.4818]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    # Seed 668
    ax1.plot(iters_668, base_668, 'o-', color=C_STD, label='CfC-only (Base)', linewidth=2, markersize=5)
    ax1.plot(iters_668, lif_668, 's-', color=C_LIF, label='CfC + LIF', linewidth=2, markersize=5)
    ax1.axvline(x=1600, color='grey', linestyle='--', alpha=0.5, label='Critical period (~1600)')
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Seed 668')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 3000)

    # Seed 1337
    ax2.plot(iters_1337, base_1337, 'o-', color=C_STD, label='CfC-only (Base)', linewidth=2, markersize=5)
    ax2.plot(iters_1337, lif_1337, 's-', color=C_LIF, label='CfC + LIF', linewidth=2, markersize=5)
    ax2.axvline(x=1600, color='grey', linestyle='--', alpha=0.5, label='Critical period (~1600)')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Seed 1337')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 3000)

    fig.suptitle('LIF Crossover: Base leads early, LIF overtakes at iter ~1600', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_crossover.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


def fig2_depth_hierarchy():
    """Cross-backbone depth hierarchy comparison."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Transformer (6 layers, attention entropy)
    layers_t = np.arange(6)
    # Approximate from DESIGN.md entropy data
    std_entropy_t = [1.43, 1.5, 1.55, 1.58, 1.62, 1.69]
    lif_entropy_t = [1.25, 1.4, 1.6, 1.8, 2.1, 2.47]

    ax1.plot(layers_t, std_entropy_t, 'o-', color=C_STD, label='Standard', linewidth=2, markersize=6)
    ax1.plot(layers_t, lif_entropy_t, 's-', color=C_LIF, label='LIF', linewidth=2, markersize=6)
    ax1.fill_between(layers_t, std_entropy_t, lif_entropy_t, alpha=0.1, color=C_LIF)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Attention Entropy')
    ax1.set_title('Transformer (Attention Heads)')
    ax1.legend()
    ax1.set_xticks(layers_t)
    ax1.set_xticklabels([f'L{i}' for i in layers_t])

    # CfC (4 layers, neuron firing entropy)
    layers_c = np.arange(4)
    base_entropy_c = [0.0, 0.0, 0.0, 0.0]
    lif_entropy_c = [0.067, 0.133, 0.144, 0.161]  # Cross-seed mean

    ax2.plot(layers_c, base_entropy_c, 'o-', color=C_STD, label='CfC-only (Base)', linewidth=2, markersize=6)
    ax2.plot(layers_c, lif_entropy_c, 's-', color=C_LIF, label='CfC + LIF', linewidth=2, markersize=6)
    ax2.fill_between(layers_c, base_entropy_c, lif_entropy_c, alpha=0.1, color=C_LIF)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Neuron Firing Entropy')
    ax2.set_title('CfC / Liquid Ember (Neurons)')
    ax2.legend()
    ax2.set_xticks(layers_c)
    ax2.set_xticklabels([f'L{i}' for i in layers_c])

    fig.suptitle('Depth Hierarchy: LIF creates progressive specialization in both backbones', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_depth_hierarchy.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


def fig3_param_efficiency():
    """Parameter efficiency: val_loss improvement vs extra params."""

    mechanisms = ['LIF-learnable', 'LIF-refractory', 'LIF-fixed', 'Qwen-gate']
    extra_params = [108, 180, 108, 884736]
    val_improvement = [-0.75, -0.40, 0.13, 0.88]  # positive = worse

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = [C_LIF, C_REFRAC, C_FIXED, C_QWEN]
    for i, (mech, params, imp) in enumerate(zip(mechanisms, extra_params, val_improvement)):
        marker = 'o' if imp < 0 else 'x'
        size = 120 if imp < 0 else 80
        ax.scatter(params, imp, color=colors[i], s=size, marker=marker, zorder=5,
                   label=f'{mech} ({params:,} params)')

    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5, label='Standard baseline')
    ax.set_xscale('log')
    ax.set_xlabel('Extra Parameters (log scale)')
    ax.set_ylabel('Val Loss Change vs Standard (%)')
    ax.set_title('Parameter Efficiency: LIF achieves more with 8,000x fewer parameters')
    ax.legend(loc='upper left')

    # Annotate
    ax.annotate('Better', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, color='green', alpha=0.6)
    ax.annotate('Worse', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, color='red', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_param_efficiency.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


def fig4_seed_summary():
    """3-seed summary bar chart for Transformer ablation."""

    conditions = ['Standard', 'LIF-fixed', 'LIF-learnable', 'LIF-refractory', 'Qwen-gate']
    means = [1.4784, 1.4803, 1.4673, 1.4725, 1.4914]
    stds = [0.0104, 0.0108, 0.0015, 0.0057, 0.0032]
    colors = [C_STD, C_FIXED, C_LIF, C_REFRAC, C_QWEN]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Highlight winner
    bars[2].set_edgecolor(C_LIF)
    bars[2].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylabel('Val Loss (mean ± std, 3 seeds)')
    ax.set_title('Transformer Ember: 3-Seed Ablation (2000 iters, Shakespeare)')

    # Add val_loss labels
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f'{m:.4f}\n±{s:.4f}'
        ax.text(i, m + s + 0.001, label, ha='center', va='bottom', fontsize=8)

    ax.set_ylim(1.455, 1.510)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_seed_summary.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


def fig5_cfc_summary():
    """CfC 3-seed summary (Base vs LIF)."""

    conditions = ['CfC-only (Base)', 'CfC + LIF']
    means = [1.4813, 1.4804]
    stds = [0.0042, 0.0042]
    colors = [C_STD, C_LIF]

    # Per-seed data
    seeds = [42, 668, 1337]
    base_vals = [1.4856, 1.4757, 1.4826]
    lif_vals = [1.4848, 1.4747, 1.4818]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    x = np.arange(2)
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5, width=0.5)
    bars[1].set_edgecolor(C_LIF)
    bars[1].set_linewidth(2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylabel('Val Loss (mean ± std)')
    ax1.set_title('CfC / Liquid Ember: 3-Seed Mean')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 0.0005, f'{m:.4f}±{s:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylim(1.470, 1.495)

    # Per-seed comparison
    x_seeds = np.arange(3)
    w = 0.3
    ax2.bar(x_seeds - w/2, base_vals, w, color=C_STD, label='CfC-only', alpha=0.85)
    ax2.bar(x_seeds + w/2, lif_vals, w, color=C_LIF, label='CfC + LIF', alpha=0.85)
    ax2.set_xticks(x_seeds)
    ax2.set_xticklabels([f'Seed {s}' for s in seeds])
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Per-Seed: LIF wins all 3 seeds')
    ax2.legend()
    ax2.set_ylim(1.470, 1.490)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_cfc_summary.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


FIGURES = {
    1: fig1_crossover,
    2: fig2_depth_hierarchy,
    3: fig3_param_efficiency,
    4: fig4_seed_summary,
    5: fig5_cfc_summary,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=int, default=0, help='Figure number (0=all)')
    args = parser.parse_args()

    if args.fig == 0:
        for num, fn in FIGURES.items():
            print(f'\n--- Figure {num} ---')
            fn()
    elif args.fig in FIGURES:
        FIGURES[args.fig]()
    else:
        print(f'Unknown figure {args.fig}. Available: {list(FIGURES.keys())}')

    print(f'\nAll figures saved to: {OUT_DIR}')
