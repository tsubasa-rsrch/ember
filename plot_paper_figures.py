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


def fig6_ei_balance():
    """v3.5 E/I Balance: individual components hurt, combination wins."""

    conditions = ['Standard', 'LIF-fixed', 'LIF-learnable', 'LIF-refrac', 'Head-persist', 'Refrac+Head']
    means = [1.4854, 1.4781, 1.4743, 1.4914, 1.4910, 1.4737]
    stds = [0.0202, 0.0154, 0.0097, 0.0437, 0.0417, 0.0070]

    # Color coding: blue=baseline, grey=components, red=inhibitory-only, green=best
    C_INHIB = '#E91E63'  # Pink for inhibitory-only
    C_BEST = '#00BCD4'   # Teal for best combo
    colors = [C_STD, C_FIXED, C_LIF, C_INHIB, C_INHIB, C_BEST]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean val_loss
    x = np.arange(len(conditions))
    bars = ax1.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    bars[5].set_edgecolor(C_BEST)
    bars[5].set_linewidth(2.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Val Loss (mean ± std, 3 seeds)')
    ax1.set_title('v3.5 Ablation: E/I Balance Discovery')

    # Add delta labels
    for i, (m, s) in enumerate(zip(means, stds)):
        delta = (m - means[0]) / means[0] * 100
        sign = '+' if delta > 0 else ''
        color = 'red' if delta > 0 else 'green'
        ax1.text(i, m + s + 0.002, f'{sign}{delta:.2f}%', ha='center', va='bottom',
                 fontsize=8, color=color, fontweight='bold')

    ax1.axhline(y=means[0], color='grey', linestyle='--', alpha=0.4)
    ax1.set_ylim(1.43, 1.56)

    # Right: Std reduction factor
    std_factors = [s / stds[0] for s in stds]
    bar_colors = ['grey' if f >= 1.0 else 'green' for f in std_factors]
    bar_colors[5] = C_BEST  # Highlight best

    ax2.bar(x, std_factors, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5, label='Baseline std')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Std / Baseline Std')
    ax2.set_title('Seed Stability: Lower = More Consistent')

    for i, f in enumerate(std_factors):
        label = f'{f:.1f}×' if f >= 1.0 else f'{1/f:.1f}× ↓'
        color = 'red' if f >= 1.0 else 'green'
        ax2.text(i, f + 0.05, label, ha='center', va='bottom', fontsize=9,
                 color=color, fontweight='bold')

    ax2.set_ylim(0, 2.5)

    # Add biological annotation
    fig.text(0.5, -0.02,
             'Inhibitory alone (Refrac, Head) → worse.  Combined with excitatory (LIF) → best.\n'
             'Mirrors biological E/I balance requirement for critical period opening (Hensch, 2005).',
             ha='center', fontsize=9, style='italic', color='#555')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_ei_balance.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


def fig7_architecture():
    """Architecture diagram: LIF gate position in Transformer and CfC."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    # --- Left: Transformer with LIF ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Transformer Ember', fontsize=14, fontweight='bold', pad=15)

    def draw_box(ax, x, y, w, h, text, color='#E3F2FD', ec='#1565C0', lw=1.5, fs=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=ec, linewidth=lw, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2), zorder=1)

    # Transformer blocks
    cx = 5  # center x
    bw = 5  # box width
    bh = 0.9

    # Input
    draw_box(ax1, cx-bw/2, 0.5, bw, bh, 'Token Embeddings + Pos Enc', '#FFF9C4', '#F9A825')
    draw_arrow(ax1, cx, 0.5+bh, cx, 2.1)

    # Layer norm
    draw_box(ax1, cx-bw/2, 2.1, bw, bh*0.7, 'Layer Norm', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax1, cx, 2.1+bh*0.7, cx, 3.3)

    # Multi-head attention
    draw_box(ax1, cx-bw/2, 3.3, bw, bh, 'Multi-Head Attention\n(Q, K → Softmax)', '#E3F2FD', '#1565C0')
    draw_arrow(ax1, cx, 3.3+bh, cx, 4.7)

    # LIF GATE (highlighted)
    draw_box(ax1, cx-bw/2, 4.7, bw, bh*1.1,
             '** LIF Gate **\nfire/smolder + norm-preserve',
             '#FFEBEE', '#C62828', lw=3, fs=10)
    draw_arrow(ax1, cx, 4.7+bh*1.1, cx, 6.3)

    # Value projection
    draw_box(ax1, cx-bw/2, 6.3, bw, bh*0.7, 'Value Projection (c_proj)', '#E3F2FD', '#1565C0')
    draw_arrow(ax1, cx, 6.3+bh*0.7, cx, 7.5)

    # Residual + FFN
    draw_box(ax1, cx-bw/2, 7.5, bw, bh*0.7, 'Residual + Layer Norm', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax1, cx, 7.5+bh*0.7, cx, 8.7)

    draw_box(ax1, cx-bw/2, 8.7, bw, bh, 'Feed-Forward Network', '#E8F5E9', '#2E7D32')
    draw_arrow(ax1, cx, 8.7+bh, cx, 10.1)

    draw_box(ax1, cx-bw/2, 10.1, bw, bh*0.7, 'Residual → Next Layer', '#F3E5F5', '#7B1FA2')

    # Annotation
    ax1.annotate('108 params\n(6L × 6H × 3)',
                 xy=(cx+bw/2, 5.2), xytext=(cx+bw/2+1.2, 5.2),
                 fontsize=8, color='#C62828', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C62828', lw=1))

    # Repeat annotation
    ax1.text(0.3, 6.0, '×6\nlayers', fontsize=11, fontweight='bold', color='#666',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eee', edgecolor='#999'))

    # --- Right: CfC with LIF ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Liquid Ember (CfC)', fontsize=14, fontweight='bold', pad=15)

    # Input
    draw_box(ax2, cx-bw/2, 0.5, bw, bh, 'Token Embeddings', '#FFF9C4', '#F9A825')
    draw_arrow(ax2, cx, 0.5+bh, cx, 2.1)

    # Layer norm
    draw_box(ax2, cx-bw/2, 2.1, bw, bh*0.7, 'Layer Norm', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax2, cx, 2.1+bh*0.7, cx, 3.3)

    # CfC cell
    draw_box(ax2, cx-bw/2, 3.3, bw, bh*1.2,
             'CfC Cell\n(Continuous-time ODE)', '#E8F5E9', '#2E7D32', fs=10)
    draw_arrow(ax2, cx, 3.3+bh*1.2, cx, 5.1)

    # LIF GATE (highlighted)
    draw_box(ax2, cx-bw/2, 5.1, bw, bh*1.1,
             '** LIF Gate **\nfire/smolder + norm-preserve',
             '#FFEBEE', '#C62828', lw=3, fs=10)
    draw_arrow(ax2, cx, 5.1+bh*1.1, cx, 6.8)

    # Residual
    draw_box(ax2, cx-bw/2, 6.8, bw, bh*0.7, 'Residual Connection', '#F3E5F5', '#7B1FA2')
    draw_arrow(ax2, cx, 6.8+bh*0.7, cx, 8.0)

    # Dropout + next
    draw_box(ax2, cx-bw/2, 8.0, bw, bh*0.7, 'Dropout → Next Layer', '#F3E5F5', '#7B1FA2')

    # Annotation
    ax2.annotate('1,536 params\n(4L × 128d × 3)',
                 xy=(cx+bw/2, 5.6), xytext=(cx+bw/2+1.2, 5.6),
                 fontsize=8, color='#C62828', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C62828', lw=1))

    ax2.text(0.3, 5.5, '×4\nlayers', fontsize=11, fontweight='bold', color='#666',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eee', edgecolor='#999'))

    # LIF gate detail box at bottom
    ax2.text(cx, 9.5,
             'LIF Gate:  fire = σ(k·(|x| − θ)),  out = x ⊙ (fire + λ·smolder),  norm-preserved',
             ha='center', fontsize=8, style='italic', color='#555',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100'))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_architecture.png')
    plt.savefig(path)
    print(f'Saved: {path}')
    plt.close()


FIGURES = {
    1: fig1_crossover,
    2: fig2_depth_hierarchy,
    3: fig3_param_efficiency,
    4: fig4_seed_summary,
    5: fig5_cfc_summary,
    6: fig6_ei_balance,
    7: fig7_architecture,
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
