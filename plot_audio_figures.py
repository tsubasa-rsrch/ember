"""
Generate figures for Audio Liquid Ember (Paper 2).

Usage:
    python plot_audio_figures.py

Figures:
    1. 3-seed accuracy bar chart (Base vs LIF)
    2. Learning curves per seed (showing crossover)
    3. Cross-modality depth hierarchy comparison (text vs audio)

Requires: all 6 audio training runs completed.

2026-02-19 — Tsubasa
"""

import os
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'out')
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
C_BASE = '#2196F3'   # Blue
C_LIF = '#FF5722'    # Deep Orange


def parse_log(path):
    """Parse training log to extract epoch-by-epoch metrics."""
    epochs = []
    with open(path) as f:
        for line in f:
            # Match: "     5       0.8712     0.7651     0.8515     0.7781    617.5s"
            m = re.match(
                r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)s',
                line
            )
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'train_acc': float(m.group(3)),
                    'val_loss': float(m.group(4)),
                    'val_acc': float(m.group(5)),
                    'time': float(m.group(6)),
                })
    return epochs


def parse_best(path):
    """Extract best val_acc and test_acc from log."""
    best_val = None
    test_acc = None
    with open(path, errors='replace') as f:
        for line in f:
            # Format: "best_val_acc=0.8827 test_acc=0.8656"
            if 'best_val_acc=' in line:
                import re as _re
                m = _re.search(r'best_val_acc=([\d.]+)', line)
                if m:
                    best_val = float(m.group(1))
                m = _re.search(r'test_acc=([\d.]+)', line)
                if m:
                    test_acc = float(m.group(1))
            # Fallback: "Best val accuracy: 0.8827"
            if 'Best val accuracy' in line:
                best_val = float(line.split(':')[-1].strip())
            if 'Test accuracy' in line:
                test_acc = float(line.split(':')[-1].strip())
    return best_val, test_acc


def fig1_accuracy_bar():
    """3-seed accuracy comparison: Base vs LIF."""
    seeds = [42, 668, 1337]
    base_accs = []
    lif_accs = []
    base_tests = []
    lif_tests = []

    for seed in seeds:
        bp = os.path.join(LOG_DIR, f'audio_base_s{seed}.log')
        lp = os.path.join(LOG_DIR, f'audio_lif_s{seed}.log')
        if os.path.exists(bp):
            bv, bt = parse_best(bp)
            if bv is not None:
                base_accs.append(bv)
                base_tests.append(bt)
        if os.path.exists(lp):
            lv, lt = parse_best(lp)
            if lv is not None:
                lif_accs.append(lv)
                lif_tests.append(lt)

    if not base_accs or not lif_accs:
        print("Not enough data for fig1")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    x = np.arange(len(seeds))
    w = 0.35

    bars1 = ax.bar(x - w/2, [a*100 for a in base_accs], w, label='CfC-only (Base)',
                   color=C_BASE, alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + w/2, [a*100 for a in lif_accs], w, label='CfC + LIF',
                   color=C_LIF, alpha=0.8, edgecolor='white')

    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_xlabel('Random Seed')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds[:len(base_accs)]])
    ax.legend()
    ax.set_ylim(80, 100)

    # Mean lines
    base_mean = np.mean(base_accs) * 100
    lif_mean = np.mean(lif_accs) * 100
    ax.axhline(base_mean, color=C_BASE, linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(lif_mean, color=C_LIF, linestyle='--', alpha=0.5, linewidth=1)

    # Annotate means
    ax.text(len(seeds)-0.5, base_mean + 0.3, f'Base mean: {base_mean:.1f}%',
            color=C_BASE, fontsize=9, ha='right')
    ax.text(len(seeds)-0.5, lif_mean + 0.3, f'LIF mean: {lif_mean:.1f}%',
            color=C_LIF, fontsize=9, ha='right')

    ax.set_title('Audio Classification: Speech Commands v2 (35 words)')
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'audio_fig1_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    # Print summary
    print(f"\n  Base: {base_mean:.2f}% ± {np.std(base_accs)*100:.2f}%")
    print(f"  LIF:  {lif_mean:.2f}% ± {np.std(lif_accs)*100:.2f}%")
    print(f"  Delta: {lif_mean - base_mean:+.2f}%")
    if base_tests and lif_tests:
        print(f"  Test Base: {np.mean(base_tests)*100:.2f}% ± {np.std(base_tests)*100:.2f}%")
        print(f"  Test LIF:  {np.mean(lif_tests)*100:.2f}% ± {np.std(lif_tests)*100:.2f}%")


def fig2_learning_curves():
    """Per-seed learning curves showing potential crossover."""
    seeds = [42, 668, 1337]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, seed in enumerate(seeds):
        ax = axes[i]
        bp = os.path.join(LOG_DIR, f'audio_base_s{seed}.log')
        lp = os.path.join(LOG_DIR, f'audio_lif_s{seed}.log')

        if os.path.exists(bp):
            base_data = parse_log(bp)
            if base_data:
                epochs = [d['epoch'] for d in base_data]
                vals = [d['val_acc'] * 100 for d in base_data]
                ax.plot(epochs, vals, '-o', color=C_BASE, markersize=3,
                        label='CfC-only', linewidth=1.5)

        if os.path.exists(lp):
            lif_data = parse_log(lp)
            if lif_data:
                epochs = [d['epoch'] for d in lif_data]
                vals = [d['val_acc'] * 100 for d in lif_data]
                ax.plot(epochs, vals, '-s', color=C_LIF, markersize=3,
                        label='CfC+LIF', linewidth=1.5)

        ax.set_title(f'Seed {seed}')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Validation Accuracy (%)')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Audio Liquid Ember: Learning Curves per Seed', y=1.02)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'audio_fig2_learning_curves.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig3_cross_modality_hierarchy():
    """Cross-modality depth hierarchy: text (Transformer) + audio (CfC).

    Uses hardcoded text results from paper + audio results from analyze_audio.py.
    This figure shows entropy gradient is backbone AND modality independent.
    """
    # Text results (from paper_v1_draft.md Table in Section 3.3)
    text_std_entropy = [1.43, 1.50, 1.55, 1.58, 1.62, 1.69]  # L0-L5
    text_lif_entropy = [1.25, 1.40, 1.60, 1.80, 2.10, 2.47]  # L0-L5

    # CfC text results (from DESIGN.md cross-seed mean)
    cfc_std_entropy = [0.000, 0.000, 0.000, 0.000]  # L0-L3
    cfc_lif_entropy = [0.067, 0.133, 0.144, 0.161]  # L0-L3

    # Audio CfC results (from analyze_audio.py cross-seed mean, 2026-02-19)
    audio_std_entropy = [0.000, 0.000, 0.000, 0.000]  # L0-L3 (Base = no gating)
    audio_lif_entropy = [0.112, 0.147, 0.092, 0.055]  # L0-L3 (LIF = declining trend)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Transformer (text)
    ax = axes[0]
    layers = range(6)
    ax.plot(layers, text_std_entropy, '-o', color=C_BASE, label='Standard', linewidth=2)
    ax.plot(layers, text_lif_entropy, '-s', color=C_LIF, label='LIF', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Transformer (Text)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{i}' for i in layers])

    # Panel 2: CfC (text)
    ax = axes[1]
    layers = range(4)
    ax.plot(layers, cfc_std_entropy, '-o', color=C_BASE, label='CfC-only', linewidth=2)
    ax.plot(layers, cfc_lif_entropy, '-s', color=C_LIF, label='CfC+LIF', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron Firing Entropy')
    ax.set_title('CfC (Text)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{i}' for i in layers])

    # Panel 3: CfC (Audio)
    ax = axes[2]
    if audio_std_entropy and audio_lif_entropy:
        layers = range(4)
        ax.plot(layers, audio_std_entropy, '-o', color=C_BASE, label='CfC-only', linewidth=2)
        ax.plot(layers, audio_lif_entropy, '-s', color=C_LIF, label='CfC+LIF', linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Neuron Firing Entropy')
        ax.set_title('CfC (Audio)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)
        ax.set_xticklabels([f'L{i}' for i in layers])
    else:
        ax.text(0.5, 0.5, 'Audio analysis\npending...',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='gray')
        ax.set_title('CfC (Audio)')

    plt.suptitle('Cross-Backbone, Cross-Modality Depth Hierarchy', y=1.02, fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'audio_fig3_cross_modality.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("AUDIO LIQUID EMBER — FIGURE GENERATION")
    print("=" * 60)

    # Check what data we have
    seeds = [42, 668, 1337]
    available = []
    for seed in seeds:
        for cond in ['base', 'lif']:
            path = os.path.join(LOG_DIR, f'audio_{cond}_s{seed}.log')
            if os.path.exists(path):
                bv, bt = parse_best(path)
                if bv is not None:
                    available.append(f"{cond}_s{seed}: val={bv:.4f}, test={bt:.4f}")
    print(f"\nAvailable results ({len(available)}/6):")
    for a in available:
        print(f"  {a}")
    print()

    fig1_accuracy_bar()
    fig2_learning_curves()
    fig3_cross_modality_hierarchy()

    print("\nDone!")
