#!/usr/bin/env python3
"""Figure 10: Cross-Architecture Heatmap + Neuroanatomy Diagram
LIF effect on Transformer vs CfC at XS and Wide scales.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Style conventions (from existing plots)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colors
C_STD = '#2196F3'
C_LIF = '#FF5722'
C_BEST = '#00BCD4'

# === DATA ===
# LIF effect (%) for each cell
# Rows: Transformer, CfC
# Cols: XS, Wide, Full (Transformer only)
data_2x2 = np.array([
    [-0.10, -0.12],   # Transformer: XS, Wide
    [+0.10, +0.01],   # CfC: XS, Wide
])

labels_2x2 = [
    ['-0.10%', '-0.12%'],
    ['+0.10%', '+0.01%'],
]

sublabels_2x2 = [
    ['3/3 wins', '2/3 wins'],
    ['1/3 wins', '1/3 wins'],
]

row_labels = ['Transformer\n(parallel attention)', 'CfC\n(sequential ODE)']
col_labels = ['XS\n(0.4-0.6M)', 'Wide\n(8.9-10.7M)']

# === FIGURE ===
fig = plt.figure(figsize=(14, 5.5))

# Left panel: 2×2 heatmap
ax1 = fig.add_axes([0.04, 0.12, 0.42, 0.78])

# Custom colormap: green for negative (LIF helps), red for positive (LIF hurts), white at 0
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

colors_neg = ['#1B5E20', '#4CAF50', '#C8E6C9', '#FFFFFF']  # dark green -> white
colors_pos = ['#FFFFFF', '#FFCDD2', '#F44336', '#B71C1C']  # white -> dark red
cmap_neg = LinearSegmentedColormap.from_list('neg', colors_neg, N=128)
cmap_pos = LinearSegmentedColormap.from_list('pos', colors_pos, N=128)

# Merge into a diverging colormap
colors_all = []
for i in range(128):
    colors_all.append(cmap_neg(i))
for i in range(128):
    colors_all.append(cmap_pos(i))
cmap = LinearSegmentedColormap.from_list('lif_effect', colors_all, N=256)

norm = TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15)

im = ax1.imshow(data_2x2, cmap=cmap, norm=norm, aspect='auto')

# Add text annotations
for i in range(2):
    for j in range(2):
        val = data_2x2[i, j]
        color = 'white' if abs(val) > 0.08 else 'black'
        weight = 'bold'
        ax1.text(j, i - 0.08, labels_2x2[i][j], ha='center', va='center',
                fontsize=16, fontweight=weight, color=color)
        ax1.text(j, i + 0.22, sublabels_2x2[i][j], ha='center', va='center',
                fontsize=9, color=color, style='italic')

ax1.set_xticks([0, 1])
ax1.set_xticklabels(col_labels, fontsize=11)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(row_labels, fontsize=11)
ax1.set_title('LIF Effect by Architecture × Scale', fontsize=14, fontweight='bold', pad=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label('LIF effect (%)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Add "LIF helps" / "LIF neutral" annotations
ax1.annotate('LIF helps ↓', xy=(0.5, -0.55), fontsize=9, ha='center',
            color='#1B5E20', fontweight='bold', xycoords='data')
ax1.annotate('LIF neutral', xy=(0.5, 1.55), fontsize=9, ha='center',
            color='#B71C1C', fontweight='bold', xycoords='data')

# Transformer Full-scale reference
ax1.annotate('Full (768d): -0.75%', xy=(1.35, -0.15), fontsize=8,
            ha='center', color='#1B5E20', style='italic', xycoords='data')

# Right panel: Neuroanatomy diagram
ax2 = fig.add_axes([0.54, 0.08, 0.44, 0.84])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Neuroanatomical Correspondence', fontsize=14, fontweight='bold', pad=8)

# --- Brain pathway 1: Thalamus → Cortex (= LIF → Transformer) ---
# Sensory input
rect_input1 = patches.FancyBboxPatch((0.3, 5.8), 2.0, 1.0,
    boxstyle="round,pad=0.1", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
ax2.add_patch(rect_input1)
ax2.text(1.3, 6.3, 'Sensory\nInput', ha='center', va='center', fontsize=9, fontweight='bold')

# Thalamus (= LIF gate)
rect_thal = patches.FancyBboxPatch((3.2, 5.8), 2.2, 1.0,
    boxstyle="round,pad=0.1", facecolor='#FFCCBC', edgecolor='#E64A19', linewidth=2.0)
ax2.add_patch(rect_thal)
ax2.text(4.3, 6.3, 'Thalamus\n≈ LIF Gate', ha='center', va='center', fontsize=9, fontweight='bold',
        color='#BF360C')

# Cortex (= Transformer)
rect_cortex = patches.FancyBboxPatch((6.4, 5.8), 2.8, 1.0,
    boxstyle="round,pad=0.1", facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2.0)
ax2.add_patch(rect_cortex)
ax2.text(7.8, 6.3, 'Cerebral Cortex\n≈ Transformer', ha='center', va='center', fontsize=9,
        fontweight='bold', color='#0D47A1')

# Arrows for pathway 1
ax2.annotate('', xy=(3.2, 6.3), xytext=(2.3, 6.3),
            arrowprops=dict(arrowstyle='->', color='#424242', lw=2))
ax2.annotate('', xy=(6.4, 6.3), xytext=(5.4, 6.3),
            arrowprops=dict(arrowstyle='->', color='#E64A19', lw=2.5))

# "filters" label on arrow
ax2.text(5.9, 6.65, 'filters', fontsize=8, ha='center', color='#E64A19',
        fontweight='bold', style='italic')

# Effect label
ax2.text(7.8, 5.35, '→ LIF helps: -0.10% to -0.75%', fontsize=9, ha='center',
        color='#1B5E20', fontweight='bold')

# --- Brain pathway 2: Brainstem → Cerebellum (= Input → CfC, no thalamic relay) ---
# Sensory input 2
rect_input2 = patches.FancyBboxPatch((0.3, 2.2), 2.0, 1.0,
    boxstyle="round,pad=0.1", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
ax2.add_patch(rect_input2)
ax2.text(1.3, 2.7, 'Motor/\nPredictive', ha='center', va='center', fontsize=9, fontweight='bold')

# Bypass box (dashed, no thalamus)
rect_bypass = patches.FancyBboxPatch((3.2, 2.2), 2.2, 1.0,
    boxstyle="round,pad=0.1", facecolor='#F5F5F5', edgecolor='#9E9E9E',
    linewidth=1.5, linestyle='--')
ax2.add_patch(rect_bypass)
ax2.text(4.3, 2.7, 'No Thalamic\nRelay', ha='center', va='center', fontsize=9,
        color='#757575', style='italic')

# Cerebellum (= CfC)
rect_cereb = patches.FancyBboxPatch((6.4, 2.2), 2.8, 1.0,
    boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2.0)
ax2.add_patch(rect_cereb)
ax2.text(7.8, 2.7, 'Cerebellum\n≈ CfC (ODE)', ha='center', va='center', fontsize=9,
        fontweight='bold', color='#1B5E20')

# Arrows for pathway 2 (direct bypass)
ax2.annotate('', xy=(6.4, 2.7), xytext=(2.3, 2.7),
            arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=2, linestyle='--'))

# "bypasses" label
ax2.text(4.3, 3.55, 'bypasses', fontsize=8, ha='center', color='#757575',
        fontweight='bold', style='italic')

# Effect label
ax2.text(7.8, 1.75, '→ LIF has no effect: +0.01%', fontsize=9, ha='center',
        color='#B71C1C', fontweight='bold')

# --- Key insight box ---
rect_key = patches.FancyBboxPatch((0.5, 0.1), 9.0, 1.2,
    boxstyle="round,pad=0.15", facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
ax2.add_patch(rect_key)
ax2.text(5.0, 0.7, 'Convergent Evolution: Efficient information processing under\n'
        'threshold-based gating converges on the same architecture\n'
        'in both biological and artificial neural systems.',
        ha='center', va='center', fontsize=9, style='italic', color='#BF360C')

# --- Processing mode labels ---
ax2.text(9.6, 6.3, 'parallel', fontsize=8, ha='center', va='center',
        color='#0D47A1', fontweight='bold', rotation=0)
ax2.text(9.6, 2.7, 'sequential', fontsize=8, ha='center', va='center',
        color='#1B5E20', fontweight='bold', rotation=0)

# Save
outpath = '/Users/tsubasa/Documents/TsubasaWorkspace/ember/figures/fig10_cross_architecture.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.1)
print(f'Saved: {outpath}')
plt.close()
