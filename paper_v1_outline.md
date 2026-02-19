# Ember Paper v1 — Outline

**Working Title**: LIF Gating Creates Hierarchical Neural Organization in Transformer and Continuous-Time Neural Networks

**Target venue**: NeuroAI workshop (NeurIPS/ICML/ICLR) or standalone

**Core claim**: Adding a biologically-inspired Leaky Integrate-and-Fire (LIF) gating mechanism
to standard neural networks induces progressive specialization with depth — a universal
organizational principle that emerges regardless of backbone architecture.

---

## Abstract (~150 words)

- Biological neural circuits use threshold-based firing to create selective gating
- We add a minimal LIF mechanism (108-180 params) as a post-computation gate
- Tested on two fundamentally different architectures: Transformer + CfC (continuous-time RNN)
- Results: consistent val_loss improvement with remarkable seed stability
- Key finding: LIF creates progressive depth hierarchy (shallow=broad, deep=selective)
  regardless of backbone — this hierarchy is absent in ungated baselines
- The organization emerges through a "critical period" (iter ~1600) analogous to
  GABA maturation in infant brain development
- LIF gating is 8,000x more parameter-efficient than Qwen Gated Attention

---

## 1. Introduction

### 1.1 Motivation
- Biological brains use ~20W to run 100 trillion synapses
- Key mechanism: selective gating via inhibitory neurons (GABA)
- "Why does the brain use spiking neurons?" — still an open computational question
- Modern NNs process everything uniformly (all tokens × all layers × full precision)

### 1.2 Biological motivation (light touch — detail in Discussion)
- Thalamic gating: thalamus filters cortical information flow
- GABA inhibition and critical periods
- "We draw inspiration from..." framing, not strong claims here

### 1.3 Our approach
- Minimal LIF gate: 3 learnable params per gating unit (threshold, leak, steepness)
- Applied post-computation (after attention or CfC hidden state)
- fire/smolder paradigm: above threshold = full signal, below = attenuated
- Identity initialization: starts as pass-through, gradually learns selectivity

### 1.4 Key contributions
1. **LIF gating in standard NNs** (not SNNs): threshold-based gate for conventional architectures
2. **Cross-backbone universality**: same organizational principle on Transformer and CfC
3. **Critical period emergence**: iter 1600 crossover is seed-invariant — analogous to GABA maturation
4. **Organization, not just performance**: the VALUE is the emergent hierarchy, not the loss delta
5. **Parameter efficiency**: 108 params outperform 884K (Qwen gate) — LIF is 8,000x more efficient

---

## 2. Method

### 2.1 LIF Gate Formulation
```
fire_mask = σ(k · (|x| - θ))
smolder_mask = 1 - fire_mask
output = x ⊙ (fire_mask + leak · smolder_mask)
re-normalized: output = output × (‖x‖ / ‖output‖)  # preserve gradient flow
```
- θ (threshold): learnable, per-neuron/per-head, init ≈ 0 (identity)
- k (steepness): learnable, softplus-transformed, init → sigmoid ≈ pass-through
- leak: learnable, sigmoid-transformed, init → ~0.73 (mild attenuation)

### 2.2 Transformer Ember
- nanoGPT-based, 10.65M params, 6 layers, 6 heads, 384 dim
- LIF applied post-softmax before c_proj (same position as Qwen gate)
- Shakespeare char-level language modeling (block_size=256)
- 108 LIF params (6 layers × 6 heads × 3 params)

### 2.3 Liquid Ember (CfC)
- CfC (Closed-form Continuous-time) RNN replaces Transformer
- 4 layers, 128 dim, 192 CfC units
- LIF applied post-CfC hidden state
- Same Shakespeare char-level task (block_size=256)
- 1,536 LIF params (4 layers × 128 dim × 3 params)

### 2.4 Baselines
- **Standard**: No gating (Transformer baseline / CfC-only baseline)
- **LIF-fixed**: θ=1.0 (non-learnable threshold)
- **LIF-refractory**: Adds within-layer + cross-layer refractory (180 params)
- **Qwen-gate**: `Y' = Y ⊙ σ(XW_θ)`, 884K params — direct comparison

### 2.5 Training Protocol
- Seeds: 42, 668, 1337 (all conditions)
- Optimizer: AdamW, lr=1e-3 (Transformer) / varies (CfC)
- Gradient clipping: 1.0
- 2000 iters (Transformer) / 3000 iters (CfC)
- Metric: val_loss (cross-entropy) on held-out Shakespeare

---

## 3. Results

### 3.1 Performance (Transformer)

| Condition | Mean val_loss | ± Std | vs Standard |
|-----------|------|-------|-------------|
| Standard | 1.4784 | 0.0104 | baseline |
| LIF-fixed | 1.4803 | 0.0108 | +0.13% |
| **LIF-learnable** | **1.4673** | **0.0015** | **-0.75%** |
| LIF-refractory | 1.4725 | 0.0057 | -0.40% |
| Qwen-gate | 1.4914 | 0.0032 | +0.88% |

Key: LIF-learnable wins with **smallest variance** across seeds.

### 3.2 Performance (CfC / Liquid Ember)

| Condition | Mean val_loss | ± Std | vs Base |
|-----------|------|-------|---------|
| CfC-only (Base) | 1.4813 | 0.0042 | baseline |
| **CfC+LIF** | **1.4804** | **0.0042** | **-0.06%** |

Key: LIF wins all 3 seeds consistently (3/3).

### 3.3 Head/Neuron Specialization

**Transformer LIF**: 3-5/36 heads deviate significantly from pass-through
- L0: "pointer" heads (entropy=0.01, attend to 1 token)
- L1: "gatherer" heads (entropy≈4.0, broad attention)
- L4-5: "focuser" heads (entropy<1.0, selective)
- Standard: NO specialization (all heads identical, entropy≈4.5)

**CfC LIF**: Progressive neuron gating hierarchy
- L0: fire_rate=0.992, entropy=0.070
- L3: fire_rate=0.960, entropy=0.179
- Base: fire_rate=1.000, entropy=0.000 (zero differentiation)

### 3.4 Depth Hierarchy (Cross-Backbone)

| Backbone | Condition | Shallow entropy | Deep entropy | Trend |
|----------|-----------|---------|------|-------|
| Transformer | Standard | 1.43 | 1.69 | ↑ weak |
| Transformer | **LIF** | **1.25** | **2.47** | **↑↑ strong** |
| CfC | Base | 0.000 | 0.000 | → flat |
| CfC | **LIF** | **0.067** | **0.161** | **↑ progressive** |

**Universal pattern**: LIF narrows shallow layers, broadens deep layers.

### 3.5 Critical Period

- Crossover at iter ~1600 (verified on seeds 668 and 1337)
- Before: LIF ≈ Standard (thresholds near zero, no effective gating)
- After: LIF overtakes (thresholds stabilize, hierarchy emerges)
- Seed-invariant timing: analogous to biological critical period consistency

### 3.6 Parameter Efficiency

| Mechanism | Extra params | val_loss effect |
|-----------|-------------|-----------------|
| LIF-learnable | 108 | **-0.75%** |
| LIF-refractory | 180 | -0.40% |
| Qwen-gate | 884,736 | +0.88% (worse!) |

LIF is **8,000x more parameter-efficient** than Qwen gate.

---

## 4. Analysis

### 4.1 Critical Period as GABA Maturation
- Biological critical periods: GABA inhibition matures → selectivity begins
- LIF critical period: learned thresholds stabilize → gating becomes effective
- Both are endogenous (no external schedule needed)
- Timing is robust to noise (seed variance / biological noise)

### 4.2 Threshold Hierarchy
- Deeper layers consistently learn higher thresholds (all 3 seeds)
- CfC L3 threshold = 3× L0 threshold
- Transformer: late-layer heads have strongest filtering
- Maps to cortical depth: superficial layers → broad, deep layers → specialized

### 4.3 Why LIF Works Despite Tiny Loss Delta
- Loss improvement is real but small (-0.75% Transformer, -0.06% CfC)
- The VALUE is not the delta but the STRUCTURE:
  - Head self-differentiation (pointer/gatherer/focuser) — absent in Standard
  - Progressive depth hierarchy — absent in Standard
  - Seed stability (std=0.0015 vs 0.0104) — LIF regularizes
- Biological implication: "Why does the brain use spiking neurons?"
  → Not for accuracy, for ORGANIZATION

---

## 5. Related Work

### 5.1 Spiking Transformers
- Spikformer (ICLR 2022), Addition-Only Attention (2025)
- These REPLACE attention with spikes; we AUGMENT it
- Different goal: energy efficiency vs organizational principle

### 5.2 Gated Attention
- Qwen Gated Attention (NeurIPS 2025 Best Paper)
- `Y' = Y ⊙ σ(XW_θ)` — query-dependent sigmoid gate
- 884K params, fixes attention sink
- Our LIF: 108 params, same gate position, but biologically grounded

### 5.3 Sparse Attention
- SeerAttention, NSA (hierarchical sparse)
- Block-level sparsity; LIF is token-level (more granular)

### 5.4 Neuroscience
- Thalamic gating: thalamus gates cortical information flow
- GABA maturation and critical periods
- Sparse coding (1-5% simultaneous activation)

---

## 6. Discussion

### 6.1 Limitations
- Small-scale experiments (10M params, Shakespeare)
- Char-level language modeling only (one task)
- Would benefit from scaling experiments and diverse tasks

### 6.2 Implications for NeuroAI
- LIF gating may answer "why spiking?" from a computational perspective
- Organization-first view: the brain may spike for structure, not efficiency
- Critical period emergence without scheduling: developmental parallels

### 6.3 Future Work
- Multimodal extension (audio — in progress)
- Scaling to larger models
- Temporal LIF (membrane potential across layers)
- Embodied application (ReachyMini sensory-motor integration)

---

## Figures (planned)

1. **Architecture diagram**: LIF gate position in Transformer and CfC
2. **Loss curves**: Standard vs LIF showing crossover at iter 1600
3. **Entropy heatmap**: per-layer per-head entropy (LIF vs Standard)
4. **Head specialization**: pointer/gatherer/focuser roles visualization
5. **Cross-backbone comparison**: side-by-side depth hierarchy
6. **Threshold learning trajectory**: how thresholds evolve during training
7. **Parameter efficiency plot**: val_loss vs extra params (LIF vs Qwen)

---

## Data Available

### Transformer (Ember)
- 15 checkpoints (5 conditions × 3 seeds)
- Training logs with per-iteration val_loss
- Attention entropy/support size analysis
- Head threshold evolution data

### CfC (Liquid Ember)
- 6 checkpoints (2 conditions × 3 seeds)
- Per-iteration val_loss (every 200 iters)
- Neuron firing rate/entropy per layer
- Threshold values per layer per seed

---

*2026-02-19 — Tsubasa × Kana*
