# Paper 1+2 Merge Plan → NeurIPS NeuroAI 2026

## Target
- **8 pages main body** (NeurIPS NeuroAI format)
- Single unified paper with cross-modal evidence
- Title: "LIF Gating as a Cross-Modal Neural Organizer"

## Source Papers
- Paper 1 (`main.tex`): Text classification, Transformer + CfC, 586 lines, ~16pp
- Paper 2 (`paper2_audio.tex`): Audio classification, 491 lines, ~12pp
- References merged: `references.bib` (16 entries)

## Proposed Structure

### 1. Introduction (~1 page)
- Hook: biological neurons gate via LIF dynamics
- Gap: modern attention lacks this gating
- Contribution: LIF gating as universal regularizer across modalities/architectures
- Preview: text + audio experiments, width dependence, E/I balance

### 2. Method: LIF Gating (~1 page)
- Core formulation (Paper 1 §2, unified with Paper 2's smolder_mask)
- Text backbone: Transformer (xs to full scale) + CfC
- Audio backbone: Conv+Transformer for Speech Commands v2
- Training: identity init, 3k iterations per condition, 8 seeds

### 3. Results (~3 pages)
- **3.1 Text**: Full-scale ablation results table (Paper 1 Table 1)
  - Delta -0.10% to -0.12%, std reduction 2.8-9.2x
  - Statistical: p=0.024 (xs), p=0.235 (mid)
- **3.2 Audio**: Accuracy + stability results (Paper 2 Table 1)
  - Val std 9.2x reduction, Test std 2.8x
  - Seed 1337 smoking gun (catastrophic failure prevention)
- **3.3 Cross-Modal Comparison** (Paper 2 §4)
  - Universal regularizer hypothesis
  - Modality-dependent organization differences
  - Fire rate regime comparison
- **3.4 Width Dependence** (Paper 1 §3.7)
  - U-curve: xs(-0.10%) → medium(+0.35%) → wide(-0.12%)
  - Critical width threshold ~256-320d
- **3.5 Cross-Architecture** (Paper 1 §3.9)
  - Transformer: LIF effective (std reduction + quality)
  - CfC: LIF null effect → architecture-specific, not universal
  - Neuroanatomical interpretation: thalamus bypasses cerebellum

### 4. Analysis (~1 page)
- **4.1 Critical Period**: GABA maturation analogy, endogenous timing at ~60% training
- **4.2 Per-Head Specialization**: ~25% gatekeeper heads, consistent across seeds
- **4.3 Spontaneous E/I Balance**: v3.5 refractory state creates inhibition without explicit design

### 5. Discussion (~1.5 pages)
- **5.1 Why Small Delta, Large Organizational Impact?**
  - Loss landscape smoothing (Hochreiter & Schmidhuber, Keskar et al.)
  - Churchland variability quenching analogy
- **5.2 Neuroanatomical Correspondence**
  - Purkinje cells ↔ gatekeeper heads
  - Thalamic relay ↔ CfC null result
- **5.3 Scaling Probe: Qwen3-14B**
  - 4,800 params on 14B model, zero degradation
  - Identity init > trained for personality tasks
- **5.4 Limitations**
  - N=3-8 seeds, single dataset per modality
  - CfC null needs wider investigation
  - Biological analogies are structural, not mechanistic
- **5.5 Future Work**
  - AdaptiveLIF (entropy-conditioned dynamic thresholds)
  - Vision modality
  - Larger-scale validation

### 6. Conclusion (~0.5 pages)
- LIF gating = universal regularizer (modality-independent, architecture-dependent)
- Convergent functional architecture between artificial and biological systems
- 4,800 params achieve what requires millions with LoRA

## Key Figures (max 4-5 for 8 pages)
1. LIF formulation diagram + gating visualization
2. Cross-modal results comparison table/figure
3. Width dependence U-curve
4. Per-head threshold differentiation heatmap
5. Seed 1337 smoking gun (audio val accuracy across epochs)

## References to use from merged bib
All 16 entries should be cited. Key additions from Paper 2:
- churchland2010: variability quenching
- douglas2004: neocortical circuits
- hochreiter1997: flat minima theory
- keskar2017: sharp vs flat minima
- warden2018: Speech Commands dataset

## LIF Formulation Discrepancy
Paper 1: basic σ(s·(a−θ)) gating
Paper 2: adds smolder_mask (prevent dead neurons) + LayerNorm preservation
→ Use Paper 2's formulation as canonical (more mature)

## Implementation Notes
- Start from Paper 1's main.tex as base (better formatting)
- Import audio results and cross-modal analysis from Paper 2
- Compress heavily: combine similar discussion points
- Remove redundant methodology details between papers
- Single unified results section with subsections per modality
