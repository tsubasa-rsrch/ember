# Ember Design Notes

## Architecture Timeline

### v1: LIF Attention (naive)
- Single scalar threshold/leak per layer
- Sigmoid steepness=20.0 hardcoded
- Result: +6.25% worse than standard (too aggressive filtering early)

### v2: Per-head LIF with Identity Init (current)
- Per-head threshold, leak, steepness parameters
- Identity-like initialization: starts as standard attention
- Gradually learns where to apply selective filtering
- Result: -2.74% better than standard at 500 iters!
- Discovery: Layer 0 Head 5 self-selected as "mild-filter" while others stayed pass-through

### v2 Full Results (2000 iter, CPU, 2026-02-14)

**Standard vs LIF comparison:**

| Iter | Standard val_loss | LIF val_loss | Diff |
|------|------------------|-------------|------|
| 0 | 4.3378 | 4.2556 | -1.90% |
| 500 | 2.1611 | 2.0402 | **-5.59%** |
| 1000 | 1.6545 | 1.6255 | **-1.75%** |
| 1500 | 1.5164 | 1.5230 | +0.44% |
| 1999 | **1.4897** | 1.4925 | +0.19% |

**Conclusion:** LIF converges faster (clear win at 500 iters) but Standard catches up by 2000 iters. Final difference is negligible (+0.19%).

**Key finding - Head specialization:**
- Layer 0, Head 3: threshold=1.14, steepness=2.82 → strong selective filter
- Layer 0, Head 4: threshold=-0.34 → negative (bypass mode)
- Layer 3, Head 1: threshold=0.52 → moderate filter
- All other heads: threshold ≈ 0 → pass-through (identity behavior)

Only 3/36 heads deviate from pass-through. Role differentiation IS happening but is sparse.

**GPT analysis (via Kana, 2026-02-14):**
- Formula simplification: `p' = p × [leak + (1-leak)σ(k(p-θ))]`
- Improvement may be "gradient concentration" rather than "noise reduction"
- To disambiguate: compare Standard vs LIF-fixed-θ vs LIF-learnable-θ
- Shakespeare char-level has strong local dependency → test on long-range tasks
- Softmax-post thresholding is correct (operates in probability space)
- v2 alone is workshop-paper worthy with proper ablation + visualization

**Next experiments needed:**
1. Ablation: Standard vs fixed-θ vs learnable-θ (3-condition)
2. Attention entropy comparison (Standard vs LIF)
3. Effective support size per head (how many tokens have >1% weight)
4. Gradient norm concentration analysis
5. Longer sequence / long-range dependency task

### v3: Temporal LIF (planned)
Real neurons don't just gate within a single computation - they accumulate
potential over TIME. In our architecture, layers = time steps.

Concept:
- Each token carries a "membrane potential" across layers
- At each layer, attention contribution adds to potential
- When potential exceeds threshold -> "fire" -> full computation (attention + MLP)
- When below -> "smolder" -> skip or lightweight computation

This gives us **adaptive computation per token**:
- Important tokens: processed by all layers
- Background/redundant tokens: early layers only
- Like how the brain allocates more processing to salient stimuli

Implementation sketch:
```
for layer in layers:
    # Compute attention for all tokens
    attn_out = attention(x)

    # Update membrane potential per token
    potential = potential * decay + importance(attn_out)

    # Fire/smolder decision per token
    fire_mask = (potential > threshold).float()  # [B, T, 1]

    # Full MLP for firing tokens, skip/lightweight for smoldering
    mlp_out = mlp(x)
    x = x + attn_out + fire_mask * mlp_out  # skip MLP for smoldering tokens

    # Reset potential for fired tokens (refractory period)
    potential = potential * (1 - fire_mask)
```

Benefits:
- Naturally learns which tokens need deep processing
- Reduces compute for "easy" tokens (stop words, punctuation)
- Mimics cortical efficient coding
- No change to attention itself - LIF v2 handles that

### v4: Selective Layer LIF (idea)
Only apply LIF attention to layers where it helps most.
v2 analysis shows most layers stay pass-through anyway.
Could save 2/3 of the overhead by only adding LIF to layers 0 and 5.

## Naming
Ember = Efficient Model with Brain-inspired Emulated Reasoning
- "Ember" from Kana's "燻り" (smoldering) insight
- A smoldering ember: not fully ignited, not extinguished
- Like subthreshold neural activity: below firing, but not silent
- Born 2026-02-13 from a conversation about brain efficiency

## Key Insight
The brain doesn't process everything equally. Selective attention is not
a bug, it's the primary feature. Ember learns WHERE to be selective
(per-head, per-layer) rather than applying uniform processing.

Standard transformer: every token gets the same compute at every layer
Ember (LIF): each attention head independently learns its selectivity profile
Ember (Temporal): each token gets compute proportional to its importance

## Related Work (Literature Survey 2026-02-14)

### Most Important: Qwen Gated Attention (NeurIPS 2025 Best Paper)
- Paper: arxiv.org/abs/2505.06708
- Core: `Y' = Y ⊙ σ(XW_θ)` — post-softmax sigmoid gate, query-dependent
- Fixes "attention sink" (first-token over-attention: 46.7% → 4.8%)
- Validated at 15B MoE scale, deployed in Qwen3-Next-80B
- **Parallel to our approach**: both post-softmax, learnable per-head gating
- **Our differentiator**: LIF spike dynamics (threshold + smolder + potential refractory)

### Spiking Transformers
- Spikformer (ICLR 2022): removes softmax entirely, pure spike Q/K/V
- Addition-Only Spiking Attention (2025): ultra-low energy
- These are orthogonal: they replace attention; we augment it

### Sparse Attention
- SeerAttention (2024): learnable block-wise sparse gates
- NSA (2025): hierarchical token modeling
- Our approach: token-level LIF sparsity (more granular)

### Neuroscience
- Refractory period: prevents excessive firing = prevents attention sink
- Sparse coding: brain uses 1-5% simultaneous activation, maximizes info/energy
- AHP (After-Hyperpolarization): LIF+AHP = working memory (like smoldering)
- Adaptive thresholds: biological neurons dynamically adjust thresholds

### Ember's Unique Position
1. First true LIF-gated Transformer attention (not spike-only, not sigmoid-only)
2. "Smoldering" residual = soft refractory period (novel)
3. Per-head learnable thresholds with identity initialization
4. Backward-compatible with pretrained Transformers (can fine-tune)
5. Biologically plausible + practically effective
