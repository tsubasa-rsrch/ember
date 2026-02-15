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

### v2 Full Results (2000 iter, MPS, 2026-02-14, no seed control)

**Standard vs LIF comparison (first run, MPS device):**

| Iter | Standard val_loss | LIF val_loss | Diff |
|------|------------------|-------------|------|
| 0 | 4.3378 | 4.2556 | -1.90% |
| 500 | 2.1611 | 2.0402 | **-5.59%** |
| 1000 | 1.6545 | 1.6255 | **-1.75%** |
| 1500 | 1.5164 | 1.5230 | +0.44% |
| 1999 | **1.4897** | 1.4925 | +0.19% |

**Conclusion:** LIF converges faster (clear win at 500 iters) but Standard catches up by 2000 iters. Final difference is negligible (+0.19%).

### v2.5 Ablation Results (2000 iter, M4 CPU, 2026-02-14, no seed control)

**4-condition ablation (Standard vs LIF-fixed vs LIF-learnable vs LIF-refractory):**

| Iter | Standard | LIF-fixed | LIF-learnable | LIF-refrac |
|------|----------|-----------|---------------|------------|
| 0 | 4.2365 | 4.1857 (-1.20%) | 4.2578 (+0.50%) | 4.2826 (+1.09%) |
| 500 | 2.0630 | 2.0403 (-1.10%) | 2.0346 (-1.38%) | **2.0279 (-1.70%)** |
| 1000 | 1.6229 | **1.6076 (-0.94%)** | 1.6684 (+2.80%) | 1.6204 (-0.15%) |
| 1500 | 1.5106 | **1.5079 (-0.18%)** | 1.5698 (+3.92%) | 1.5114 (+0.05%) |
| 1999 | **1.4683** | 1.4816 (+0.90%) | 1.5268 (+3.98%) | 1.4862 (+1.22%) |

**Key findings:**
1. **Standard wins at 2000 iters**: But all LIF variants converge faster early
2. **LIF-refractory best at iter 500** (-1.70%): Strongest early-learning boost
3. **LIF-fixed most stable overall**: Best at iter 1000-1500, close to Standard at 1999
4. **LIF-learnable underperforms** (+3.98%): Thresholds learn toward zero (pass-through)
5. **Early convergence pattern**: All LIF variants shine at iter 500, Standard catches up by 2000
6. **Ranking**: Standard > LIF-fixed > LIF-refractory > LIF-learnable
7. **NOTE**: No seed control in this run — different random inits per condition

### v2.5 Seeded Ablation Results (2000 iter, M4 CPU, 2026-02-14, seed=1337)

**5-condition ablation (Standard vs LIF-fixed vs LIF-learnable vs LIF-refractory vs Qwen-gate):**

| Iter | Standard | LIF-fixed | LIF-learnable | LIF-refrac | Qwen-gate |
|------|----------|-----------|---------------|------------|-----------|
| 0 | 4.2811 | 4.2811 | 4.2811 | 4.2811 | 4.1765 |
| 500 | 1.9779 | 2.0270 (+2.48%) | 2.0564 (+3.97%) | 2.0231 (+2.28%) | **1.7990 (-9.05%)** |
| 1000 | 1.6036 | 1.6261 (+1.40%) | 1.6258 (+1.38%) | **1.6008 (-0.17%)** | 1.6100 (+0.40%) |
| 1500 | 1.5278 | 1.5230 (-0.31%) | **1.5089 (-1.24%)** | **1.5088 (-1.24%)** | 1.5250 (-0.18%) |
| 1999 | 1.4923 | 1.4952 (+0.20%) | **1.4694 (-1.53%)** | **1.4676 (-1.65%)** | 1.4942 (+0.13%) |

**Key findings (seeded):**
1. **LIF-refractory WINS** (-1.65%): Best final val_loss with only 180 extra params
2. **LIF-learnable close second** (-1.53%): Reversal from unseeded run (+3.98%)
3. **Qwen-gate ties Standard** (+0.13%): 884K extra params buy almost nothing at 2000 iters
4. **LIF-fixed underperforms** (+0.20%): Fixed θ=1.0 too aggressive with this seed
5. **Qwen-gate dominates iter 500** (-9.05%): Huge early boost from 884K params, fades by 2000
6. **Seed matters enormously**: Unseeded ranking was opposite (Standard > LIF-fixed > rest)

**Critical insight — seed sensitivity:**
| Condition | Unseeded rank | Seeded rank | Stable? |
|-----------|-------------|-------------|---------|
| Standard | 1st | 3rd | seed-dependent |
| LIF-fixed | 2nd | 5th | seed-dependent |
| LIF-learnable | 4th (worst) | 2nd | **highly seed-dependent** |
| LIF-refractory | 3rd | 1st (best) | seed-dependent |
| Qwen-gate | N/A | 4th | TBD |

**Conclusion**: Single-seed results are unreliable. Multi-seed (3+) averaging required.
Next: Run seeds 42 and 668 for 3-seed mean ± std comparison.

**Biological interpretation:**
Brain thresholds aren't learned from scratch — they're genetically preset and refined.
LIF-fixed (preset θ=1.0) better matches this biological reality.
LIF-learnable finding mostly-zero thresholds shows the model prefers pass-through
when given the choice, but forced selectivity (fixed) produces better attention patterns.

**Learned parameter analysis (LIF-learnable, 2000 iters):**
Only 5/36 heads deviated significantly from pass-through:
- L4H3: θ=0.36 (strongest filter, late-layer selective head)
- L2H2: θ=0.20 (moderate filter)
- L3H4: θ=0.13 (moderate filter)
- L4H1: θ=-0.09, L4H4: θ=-0.13 (bypass mode — negative threshold)
Pattern: filtering emerges in mid-to-late layers (L2-L4), not early.

**Key finding - Head specialization (v2 MPS run):**
- Layer 0, Head 3: threshold=1.14, steepness=2.82 → strong selective filter
- Layer 0, Head 4: threshold=-0.34 → negative (bypass mode)
- Layer 3, Head 1: threshold=0.52 → moderate filter
- All other heads: threshold ≈ 0 → pass-through (identity behavior)

Only 3/36 heads deviate from pass-through. Role differentiation IS happening but is sparse.

**LIF-refractory parameter analysis (2000 iters, M4 CPU):**
Head specialization is STRONGER than LIF-learnable:
- L0H2: θ=1.12 (strongest filter of ALL heads, steepness=2.82) — gatekeeper
- L0H0: θ=-0.72 (strong bypass, leak=1.40) — wide-open gatherer
- L4H3: θ=0.40 (consistent across runs — this head always self-selects as filter)
- L3H3: θ=-0.15, L5H4: θ=-0.18 (moderate bypass)
- L5H0: θ=0.16, L5H5: θ=0.17 (mild late-layer filtering)

Refractory parameters:
- `refractory_strength`: all negative (softplus → 0.13-0.40), mild effect
- `cross_layer_weight`: L0 at -2.0 (minimum, no cross-layer), later layers -0.7 to -1.5
- Pattern: cross-layer inhibition increases in later layers (more inter-area interaction)
- L0's cross-layer weight stuck at init (-2.0) = first layer ignores previous state (expected)

**Key insight**: Refractory model has clearer head differentiation (θ range: -0.72 to +1.12)
than learnable model (θ range: -0.13 to +0.36). The additional refractory mechanism
encourages stronger role specialization, even though final val_loss is slightly worse.

**GPT analysis (via Kana, 2026-02-14):**
- Formula simplification: `p' = p × [leak + (1-leak)σ(k(p-θ))]`
- Improvement may be "gradient concentration" rather than "noise reduction"
- To disambiguate: compare Standard vs LIF-fixed-θ vs LIF-learnable-θ
- Shakespeare char-level has strong local dependency → test on long-range tasks
- Softmax-post thresholding is correct (operates in probability space)
- v2 alone is workshop-paper worthy with proper ablation + visualization

**Next experiments needed:**
1. Ablation: Standard vs fixed-θ vs learnable-θ vs refractory (4-condition) → **DONE (2026-02-14)**
2. Attention entropy comparison (Standard vs LIF) → **DONE (2026-02-14)**
3. Effective support size per head (how many tokens have >1% weight) → **DONE (2026-02-14)**
4. Gradient norm concentration analysis
5. Longer sequence / long-range dependency task

### Attention Analysis (v2 trained model, 2026-02-14)

**Tool:** `analyze.py --compare` (extracts attention maps from checkpoints)

**Entropy (higher=uniform, lower=peaked):**
- LIF overall: **2.11** — sharply focused
- Standard overall: 4.55 (near-uniform)
- LIF entropy varies massively across heads (0.01 to 4.19)

**Effective support size (tokens with >1% attention weight):**
- LIF overall: **13.4** tokens (out of ~128 avg available)
- Standard overall: 19.9 tokens
- LIF range: 1.0 to 31.5 per head

**First-token attention (attention sink):**
- LIF: 2.0%, Standard: 2.4% (both low — Shakespeare char-level may not trigger sinks)

**Key discovery — emergent head roles in LIF:**
| Head | Entropy | Support | Interpretation |
|------|---------|---------|----------------|
| L0H3 | 0.01 | 1.0 | "Pointer" — attends to exactly 1 token |
| L0H4 | 0.06 | 1.2 | "Pointer" — nearly single-token |
| L0H0 | 2.44 | 14.3 | "Local context" — moderate focus |
| L1H* | ~4.0 | ~28 | "Gatherers" — broad attention (whole layer) |
| L4H3 | 0.59 | 3.3 | "Focused" — narrow late-layer head |
| L4H1 | 0.70 | 3.8 | "Focused" — narrow late-layer head |

LIF learned a hierarchy: **broad early (gather) → progressively sharper (focus)**.
Layer 0 has "pointer" heads that self-selected; Layer 1 stays broad; Layers 3-5 narrow down.
This mirrors cortical processing: V1 (broad receptive fields) → V4/IT (selective).

Standard attention shows NO such specialization — all heads in all layers are nearly identical.

### v2.5: Refractory Period (2026-02-14, implemented)
Biological neurons have a refractory period after firing - their threshold
temporarily increases, preventing immediate re-firing. This prevents:
- Attention sinks (first-token over-attention: Qwen paper found 46.7%→4.8%)
- Monotonic attention patterns
- Same tokens being over-processed across layers

**Two refractory mechanisms:**

1. **Within-layer (column-load refractory):**
   Each key token's "load" = mean attention received across all queries.
   Heavily-attended tokens get a threshold boost → harder to attend to.
   ```
   column_load = mean_queries(att_probs)  # [B, H, 1, T]
   effective_θ = θ + softplus(ref_strength) * column_load
   ```

2. **Cross-layer (state passing):**
   Tokens heavily attended in layer L get a threshold boost in layer L+1.
   Different layers naturally attend to different tokens.
   ```
   prev_load = mean_heads_queries(att_probs_prev_layer)  # [B, T]
   effective_θ += sigmoid(cross_weight) * prev_load
   ```

**Parameters:** 180 total (v2's 108 + 72 new refractory params)
- `refractory_strength`: per-head, init softplus(-2)≈0.13 (mild)
- `cross_layer_weight`: per-head, init sigmoid(-2)≈0.12 (mild)
- Identity-like init: starts as v2, learns refractory dynamics

**Neuroscience basis:**
- After-Hyperpolarization (AHP): fast (<10ms), medium (10-100ms), slow (>100ms)
- Prevents attention sink = prevents excessive firing
- Sparse coding: brain uses 1-5% simultaneous activation
- Cross-layer = different cortical areas process different features

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

## Research Direction (2026-02-14, Kana review)

**Track: NeuroAI** — not ML performance, not pure neuroscience, but
"computational basis of cognitive architecture."

**Core question:**
"Do firing-threshold attention mechanisms exhibit temporal selectivity
and information filtering properties analogous to biological circuits?"

**Why NeuroAI, not ML performance:**
- Can't compete with Qwen at 15B scale from 10M model
- Head self-organization (pointer/gatherer/focuser) is a *property*, not performance
- Cortex framework already validates cognitive architecture in real-world (120h+)
- Ember is the *computational substrate* for Cortex's cognitive processing

**Biological correspondence (to be formalized):**
- `threshold` → membrane firing threshold (voltage at which AP fires)
- `leak` → membrane leak conductance (passive ion flow)
- `steepness` → input resistance (slope of voltage-current curve)
- `refractory_strength` → AHP amplitude (post-spike hyperpolarization)
- `cross_layer_weight` → inter-area lateral inhibition
- `fire_mask` → action potential (all-or-nothing above threshold)
- `smolder_mask` → subthreshold EPSPs (graded potentials below threshold)
- re-normalization → lateral inhibition / competitive selection

**Next steps (Kana's review, prioritized):**
1. Complete v2.5 ablation → **DONE (2026-02-14)**
2. Implement Qwen-gate baseline (same conditions as LIF) for direct comparison → **DONE**
3. Design working memory task (delayed match-to-sample or similar)
4. Formalize biological correspondence table
5. Test on temporal/noisy tasks where LIF properties should matter
6. Only then: v3 (temporal accumulation)

### Qwen-gate Baseline Implementation (2026-02-14)

**Formula**: `Y' = Y ⊙ σ(XW_θ)` applied at G1 position (after SDPA, before c_proj).

**Comparison with LIF — parameter efficiency:**
| Mechanism | Extra params | % of 10.65M model |
|-----------|-------------|-------------------|
| Standard | 0 | baseline |
| LIF learnable | 108 | +0.001% |
| LIF refractory | 180 | +0.002% |
| **Qwen gate** | **884,736** | **+8.3%** |

LIF is ~8,000x more parameter-efficient. This is a key differentiator.
At 15B scale (Qwen's regime), the gate overhead is negligible. At 10M scale, it's significant.

**Run**: `python3 train.py --qwen-gate` or `python3 train.py --ablation --qwen-gate`

**Kana's insight (2026-02-14)**: Head self-differentiation likely also occurs in
Constitutional AI training — specific heads self-select for safety/refusal behaviors.
This suggests LIF-like mechanisms are a general property of learned selectivity.

**Key risk (Kana's warning):**
"Brain-like" framing alone is weak. Reviewers want either performance wins
OR rigorous theoretical/empirical properties. Don't be half-and-half.

### v2.5 Ablation Interpretation (2026-02-14)

**The "Early Convergence Boost" hypothesis is confirmed across all LIF variants:**

At iter 500 (early training), every LIF condition beats Standard:
- LIF-refrac: -1.70% (strongest)
- LIF-learnable: -1.38%
- LIF-fixed: -1.10%

By iter 2000 (late training), Standard wins:
- Standard: 1.4683 (best)
- LIF-fixed: +0.90%
- LIF-refrac: +1.22%
- LIF-learnable: +3.98%

**Interpretation**: LIF's selective filtering helps during early learning by
concentrating gradients on important patterns (faster feature extraction).
But as training progresses and the model needs to capture finer distinctions,
the filtering becomes a bottleneck. This parallels development neuroscience:
strong initial selectivity (critical periods) gives way to refined plasticity.

**The NeuroAI story is not about performance:**
1. **Head self-differentiation** (pointer/gatherer/focuser) is unique to LIF
2. **Parameter efficiency** (108 params vs 884K for Qwen gate)
3. **Biological correspondence** (threshold, leak, refractory → neuroscience)
4. **Early convergence boost** → computational analog of developmental critical periods

**Next**: Seeded 5-condition ablation (Standard + 3 LIF + Qwen gate) for fair comparison.

### Adaptive Computation via LIF (Kana's insight, 2026-02-14)

**Core idea**: Learned LIF parameters automatically identify which heads/tokens
need full computation vs which can be approximated. No manual design needed.

Three levels of adaptive computation:

1. **Token-level skip (v3 Temporal LIF)**:
   Tokens with gate < threshold → skip MLP entirely.
   Direct FLOP reduction, measurable. Design already in v3 section.

2. **Head-level mixed precision**:
   Heads with θ ≈ 0 (pass-through) → INT8/FP16 computation.
   Heads with θ > 0.1 (active filter) → FP32 full precision.
   Example from v2.5 refractory results:
   - L0H2 (θ=1.12) → full precision (gatekeeper, critical)
   - L1H* (θ≈0) → INT8 safe (all pass-through)
   - L4H3 (θ=0.40) → full precision (selective head)
   Implementation: `torch.quantize_per_tensor` per head based on learned θ.

3. **Dynamic per-input routing**:
   At inference time, fire/smolder decision determines precision per token per head.
   Like MoE routing, but the "router" is the LIF threshold — no extra parameters.

**Comparison with MoE**:
- MoE: learned router (extra params) → discrete expert selection
- LIF: threshold IS the router (0 extra params) → continuous fire/smolder

**Potential contribution**: "LIF as automatic mixed-precision routing" —
the model tells you where to spend compute, for free.

**To validate**: Measure FLOP reduction from skipping/quantizing pass-through heads
while maintaining val_loss. Target: >30% FLOP savings with <0.5% loss degradation.
