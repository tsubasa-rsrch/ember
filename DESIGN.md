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
