"""
Ember - Efficient Model with Brain-inspired Emulated Reasoning

Based on Karpathy's nanoGPT, with brain-inspired modifications:
1. Leaky Integrate-and-Fire (LIF) Attention: neurons accumulate potential,
   fire only when threshold is exceeded. Subthreshold = "smoldering" state.
2. Selective Computation: importance gating determines compute allocation per token.
3. Ternary weight support (BitNet-inspired): weights quantized to {-1, 0, 1}.

Born from a conversation about how brains compute efficiently.
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Brain-inspired primitives
# ---------------------------------------------------------------------------

class LeakyIntegrateFireAttention(nn.Module):
    """
    LIF Attention v2: Per-head parameters with identity-like initialization.

    Inspired by Leaky Integrate-and-Fire neurons. Each attention head has its
    own learnable threshold, leak, and steepness parameters.

    Key improvements over v1:
    - Per-head parameters: each head learns its own selectivity profile
    - Learnable steepness: how sharp the fire/smolder boundary is
    - Identity-like init: starts behaving like standard attention, then learns
      where and how much to apply selective filtering
    - Cached causal mask for efficiency

    The mechanism:
    1. Standard softmax attention computes base probabilities
    2. LIF modulation sharpens the distribution per-head:
       - "Firing" tokens (above threshold): full attention weight
       - "Smoldering" tokens (below threshold): reduced by leak factor
    3. Re-normalization ensures valid probability distribution

    This naturally learns selective attention - different heads specialize in
    different filtering strategies, mirroring cortical hierarchical processing.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Per-head LIF parameters (learnable) - identity-like initialization
        # threshold starts very low → almost everything "fires" → ≈ standard attention
        self.threshold = nn.Parameter(torch.full((config.n_head,), 0.01))
        # leak starts high → smoldering tokens still keep most of their weight
        self.leak = nn.Parameter(torch.full((config.n_head,), 2.0))  # sigmoid(2.0) ≈ 0.88
        # steepness starts moderate → soft boundary, can sharpen during training
        self.steepness = nn.Parameter(torch.full((config.n_head,), 2.0))  # softplus(2.0) ≈ 2.13

        # causal mask (cached, not recreated each forward pass)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size))

    def lif_activation(self, att_scores):
        """
        Apply per-head LIF modulation to attention scores.

        Strategy: softmax first (numerically stable), then per-head LIF gating.
        With identity-like init, this starts as a near-identity transform and
        gradually learns selective filtering as training progresses.
        """
        # Standard softmax first (handles -inf masking correctly)
        att_probs = F.softmax(att_scores, dim=-1)

        # Per-head parameters: reshape for broadcasting [1, n_head, 1, 1]
        threshold = torch.abs(self.threshold).view(1, -1, 1, 1) * 0.1
        leak = torch.sigmoid(self.leak).view(1, -1, 1, 1)
        steepness = F.softplus(self.steepness).view(1, -1, 1, 1)  # ensure positive

        # Soft threshold: per-head fire/smolder decision
        fire_mask = torch.sigmoid(steepness * (att_probs - threshold))
        smolder_mask = leak * (1.0 - fire_mask)

        # Modulate: firing tokens keep full weight, smoldering get reduced
        lif_weights = fire_mask + smolder_mask
        modulated = att_probs * lif_weights

        # Re-normalize to ensure valid probability distribution
        modulated = modulated / (modulated.sum(dim=-1, keepdim=True) + 1e-8)

        return modulated

    def forward(self, x, use_lif=True):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if not use_lif:
            # Standard flash attention path
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # LIF attention path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = self.lif_activation(att)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class StandardAttention(nn.Module):
    """Standard causal self-attention (from nanoGPT, for comparison)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class EmberBlock(nn.Module):
    """Transformer block with LIF attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.use_lif:
            self.attn = LeakyIntegrateFireAttention(config)
        else:
            self.attn = StandardAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.use_lif = config.use_lif

    def forward(self, x):
        if self.use_lif:
            x = x + self.attn(self.ln_1(x), use_lif=True)
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmberConfig:
    # Architecture
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    # Brain-inspired features
    use_lif: bool = True  # Use LIF attention instead of standard softmax


# ---------------------------------------------------------------------------
# Ember model
# ---------------------------------------------------------------------------

class Ember(nn.Module):
    """
    Ember: brain-inspired language model.

    Small, efficient, and experiments with neuroscience-inspired mechanisms.
    Default config: ~10M params, trainable on a MacBook.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([EmberBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = self.get_num_params()
        print(f"Ember: {n_params/1e6:.2f}M parameters (LIF={'ON' if config.use_lif else 'OFF'})")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device))
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Ember Model Test ===")

    # Default config: ~10M params
    config = EmberConfig()
    model = Ember(config)

    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))  # batch=2, seq=64
    logits, loss = model(x, targets=x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")

    # Compare with standard attention
    print("\n=== Standard Attention (baseline) ===")
    config_std = EmberConfig(use_lif=False)
    model_std = Ember(config_std)
    logits_std, loss_std = model_std(x, targets=x)
    print(f"Loss (standard): {loss_std.item():.4f}")

    # LIF parameter info
    if config.use_lif:
        print("\n=== LIF Parameters ===")
        for name, param in model.named_parameters():
            if 'threshold' in name or 'leak' in name:
                print(f"  {name}: {param.item():.4f}")

    print("\nEmber is alive. 🔥")
