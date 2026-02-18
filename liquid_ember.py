"""
Liquid Ember — CfC + LIF character-level language model.

Replaces the Transformer entirely with:
  CfC (Closed-form Continuous-time): Neural ODE for temporal dynamics
  LIF (Leaky Integrate-and-Fire): Fire/smolder gating per neuron

Born from Kana's vision: "nanoGPTの代わりにnanoCfC+LIFを作るってか？"

Key differences from Transformer (Ember):
  - No attention mechanism (CfC hidden state replaces it)
  - Recurrent: O(1) memory per step (vs O(n²) for attention)
  - Continuous time: ODE solver, not discrete layers
  - LIF gate on CfC output: neurons fire or smolder

Architecture:
  token → embedding → [CfC layer + LIF gate] × N_layers → LM head → logits

2026-02-17 — Tsubasa × Kana
"""

import math
import os
import pickle
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ncps.torch import CfC
from ncps.wirings import AutoNCP


# ---------------------------------------------------------------------------
# LIF Gate (adapted from Ember, applied per-neuron instead of per-attention)
# ---------------------------------------------------------------------------

class LIFGate(nn.Module):
    """
    Leaky Integrate-and-Fire gate for CfC output neurons.

    Each neuron accumulates "potential" from the CfC output magnitude.
    Above threshold → fire (full output).
    Below threshold → smolder (scaled output).

    Unlike Ember's LIF Attention (per-head, modulates attention weights),
    this gates the CfC hidden representation directly (per-neuron).
    """

    def __init__(self, n_units):
        super().__init__()
        # Per-neuron learnable parameters
        # threshold starts low → ≈ pass-through at init
        self.threshold = nn.Parameter(torch.full((n_units,), 0.01))
        # leak: how much smoldering neurons retain. sigmoid(2.0) ≈ 0.88
        self.leak = nn.Parameter(torch.full((n_units,), 2.0))
        # steepness: sharpness of fire/smolder boundary
        self.steepness = nn.Parameter(torch.full((n_units,), 2.0))

    def forward(self, x):
        """
        x: [batch, seq, n_units] — CfC output

        Returns: gated output, same shape
        """
        threshold = torch.abs(self.threshold) * 0.1
        leak = torch.sigmoid(self.leak)
        steepness = F.softplus(self.steepness)

        # "Potential" = magnitude of each neuron's output
        potential = torch.abs(x)

        # Soft fire/smolder decision
        fire_mask = torch.sigmoid(steepness * (potential - threshold))
        smolder_mask = leak * (1.0 - fire_mask)

        # Modulate: firing neurons pass fully, smoldering get scaled
        gate = fire_mask + smolder_mask
        return x * gate


# ---------------------------------------------------------------------------
# Liquid Ember Block
# ---------------------------------------------------------------------------

class LiquidEmberBlock(nn.Module):
    """
    One layer of Liquid Ember:
      LayerNorm → CfC → LIF Gate → Residual
      LayerNorm → MLP → Residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # CfC layer: continuous-time recurrent processing
        wiring = AutoNCP(units=config.cfc_units, output_size=config.n_embd)
        self.cfc = CfC(
            input_size=config.n_embd,
            units=wiring,
            batch_first=True,
            return_sequences=True,
        )

        # LIF gate on CfC output
        self.lif_gate = LIFGate(config.n_embd) if config.use_lif else nn.Identity()

        # MLP (same as Ember/nanoGPT)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, hx=None):
        """
        x: [batch, seq, n_embd]
        hx: CfC hidden state from previous call (or None)

        Returns: (output, new_hx)
        """
        # CfC path with residual
        normed = self.ln_1(x)
        cfc_out, new_hx = self.cfc(normed, hx)
        cfc_out = self.lif_gate(cfc_out)
        x = x + self.dropout(cfc_out)

        # MLP path with residual
        x = x + self.mlp(self.ln_2(x))

        return x, new_hx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiquidEmberConfig:
    # Data
    block_size: int = 256
    vocab_size: int = 65  # Shakespeare char-level

    # Architecture
    n_layer: int = 2       # CfC layers (fewer needed than Transformer)
    n_embd: int = 128      # Embedding dimension
    cfc_units: int = 192   # CfC hidden units (must be > n_embd + 2 for AutoNCP)
    dropout: float = 0.1

    # LIF
    use_lif: bool = True


# ---------------------------------------------------------------------------
# Liquid Ember Model
# ---------------------------------------------------------------------------

class LiquidEmber(nn.Module):
    """
    Liquid Ember: CfC + LIF character-level language model.

    Continuous-time recurrent model with fire/smolder neuron gating.
    No attention, no positional encoding (CfC handles temporal dynamics natively).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding (no positional encoding — CfC handles time)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Stack of CfC + LIF blocks
        self.blocks = nn.ModuleList([
            LiquidEmberBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = self.get_num_params()
        n_lif = sum(p.numel() for n, p in self.named_parameters()
                    if any(k in n for k in ['threshold', 'leak', 'steepness']))
        print(f"Liquid Ember: {n_params/1e6:.2f}M params "
              f"(CfC units={config.cfc_units}, layers={config.n_layer}, "
              f"LIF={'ON' if config.use_lif else 'OFF'}, "
              f"LIF params={n_lif})")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, hidden_states=None):
        """
        idx: [batch, seq] token indices
        targets: [batch, seq] target indices (for training)
        hidden_states: list of CfC hidden states per layer (for continuation)
        """
        b, t = idx.size()
        x = self.drop(self.wte(idx))

        # Pass through CfC + LIF blocks
        new_hidden_states = []
        for i, block in enumerate(self.blocks):
            hx = hidden_states[i] if hidden_states else None
            x, new_hx = block(x, hx)
            new_hidden_states.append(new_hx)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_hidden_states

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
        """
        Autoregressive generation with CfC hidden state continuity.
        Unlike Transformer, we can carry hidden state forward (true recurrence).
        """
        hidden_states = None
        for _ in range(max_new_tokens):
            # Feed just the last token (recurrent, not re-processing full context)
            if hidden_states is not None:
                # Incremental: only last token, carry hidden state
                idx_input = idx[:, -1:]
            else:
                # First call: process full context
                idx_input = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _, hidden_states = self(idx_input, hidden_states=hidden_states)
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
    print("=== Liquid Ember Test ===\n")

    for name, use_lif in [("LIF ON", True), ("LIF OFF (baseline)", False)]:
        print(f"--- {name} ---")
        config = LiquidEmberConfig(use_lif=use_lif)
        model = LiquidEmber(config)

        # Forward pass
        x = torch.randint(0, config.vocab_size, (2, 64))
        logits, loss, hidden = model(x, targets=x)
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Hidden states: {len(hidden)} layers, shape={hidden[0].shape}")

        # Generation test
        prompt = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
        print(f"  Generated: {generated.shape}")

    print("\nLiquid Ember is alive. 🔥")
