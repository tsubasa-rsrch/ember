"""
Audio Liquid Ember — CfC + LIF audio classification model.

Multimodal extension of Liquid Ember: proves LIF gating is modality-independent.
Uses mel spectrogram features instead of text tokens.

Architecture:
  audio → mel spectrogram → projection → [CfC + LIF] × N → mean_pool → classifier

2026-02-19 — Tsubasa × Kana
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ncps.torch import CfC
from ncps.wirings import AutoNCP

from liquid_ember import LIFGate


# ---------------------------------------------------------------------------
# Audio Liquid Ember Block (same as LiquidEmberBlock)
# ---------------------------------------------------------------------------

class AudioLiquidEmberBlock(nn.Module):
    """CfC + LIF block for audio features."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        wiring = AutoNCP(units=config.cfc_units, output_size=config.n_embd)
        self.cfc = CfC(
            input_size=config.n_embd,
            units=wiring,
            batch_first=True,
            return_sequences=True,
        )

        self.lif_gate = LIFGate(config.n_embd) if config.use_lif else nn.Identity()

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, hx=None):
        normed = self.ln_1(x)
        cfc_out, new_hx = self.cfc(normed, hx)
        cfc_out = self.lif_gate(cfc_out)
        x = x + self.dropout(cfc_out)
        x = x + self.mlp(self.ln_2(x))
        return x, new_hx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AudioLiquidEmberConfig:
    # Audio
    n_mels: int = 80           # Mel spectrogram bins
    sample_rate: int = 16000   # Speech Commands is 16kHz
    n_fft: int = 400           # 25ms window at 16kHz
    hop_length: int = 160      # 10ms hop at 16kHz
    audio_length: int = 16000  # 1 second at 16kHz

    # Architecture
    n_layer: int = 4
    n_embd: int = 128
    cfc_units: int = 192
    dropout: float = 0.1
    num_classes: int = 35      # Speech Commands v2

    # LIF
    use_lif: bool = True


# ---------------------------------------------------------------------------
# Audio Liquid Ember Model
# ---------------------------------------------------------------------------

class AudioLiquidEmber(nn.Module):
    """
    Audio classification with CfC + LIF.

    mel spectrogram → linear projection → [CfC + LIF] × N → mean pool → classifier
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Mel spectrogram → embedding dimension
        self.input_proj = nn.Linear(config.n_mels, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # CfC + LIF blocks
        self.blocks = nn.ModuleList([
            AudioLiquidEmberBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.classifier = nn.Linear(config.n_embd, config.num_classes)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        n_lif = sum(p.numel() for n, p in self.named_parameters()
                    if any(k in n for k in ['threshold', 'leak', 'steepness']))
        print(f"Audio Liquid Ember: {n_params/1e6:.2f}M params "
              f"(CfC units={config.cfc_units}, layers={config.n_layer}, "
              f"LIF={'ON' if config.use_lif else 'OFF'}, "
              f"LIF params={n_lif})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, mel, targets=None):
        """
        mel: [batch, time, n_mels] — mel spectrogram frames
        targets: [batch] — class labels

        Returns: logits, loss
        """
        # Project mel features to embedding dimension
        x = self.drop(self.input_proj(mel))

        # CfC + LIF blocks
        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)

        # Mean pooling over time
        x = x.mean(dim=1)  # [batch, n_embd]

        logits = self.classifier(x)  # [batch, num_classes]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        return optimizer


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Audio Liquid Ember Test ===\n")

    for name, use_lif in [("LIF ON", True), ("LIF OFF (baseline)", False)]:
        print(f"--- {name} ---")
        config = AudioLiquidEmberConfig(use_lif=use_lif)
        model = AudioLiquidEmber(config)

        # Simulate mel spectrogram input: [batch=4, time=100, mels=80]
        mel = torch.randn(4, 100, config.n_mels)
        targets = torch.randint(0, config.num_classes, (4,))

        logits, loss = model(mel, targets)
        print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")
        print(f"  Pred: {logits.argmax(dim=-1).tolist()}")

    print("\nAudio Liquid Ember is alive. 🔥🎵")
