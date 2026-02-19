"""
Train Audio Liquid Ember on Speech Commands v2.

Multimodal LIF experiment: does LIF create the same hierarchical
organization on audio as it does on text?

Usage:
    python train_audio.py --use_lif --seed 42
    python train_audio.py --no_lif --seed 42

2026-02-19 — Tsubasa × Kana
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from audio_liquid_ember import AudioLiquidEmber, AudioLiquidEmberConfig


# ---------------------------------------------------------------------------
# Speech Commands Dataset wrapper
# ---------------------------------------------------------------------------

# Standard 35-word label set for Speech Commands v2
LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
    'wow', 'yes', 'zero'
]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


class SpeechCommandsDataset(Dataset):
    """Wraps torchaudio SPEECHCOMMANDS with mel spectrogram preprocessing."""

    def __init__(self, root, subset, config):
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root, download=False, subset=subset)
        self.config = config

        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

        # Filter to 35-word vocabulary
        self.indices = []
        for i in range(len(self.ds)):
            _, _, label, *_ = self.ds[i]
            if label in LABEL_TO_IDX:
                self.indices.append(i)

        print(f"  {subset}: {len(self.indices)}/{len(self.ds)} samples "
              f"(filtered to {len(LABELS)} classes)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.ds[self.indices[idx]]

        # Pad/trim to fixed length
        target_len = self.config.audio_length
        if waveform.size(1) < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.size(1)))
        else:
            waveform = waveform[:, :target_len]

        # Mel spectrogram: [1, n_mels, time] → [time, n_mels]
        mel = self.mel_transform(waveform)  # [1, n_mels, time]
        mel = mel.squeeze(0).transpose(0, 1)  # [time, n_mels]

        # Log-mel
        mel = torch.log(mel + 1e-9)

        label_idx = LABEL_TO_IDX[label]
        return mel, label_idx


from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    config = AudioLiquidEmberConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        cfc_units=args.cfc_units,
        use_lif=args.use_lif,
        dropout=args.dropout,
    )

    # Dataset
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    print(f"\nLoading Speech Commands v2 from {data_root}...")
    train_ds = SpeechCommandsDataset(data_root, 'training', config)
    val_ds = SpeechCommandsDataset(data_root, 'validation', config)
    test_ds = SpeechCommandsDataset(data_root, 'testing', config)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    # Model
    model = AudioLiquidEmber(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
    )

    # Output
    tag = f"{'lif' if config.use_lif else 'base'}_s{args.seed}"
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'audio_{tag}.pt')

    best_val_acc = 0.0
    log_lines = []

    print(f"\nTraining: {tag}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'Time':>8}")
    print("-" * 70)

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for mel, labels in train_loader:
            mel = mel.to(device)
            labels = labels.to(device)

            logits, loss = model(mel, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * mel.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += mel.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(device)
                labels = labels.to(device)
                logits, loss = model(mel, labels)
                val_loss_sum += loss.item() * mel.size(0)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += mel.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        log_line = (f"{epoch+1:>6} {train_loss:>12.4f} {train_acc:>10.4f} "
                    f"{val_loss:>10.4f} {val_acc:>10.4f} {elapsed:>8.1f}s")
        print(log_line)
        log_lines.append(log_line)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'config': config,
                'model': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch + 1,
                'seed': args.seed,
            }
            torch.save(checkpoint, out_path)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {out_path}")

    # Test
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)
            labels = labels.to(device)
            logits, _ = model(mel, labels)
            test_correct += (logits.argmax(dim=-1) == labels).sum().item()
            test_total += mel.size(0)

    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f}")

    # Save log
    log_path = os.path.join(out_dir, f'audio_{tag}.log')
    with open(log_path, 'w') as f:
        f.write(f"tag={tag} seed={args.seed} use_lif={config.use_lif}\n")
        f.write(f"best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}\n")
        f.write(f"n_layer={config.n_layer} n_embd={config.n_embd} "
                f"cfc_units={config.cfc_units}\n\n")
        for line in log_lines:
            f.write(line + '\n')
    print(f"Log: {log_path}")

    # Print LIF thresholds
    if config.use_lif:
        print(f"\nLIF threshold analysis:")
        for i, block in enumerate(model.blocks):
            if hasattr(block.lif_gate, 'threshold'):
                t = torch.abs(block.lif_gate.threshold) * 0.1
                print(f"  L{i}: mean={t.mean():.4f}, std={t.std():.4f}, "
                      f"max={t.max():.4f}, min={t.min():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_lif', action='store_true', default=True)
    parser.add_argument('--no_lif', action='store_true')
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--cfc_units', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    if args.no_lif:
        args.use_lif = False

    train(args)
