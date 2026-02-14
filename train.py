"""
Ember training script.
Trains on Shakespeare character-level data, comparing LIF vs standard attention.

Usage:
    python3 train.py                    # Train LIF model (default)
    python3 train.py --no-lif           # Train standard attention (baseline)
    python3 train.py --compare          # Train both and compare
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch

from model import Ember, EmberConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")

# Training hyperparams (small, for quick experiments on M4)
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500  # reduced eval frequency for long runs on MPS
EVAL_ITERS = 50  # reduced from 200 for faster iteration on MPS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.99
GRAD_CLIP = 1.0

# Model size (small enough for M2 8GB / M4 48GB)
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2


def get_device(force=None):
    if force:
        return force
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_data():
    train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    return train_data, val_data, meta


def get_batch(data, device):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(use_lif=True, max_iters=MAX_ITERS, force_device=None):
    device = get_device(force_device)
    print(f"Device: {device}")

    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']

    config = EmberConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=False,
        use_lif=use_lif,
    )

    model = Ember(config).to(device)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), device)

    os.makedirs(OUT_DIR, exist_ok=True)
    mode_str = "LIF" if use_lif else "Standard"

    best_val_loss = float('inf')
    history = []

    t0 = time.time()
    for iter_num in range(max_iters):
        # Evaluate
        if iter_num % EVAL_INTERVAL == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            print(f"[{mode_str}] iter {iter_num:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")
            history.append({
                'iter': iter_num,
                'train_loss': losses['train'].item(),
                'val_loss': losses['val'].item(),
                'elapsed': elapsed,
            })

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt = {
                    'model': model.state_dict(),
                    'config': config,
                    'iter': iter_num,
                    'best_val_loss': best_val_loss,
                }
                suffix = "lif" if use_lif else "std"
                torch.save(ckpt, os.path.join(OUT_DIR, f'ember_{suffix}.pt'))

        # Training step
        X, Y = get_batch(train_data, device)
        _, loss = model(X, Y)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # MPS memory management: prevent GPU memory fragmentation on long runs
        if device == 'mps' and iter_num % 100 == 0:
            torch.mps.empty_cache()

    total_time = time.time() - t0

    # Generate sample
    model.eval()
    itos = meta['itos']
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=500, temperature=0.8, top_k=200)
    text = ''.join([itos[i] for i in generated[0].tolist()])

    print(f"\n{'='*60}")
    print(f"[{mode_str}] Training complete in {total_time:.1f}s")
    print(f"[{mode_str}] Best val loss: {best_val_loss:.4f}")
    print(f"\n--- Generated sample ---")
    print(text[:500])
    print(f"{'='*60}")

    # Print LIF parameters if applicable
    if use_lif:
        print(f"\n--- Learned LIF parameters (per-head) ---")
        for name, param in model.named_parameters():
            if 'threshold' in name or 'leak' in name or 'steepness' in name:
                if param.dim() == 0:
                    print(f"  {name}: {param.item():.4f}")
                else:
                    vals = [f"{v:.4f}" for v in param.tolist()]
                    print(f"  {name}: [{', '.join(vals)}]")

    return history, best_val_loss, total_time


def main():
    parser = argparse.ArgumentParser(description='Train Ember')
    parser.add_argument('--no-lif', action='store_true', help='Use standard attention (baseline)')
    parser.add_argument('--compare', action='store_true', help='Train both and compare')
    parser.add_argument('--iters', type=int, default=MAX_ITERS, help='Max training iterations')
    parser.add_argument('--device', type=str, default=None, help='Force device (cpu/mps/cuda)')
    args = parser.parse_args()

    if args.compare:
        print("=" * 60)
        print("EMBER COMPARISON: LIF vs Standard Attention")
        print("=" * 60)

        print("\n>>> Training Standard Attention (baseline)...")
        hist_std, loss_std, time_std = train_model(use_lif=False, max_iters=args.iters, force_device=args.device)

        print("\n>>> Training LIF Attention...")
        hist_lif, loss_lif, time_lif = train_model(use_lif=True, max_iters=args.iters, force_device=args.device)

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Standard:  val_loss={loss_std:.4f}  time={time_std:.1f}s")
        print(f"LIF:       val_loss={loss_lif:.4f}  time={time_lif:.1f}s")
        diff = (loss_lif - loss_std) / loss_std * 100
        print(f"Difference: {diff:+.2f}%")
        if loss_lif < loss_std:
            print("LIF wins! Brain-inspired attention is better.")
        elif loss_lif > loss_std:
            print("Standard wins. LIF needs tuning.")
        else:
            print("Tie!")
    else:
        use_lif = not args.no_lif
        train_model(use_lif=use_lif, max_iters=args.iters, force_device=args.device)


if __name__ == '__main__':
    main()
