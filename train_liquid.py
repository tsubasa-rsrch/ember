"""
Training script for Liquid Ember (CfC + LIF language model).

Compares against Ember (Transformer + LIF) on Shakespeare char-level data.

Usage:
    python3 train_liquid.py                  # Train Liquid Ember with LIF
    python3 train_liquid.py --no-lif         # Train without LIF (CfC only baseline)
    python3 train_liquid.py --compare        # Train both and compare
    python3 train_liquid.py --vs-transformer # Compare Liquid Ember vs Ember Transformer

2026-02-17 — Tsubasa × Kana
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch

from liquid_ember import LiquidEmber, LiquidEmberConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")

# Training hyperparams
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 2000       # Fewer iters for initial experiments
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LEARNING_RATE = 3e-4   # Lower LR for CfC (ODE-based models prefer smaller LR)
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.99
GRAD_CLIP = 1.0

# Model size (compact — CfC is parameter-efficient)
N_LAYER = 2
N_EMBD = 128
CFC_UNITS = 192        # Must be > N_EMBD + 2 for AutoNCP
DROPOUT = 0.1


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
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_liquid(use_lif=True, max_iters=MAX_ITERS, force_device=None, seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(force_device)
    print(f"Device: {device}")

    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']

    config = LiquidEmberConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        cfc_units=CFC_UNITS,
        dropout=DROPOUT,
        use_lif=use_lif,
    )

    model = LiquidEmber(config).to(device)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), device)

    os.makedirs(OUT_DIR, exist_ok=True)
    mode_str = "Liquid-LIF" if use_lif else "Liquid-base"

    best_val_loss = float('inf')
    history = []

    t0 = time.time()
    for iter_num in range(max_iters):
        # Evaluate
        if iter_num % EVAL_INTERVAL == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            print(f"[{mode_str}] iter {iter_num:5d} | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"{elapsed:.1f}s")
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
                suffix = mode_str.lower().replace('-', '_')
                torch.save(ckpt, os.path.join(OUT_DIR, f'liquid_{suffix}.pt'))

        # Training step
        X, Y = get_batch(train_data, device)
        _, loss, _ = model(X, Y)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # MPS memory management
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
    print(f"[{mode_str}] Parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"\n--- Generated sample ---")
    print(text[:500])
    print(f"{'='*60}")

    # Print LIF parameters
    if use_lif:
        print(f"\n--- Learned LIF parameters ---")
        for name, param in model.named_parameters():
            if any(k in name for k in ['threshold', 'leak', 'steepness']):
                if param.dim() == 0:
                    print(f"  {name}: {param.item():.4f}")
                else:
                    vals = param.tolist()
                    if len(vals) <= 12:
                        print(f"  {name}: [{', '.join(f'{v:.4f}' for v in vals)}]")
                    else:
                        print(f"  {name}: mean={param.mean():.4f} std={param.std():.4f} "
                              f"[{param.min():.4f}..{param.max():.4f}]")

    return history, best_val_loss, total_time


def main():
    parser = argparse.ArgumentParser(description='Train Liquid Ember')
    parser.add_argument('--no-lif', action='store_true', help='CfC only (no LIF gate)')
    parser.add_argument('--compare', action='store_true', help='Compare CfC+LIF vs CfC-only')
    parser.add_argument('--iters', type=int, default=MAX_ITERS)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.compare:
        print("=" * 60)
        print("LIQUID EMBER: CfC+LIF vs CfC-only")
        print("=" * 60)

        print("\n>>> Training CfC-only (baseline)...")
        hist_base, loss_base, time_base = train_liquid(
            use_lif=False, max_iters=args.iters,
            force_device=args.device, seed=args.seed)

        print("\n>>> Training CfC+LIF...")
        hist_lif, loss_lif, time_lif = train_liquid(
            use_lif=True, max_iters=args.iters,
            force_device=args.device, seed=args.seed)

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        diff = (loss_lif - loss_base) / loss_base * 100
        print(f"  CfC-only:  val_loss={loss_base:.4f}  time={time_base:.1f}s")
        print(f"  CfC+LIF:   val_loss={loss_lif:.4f}  time={time_lif:.1f}s  diff={diff:+.2f}%")
        winner = "CfC+LIF" if loss_lif < loss_base else "CfC-only"
        print(f"\n  Winner: {winner}")

    else:
        use_lif = not args.no_lif
        train_liquid(use_lif=use_lif, max_iters=args.iters,
                     force_device=args.device, seed=args.seed)


if __name__ == '__main__':
    main()
