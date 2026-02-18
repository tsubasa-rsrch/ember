"""
Liquid Ember Scaled — 4L/256d 3-seed ablation.

Compares CfC+LIF vs CfC-only at a larger scale.
Previous run: 2L/128d → base=1.6567, lif=1.6587 (nearly identical)
This run:     4L/256d → does scale help LIF differentiate?

Usage:
    python3 train_liquid_scaled.py          # Full 3-seed ablation
    python3 train_liquid_scaled.py --quick  # Single seed (quick test)

2026-02-18 — Tsubasa
"""

import os
import sys
import time
import pickle
import json

import numpy as np
import torch

from liquid_ember import LiquidEmber, LiquidEmberConfig

# ---------------------------------------------------------------------------
# Config — SCALED UP from 2L/128d to 4L/256d
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")
RESULTS_FILE = os.path.join(OUT_DIR, "liquid_scaled_results.json")

# Training hyperparams
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 3000
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LEARNING_RATE = 2e-4    # Slightly lower for bigger model
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.99
GRAD_CLIP = 1.0

# SCALED model size
N_LAYER = 4          # 2 → 4
N_EMBD = 256         # 128 → 256
CFC_UNITS = 384      # 192 → 384 (must be > N_EMBD + 2)
DROPOUT = 0.1

SEEDS = [42, 668, 1337]


def get_device():
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
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, device)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_one(use_lif, seed, device, train_data, val_data, vocab_size):
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    mode = "CfC+LIF" if use_lif else "CfC-only"
    n_params = model.get_num_params()
    print(f"\n{'='*60}")
    print(f"[{mode}] seed={seed} | {n_params/1e6:.2f}M params | {N_LAYER}L/{N_EMBD}d/{CFC_UNITS}u")
    print(f"{'='*60}")

    best_val = float('inf')
    t0 = time.time()

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            marker = " *" if losses['val'] < best_val else ""
            print(f"  iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s{marker}")
            if losses['val'] < best_val:
                best_val = losses['val']

        X, Y = get_batch(train_data, device)
        _, loss, _ = model(X, Y)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device == 'mps' and it % 100 == 0:
            torch.mps.empty_cache()

    total = time.time() - t0
    print(f"  Done: best_val={best_val:.4f} in {total:.0f}s")

    # Print LIF params if applicable
    if use_lif:
        for name, param in model.named_parameters():
            if 'threshold' in name and param.dim() > 0:
                print(f"  {name}: mean={param.abs().mean():.4f} std={param.std():.4f}")

    # Save checkpoint for analysis
    tag = "lif" if use_lif else "base"
    ckpt_path = os.path.join(OUT_DIR, f"liquid_scaled_{tag}_s{seed}.pt")
    torch.save({'model': model.state_dict(), 'config': config,
                'val_loss': best_val, 'seed': seed}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    return best_val, total


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Single seed only')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {N_LAYER}L / {N_EMBD}d / {CFC_UNITS} CfC units")
    print(f"Training: {MAX_ITERS} iters, batch={BATCH_SIZE}, block={BLOCK_SIZE}")

    train_data, val_data, meta = load_data()
    vocab_size = meta['vocab_size']

    seeds = [1337] if args.quick else SEEDS
    results = {"base": {}, "lif": {}}

    for seed in seeds:
        # CfC-only baseline
        val_base, time_base = train_one(
            use_lif=False, seed=seed, device=device,
            train_data=train_data, val_data=val_data, vocab_size=vocab_size)
        results["base"][str(seed)] = {"val_loss": val_base, "time": time_base}

        # CfC+LIF
        val_lif, time_lif = train_one(
            use_lif=True, seed=seed, device=device,
            train_data=train_data, val_data=val_data, vocab_size=vocab_size)
        results["lif"][str(seed)] = {"val_loss": val_lif, "time": time_lif}

    # Summary
    print(f"\n{'='*60}")
    print(f"LIQUID EMBER SCALED ABLATION ({N_LAYER}L/{N_EMBD}d/{CFC_UNITS}u)")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Seed':<8} {'Val Loss':<12} {'Time':<8}")
    print(f"{'-'*43}")

    for seed in seeds:
        s = str(seed)
        print(f"{'CfC-only':<15} {seed:<8} {results['base'][s]['val_loss']:<12.4f} {results['base'][s]['time']:<8.0f}s")
        print(f"{'CfC+LIF':<15} {seed:<8} {results['lif'][s]['val_loss']:<12.4f} {results['lif'][s]['time']:<8.0f}s")
        print()

    if len(seeds) > 1:
        base_vals = [results['base'][str(s)]['val_loss'] for s in seeds]
        lif_vals = [results['lif'][str(s)]['val_loss'] for s in seeds]
        base_mean = np.mean(base_vals)
        lif_mean = np.mean(lif_vals)
        base_std = np.std(base_vals)
        lif_std = np.std(lif_vals)
        delta = (lif_mean - base_mean) / base_mean * 100

        print(f"{'CfC-only mean':<20} {base_mean:.4f} ± {base_std:.4f}")
        print(f"{'CfC+LIF mean':<20} {lif_mean:.4f} ± {lif_std:.4f}")
        print(f"{'Delta':<20} {delta:+.2f}%")

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
