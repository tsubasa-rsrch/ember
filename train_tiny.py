"""
Ember-Tiny: Small-scale proof that LIF gating is a structural property, not scale-dependent.

Trains 3 architectures at ~0.8M params on Shakespeare:
  1. Standard Transformer (vanilla, no LIF)
  2. Ember-Tiny (Transformer + LIF gate)
  3. Liquid Ember-Tiny (CfC + LIF gate)

Key claim: "巨大GPU不要で構造的性質が出る" — structural effects emerge at any scale.

Usage:
    python3 train_tiny.py                  # Compare all 3 architectures (seed=42)
    python3 train_tiny.py --ablation       # 3-seed ablation (42, 668, 1337)
    python3 train_tiny.py --model standard # Single model
    python3 train_tiny.py --model lif      # Single model with LIF
    python3 train_tiny.py --model cfc      # Single CfC+LIF model

2026-02-23 — Tsubasa (社長モードで着手)
"""

import os
import sys
import time
import json
import pickle
import argparse

import numpy as np
import torch

# Import from existing codebase
from model import Ember, EmberConfig
from liquid_ember import LiquidEmber, LiquidEmberConfig

# ---------------------------------------------------------------------------
# Tiny Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out/tiny")
RESULTS_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/results")

# Tiny model hyperparams — designed to match across architectures
BATCH_SIZE = 64
BLOCK_SIZE = 128       # shorter context (tiny model)
MAX_ITERS = 3000       # enough for convergence at this scale
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LEARNING_RATE = 1e-3   # standard for Transformer
LEARNING_RATE_CFC = 3e-4  # ODE models need lower LR
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Scale presets — same architecture family, different sizes
SCALES = {
    'xs': {  # ~0.4M params
        'n_layer': 2, 'n_head': 4, 'n_embd': 128, 'dropout': 0.1,
        'cfc_layers': 2, 'cfc_units': 192, 'cfc_embd': 128,
    },
    'small': {  # ~2M params
        'n_layer': 4, 'n_head': 4, 'n_embd': 192, 'dropout': 0.15,
        'cfc_layers': 3, 'cfc_units': 256, 'cfc_embd': 192,
    },
    'medium': {  # ~5M params, 6 layers for depth study
        'n_layer': 6, 'n_head': 8, 'n_embd': 256, 'dropout': 0.2,
        'cfc_layers': 4, 'cfc_units': 320, 'cfc_embd': 256,
    },
    'mid': {  # ~8M params, 6 layers with 320d for width midpoint (U-curve valley hunt)
        'n_layer': 6, 'n_head': 8, 'n_embd': 320, 'dropout': 0.2,
        'cfc_layers': 4, 'cfc_units': 384, 'cfc_embd': 320,
    },
    'wide': {  # ~11M params, 6 layers with wider embeddings for width study
        'n_layer': 6, 'n_head': 8, 'n_embd': 384, 'dropout': 0.2,
        'cfc_layers': 4, 'cfc_units': 512, 'cfc_embd': 384,  # units > embd+2 for AutoNCP
    },
}


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_data():
    train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return train_data, val_data, meta


def get_batch(data, device):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    results = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, device)
            model_out = model(X, Y)
            loss = model_out[1]  # logits, loss[, hidden_states]
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(model_type, seed, device, train_data, val_data, meta, scale='xs'):
    """Train a single model configuration. Returns history dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab_size = meta['vocab_size']
    s = SCALES[scale]

    if model_type in ('cfc', 'cfc_base'):
        use_lif_cfc = (model_type == 'cfc')
        config = LiquidEmberConfig(
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_layer=s['cfc_layers'],
            n_embd=s['cfc_embd'],
            cfc_units=s['cfc_units'],
            dropout=s['dropout'],
            use_lif=use_lif_cfc,
        )
        model = LiquidEmber(config).to(device)
        lr = LEARNING_RATE_CFC
        mode_str = "CfC+LIF" if use_lif_cfc else "CfC-only"
    elif model_type == 'sigmoid':
        config = EmberConfig(
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_layer=s['n_layer'],
            n_head=s['n_head'],
            n_embd=s['n_embd'],
            dropout=s['dropout'],
            bias=False,
            use_lif=False,
            use_sigmoid_gate=True,
            lif_mode='learnable',
        )
        model = Ember(config).to(device)
        lr = LEARNING_RATE
        mode_str = "Sigmoid"
    else:
        use_lif = (model_type == 'lif')
        config = EmberConfig(
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_layer=s['n_layer'],
            n_head=s['n_head'],
            n_embd=s['n_embd'],
            dropout=s['dropout'],
            bias=False,
            use_lif=use_lif,
            lif_mode='learnable',
        )
        model = Ember(config).to(device)
        lr = LEARNING_RATE
        mode_str = "LIF" if use_lif else "Standard"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"[{mode_str}] seed={seed} | {n_params/1e6:.3f}M params | device={device}")
    print(f"{'='*60}")

    optimizer = model.configure_optimizers(WEIGHT_DECAY, lr, (0.9, 0.99), device)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    history = []

    t0 = time.time()
    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            print(f"  iter {iter_num:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")
            history.append({
                'iter': iter_num,
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'elapsed': elapsed,
            })
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        X, Y = get_batch(train_data, device)
        out = model(X, Y)
        loss = out[1]  # logits, loss[, hidden_states]
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device == 'mps' and iter_num % 100 == 0:
            torch.mps.empty_cache()

    total_time = time.time() - t0
    print(f"  Done: best_val={best_val_loss:.4f} in {total_time:.1f}s")

    return {
        'model_type': model_type,
        'mode_str': mode_str,
        'seed': seed,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'history': history,
    }


# ---------------------------------------------------------------------------
# Comparison & Reporting
# ---------------------------------------------------------------------------

def print_results(all_results):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print("EMBER-TINY RESULTS SUMMARY")
    print(f"{'='*80}")

    # Group by model type
    by_type = {}
    for r in all_results:
        mt = r['model_type']
        if mt not in by_type:
            by_type[mt] = []
        by_type[mt].append(r)

    # Print per-seed results
    seeds_used = sorted(set(r['seed'] for r in all_results))
    header = f"{'Model':<12}"
    for s in seeds_used:
        header += f" | Seed {s:>4}"
    header += " | Mean ± Std    | vs Standard"
    print(header)
    print("-" * len(header))

    standard_mean = None
    for mt in ['standard', 'lif', 'sigmoid', 'cfc_base', 'cfc']:
        if mt not in by_type:
            continue
        results = by_type[mt]
        label = results[0]['mode_str']
        vals = {r['seed']: r['best_val_loss'] for r in results}

        row = f"{label:<12}"
        for s in seeds_used:
            if s in vals:
                row += f" | {vals[s]:.4f} "
            else:
                row += " |   --   "

        val_list = [vals[s] for s in seeds_used if s in vals]
        if len(val_list) > 1:
            mean = np.mean(val_list)
            std = np.std(val_list)
            row += f" | {mean:.4f}±{std:.4f}"
        elif len(val_list) == 1:
            mean = val_list[0]
            std = 0.0
            row += f" | {mean:.4f}        "
        else:
            mean = None
            row += " |               "

        if mt == 'standard':
            standard_mean = mean
            row += " | baseline"
        elif standard_mean and mean:
            delta = (mean - standard_mean) / standard_mean * 100
            row += f" | {delta:+.2f}%"
        else:
            row += " |"

        print(row)

    print(f"\nParams: {all_results[0]['n_params']/1e6:.3f}M | Block: {BLOCK_SIZE} | Iters: {MAX_ITERS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ember-Tiny: scale-independent LIF effects")
    parser.add_argument('--model', choices=['standard', 'lif', 'sigmoid', 'cfc', 'cfc_base', 'all'], default='all',
                        help="Which model(s) to train")
    parser.add_argument('--ablation', action='store_true',
                        help="Run 3-seed ablation (42, 668, 1337)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (single run)")
    parser.add_argument('--iters', type=int, default=None,
                        help="Override MAX_ITERS")
    parser.add_argument('--scale', choices=['xs', 'small', 'medium', 'mid', 'wide'], default='xs',
                        help="Model scale preset (xs=0.4M, small=2M, medium=5M, mid=8M, wide=14M)")
    args = parser.parse_args()

    global MAX_ITERS
    if args.iters:
        MAX_ITERS = args.iters

    device = get_device()
    train_data, val_data, meta = load_data()

    seeds = [42, 668, 1337] if args.ablation else [args.seed]
    models = ['standard', 'lif', 'sigmoid', 'cfc'] if args.model == 'all' else [args.model]

    print(f"Scale: {args.scale} | Models: {models} | Seeds: {seeds} | Iters: {MAX_ITERS}")

    all_results = []
    for seed in seeds:
        for model_type in models:
            result = train_one(model_type, seed, device, train_data, val_data, meta, scale=args.scale)
            all_results.append(result)

    print_results(all_results)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"tiny_{timestamp}.json")
    # Convert to serializable
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items()}
        serializable.append(sr)
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
