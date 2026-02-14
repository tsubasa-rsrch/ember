"""
Ember analysis tools.
Extracts and visualizes attention patterns from trained models.

Usage:
    python3 analyze.py --model out/ember_lif.pt    # Analyze single model
    python3 analyze.py --compare                     # Compare all saved models
    python3 analyze.py --ablation-log /tmp/ember_ablation_v25.log  # Parse ablation results
"""

import os
import sys
import argparse
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F

from model import Ember, EmberConfig

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")


def load_data():
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return val_data, meta


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = Ember(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def get_attention_maps(model, x):
    """Extract attention probability maps from all layers.

    Returns list of [B, H, T, T] tensors, one per layer.
    """
    import math

    device = x.device
    b, t = x.size()

    tok_emb = model.transformer.wte(x)
    pos_emb = model.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device))
    hidden = model.transformer.drop(tok_emb + pos_emb)

    attention_maps = []
    refractory_state = None

    for block in model.transformer.h:
        attn = block.attn
        ln_out = block.ln_1(hidden)
        B, T, C = ln_out.size()

        q, k, v = attn.c_attn(ln_out).split(attn.n_embd, dim=2)
        k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
        q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)

        if block.use_lif:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(attn.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att_probs, new_refractory = attn.lif_activation(att, refractory_state)
            refractory_state = new_refractory
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
            att = att.masked_fill(causal == 0, float('-inf'))
            att_probs = F.softmax(att, dim=-1)
            refractory_state = None

        attention_maps.append(att_probs.detach())

        # Forward through block to get hidden state for next layer
        hidden, refractory_state = block(hidden, refractory_state if block.use_lif else None)

    return attention_maps


def compute_entropy(att_maps):
    """Compute attention entropy per head per layer.

    Entropy = -sum(p * log(p)), higher = more uniform, lower = more peaked.
    Returns dict with per-layer per-head entropy stats.
    """
    results = {}
    for layer_idx, att in enumerate(att_maps):
        # att: [B, H, T, T]
        # Clamp for numerical stability
        att_clamped = att.clamp(min=1e-10)
        entropy = -(att_clamped * att_clamped.log()).sum(dim=-1)  # [B, H, T]
        # Average over batch and query positions
        mean_entropy = entropy.mean(dim=(0, 2))  # [H]
        results[layer_idx] = {
            'mean': mean_entropy.numpy(),
            'per_query': entropy.mean(dim=0).numpy(),  # [H, T]
        }
    return results


def compute_support_size(att_maps, threshold=0.01):
    """Compute effective support size per head (tokens with >threshold weight).

    Lower = sparser attention (more selective).
    """
    results = {}
    for layer_idx, att in enumerate(att_maps):
        # Count tokens with weight > threshold per query
        support = (att > threshold).float().sum(dim=-1)  # [B, H, T]
        mean_support = support.mean(dim=(0, 2))  # [H]
        results[layer_idx] = {
            'mean': mean_support.numpy(),
            'per_query': support.mean(dim=0).numpy(),  # [H, T]
        }
    return results


def compute_first_token_attention(att_maps):
    """Compute how much attention goes to the first token (attention sink metric).

    Qwen paper: standard attention puts 46.7% on first token.
    """
    results = {}
    for layer_idx, att in enumerate(att_maps):
        # att[:, :, :, 0] = attention to first token from all queries
        first_token_att = att[:, :, :, 0]  # [B, H, T]
        mean_first = first_token_att.mean(dim=(0, 2))  # [H]
        results[layer_idx] = {
            'mean': mean_first.numpy(),
            'per_query': first_token_att.mean(dim=0).numpy(),  # [H, T]
        }
    return results


def analyze_model(model_path, device='cpu'):
    """Full analysis of a trained model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*60}")

    model, config = load_model(model_path, device)
    val_data, meta = load_data()

    # Get a batch of validation data
    block_size = config.block_size
    batch_size = 16
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(
        (val_data[i:i+block_size]).astype(np.int64)) for i in ix]).to(device)

    mode = "Standard" if not config.use_lif else f"LIF-{config.lif_mode}"
    print(f"Mode: {mode}")
    print(f"Layers: {config.n_layer}, Heads: {config.n_head}")

    with torch.no_grad():
        att_maps = get_attention_maps(model, x)

    # 1. Entropy analysis
    print(f"\n--- Attention Entropy (higher=uniform, lower=peaked) ---")
    entropy = compute_entropy(att_maps)
    for layer, data in entropy.items():
        vals = ' '.join(f'{v:.3f}' for v in data['mean'])
        print(f"  Layer {layer}: [{vals}]  avg={data['mean'].mean():.3f}")

    overall_entropy = np.mean([data['mean'].mean() for data in entropy.values()])
    print(f"  Overall mean entropy: {overall_entropy:.3f}")

    # 2. Support size
    print(f"\n--- Effective Support Size (tokens with >1% weight) ---")
    support = compute_support_size(att_maps, threshold=0.01)
    for layer, data in support.items():
        vals = ' '.join(f'{v:.1f}' for v in data['mean'])
        print(f"  Layer {layer}: [{vals}]  avg={data['mean'].mean():.1f}")

    overall_support = np.mean([data['mean'].mean() for data in support.values()])
    print(f"  Overall mean support: {overall_support:.1f}")

    # 3. First-token attention (attention sink)
    print(f"\n--- First-Token Attention % (attention sink) ---")
    first_tok = compute_first_token_attention(att_maps)
    for layer, data in first_tok.items():
        vals = ' '.join(f'{v:.1%}' for v in data['mean'])
        print(f"  Layer {layer}: [{vals}]  avg={data['mean'].mean():.1%}")

    overall_first = np.mean([data['mean'].mean() for data in first_tok.values()])
    print(f"  Overall mean first-token attention: {overall_first:.1%}")

    # 4. LIF parameters
    if config.use_lif and config.lif_mode != 'fixed':
        print(f"\n--- Learned LIF Parameters ---")
        lif_keywords = ['threshold', 'leak', 'steepness', 'refractory', 'cross_layer']
        for name, param in model.named_parameters():
            if any(k in name for k in lif_keywords):
                vals = [f'{v:.4f}' for v in param.tolist()]
                print(f"  {name}: [{', '.join(vals)}]")

    return {
        'mode': mode,
        'entropy': overall_entropy,
        'support': overall_support,
        'first_token': overall_first,
    }


def parse_ablation_log(log_path):
    """Parse the ablation study log file and print results table."""
    print(f"\n{'='*60}")
    print(f"Parsing ablation log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, 'r') as f:
        content = f.read()

    # Parse each condition's results
    conditions = {}
    current_condition = None

    for line in content.split('\n'):
        # Detect condition start
        match = re.match(r'>>> Training (\w[\w-]*)\.\.\.', line)
        if match:
            current_condition = match.group(1)
            conditions[current_condition] = []
            continue

        # Parse eval lines
        match = re.match(r'\[(\w[\w-]*)\]\s+iter\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)', line)
        if match:
            name = match.group(1)
            conditions.setdefault(name, [])
            conditions[name].append({
                'iter': int(match.group(2)),
                'train_loss': float(match.group(3)),
                'val_loss': float(match.group(4)),
            })

    if not conditions:
        print("No results found yet.")
        return

    # Print comparison table
    iters_set = set()
    for cond, data in conditions.items():
        for d in data:
            iters_set.add(d['iter'])

    iters_sorted = sorted(iters_set)
    cond_names = list(conditions.keys())

    # Header
    header = f"{'Iter':>6} |"
    for name in cond_names:
        header += f" {name:>16} |"
    print(header)
    print("-" * len(header))

    # Get standard val_loss for diff calculation
    std_losses = {}
    if 'Standard' in conditions:
        for d in conditions['Standard']:
            std_losses[d['iter']] = d['val_loss']

    for it in iters_sorted:
        row = f"{it:>6} |"
        for name in cond_names:
            found = [d for d in conditions[name] if d['iter'] == it]
            if found:
                val = found[0]['val_loss']
                if name != 'Standard' and it in std_losses:
                    diff = (val - std_losses[it]) / std_losses[it] * 100
                    row += f" {val:.4f} ({diff:+.2f}%) |"
                else:
                    row += f" {val:.4f}           |"
            else:
                row += f" {'---':>16} |"
        print(row)

    # Best results
    print(f"\n--- Best val_loss per condition ---")
    for name, data in conditions.items():
        if data:
            best = min(data, key=lambda d: d['val_loss'])
            print(f"  {name}: {best['val_loss']:.4f} (iter {best['iter']})")

    # Parse final summary if present
    if 'ABLATION RESULTS' in content:
        print(f"\n--- Final Summary (from log) ---")
        in_results = False
        for line in content.split('\n'):
            if 'ABLATION RESULTS' in line:
                in_results = True
                continue
            if in_results and line.strip().startswith(('Standard', 'LIF', 'Best:')):
                print(f"  {line.strip()}")


def compare_models():
    """Compare all saved models in the output directory."""
    print(f"\n{'='*60}")
    print("EMBER MODEL COMPARISON")
    print(f"{'='*60}")

    results = []
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith('.pt'):
            path = os.path.join(OUT_DIR, fname)
            try:
                r = analyze_model(path)
                r['file'] = fname
                results.append(r)
            except Exception as e:
                print(f"  Error analyzing {fname}: {e}")

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Entropy':>8} {'Support':>8} {'1st-tok%':>8}")
        print("-" * 55)
        for r in results:
            print(f"{r['mode']:<25} {r['entropy']:>8.3f} {r['support']:>8.1f} {r['first_token']:>8.1%}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Ember models')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--compare', action='store_true', help='Compare all saved models')
    parser.add_argument('--ablation-log', type=str, help='Parse ablation study log')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.ablation_log:
        parse_ablation_log(args.ablation_log)
    elif args.compare:
        compare_models()
    elif args.model:
        analyze_model(args.model, args.device)
    else:
        # Default: try to parse ablation log if it exists
        ablation_log = '/tmp/ember_ablation_v25.log'
        if os.path.exists(ablation_log):
            parse_ablation_log(ablation_log)
        else:
            compare_models()


if __name__ == '__main__':
    main()
