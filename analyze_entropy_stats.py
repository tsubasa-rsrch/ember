"""
Unified entropy analysis with cross-seed statistics for Ember Paper.

Extracts per-layer entropy from checkpoints, computes mean +/- std,
Cohen's d, and p-values across seeds. Outputs publication-ready tables
and JSON for plot_paper_figures.py.

Modalities:
  1. Transformer text (attention entropy) — single checkpoints (Standard + LIF)
  2. CfC text (neuron firing entropy) — 3 seeds (42, 668, 1337)
  3. CfC audio (neuron firing entropy) — 6 seeds (42, 668, 1337, 2024, 314, 777)

2026-03-02 — Tsubasa
"""

import os
import sys
import json
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMBER_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(EMBER_DIR, 'out')
DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
RESULTS_DIR = os.path.join(EMBER_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Transformer text — attention entropy
# ---------------------------------------------------------------------------

def load_transformer_model(path, device='cpu'):
    """Load Ember Transformer checkpoint."""
    from model import Ember, EmberConfig
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = Ember(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config, ckpt


def get_transformer_attention_maps(model, x):
    """Extract attention probability maps from all layers (from analyze.py)."""
    device = x.device
    b, t = x.size()

    tok_emb = model.transformer.wte(x)
    pos_emb = model.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device))
    hidden = model.transformer.drop(tok_emb + pos_emb)

    attention_maps = []
    refractory_state = None
    head_state = None

    if model.config.use_head_persistent:
        head_state = torch.zeros(b, model.config.n_head, device=device)

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
            att_probs, new_refractory, new_head_state = attn.lif_activation(
                att, refractory_state, head_state)
            refractory_state = new_refractory
            head_state = new_head_state
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
            att = att.masked_fill(causal == 0, float('-inf'))
            att_probs = F.softmax(att, dim=-1)
            refractory_state = None

        attention_maps.append(att_probs.detach())

        # Forward through block for hidden state
        membrane_potential = None
        if block.use_temporal_lif:
            membrane_potential = torch.zeros(b, T, device=device)
        hidden, refractory_state, membrane_potential, head_state = block(
            hidden,
            refractory_state if block.use_lif else None,
            membrane_potential,
            head_state)

    return attention_maps


def compute_attention_entropy(att_maps):
    """Compute attention entropy per layer (averaged over heads, batch, positions).

    Returns per-layer mean entropy and per-head entropy.
    """
    results = {}
    for layer_idx, att in enumerate(att_maps):
        # att: [B, H, T, T]
        att_clamped = att.clamp(min=1e-10)
        entropy = -(att_clamped * att_clamped.log()).sum(dim=-1)  # [B, H, T]
        # Per-head mean (average over batch and query positions)
        per_head = entropy.mean(dim=(0, 2))  # [H]
        results[layer_idx] = {
            'per_head': per_head.numpy(),
            'layer_mean': per_head.mean().item(),
            'layer_std': per_head.std().item(),
        }
    return results


def analyze_transformer():
    """Analyze Transformer text models."""
    print("\n" + "=" * 70)
    print("1. TRANSFORMER TEXT — ATTENTION ENTROPY")
    print("=" * 70)

    # Load validation data
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

    # Prepare consistent input batch
    block_size = 256
    batch_size = 32
    torch.manual_seed(42)  # Fixed seed for reproducible sampling
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(
        (val_data[i:i+block_size]).astype(np.int64)) for i in ix])

    # Available checkpoints
    conditions = {
        'Standard': os.path.join(OUT_DIR, 'ember_standard.pt'),
        'LIF': os.path.join(OUT_DIR, 'ember_lif.pt'),
        'LIF-fixed': os.path.join(OUT_DIR, 'ember_lif_fixed.pt'),
        'LIF-refrac': os.path.join(OUT_DIR, 'ember_lif_refrac.pt'),
        'Refrac+Head': os.path.join(OUT_DIR, 'ember_refrac+head.pt'),
    }

    all_results = {}
    for cond_name, path in conditions.items():
        if not os.path.exists(path):
            print(f"  Skipping {cond_name}: checkpoint not found")
            continue

        model, config, ckpt = load_transformer_model(path)
        raw_val_loss = ckpt.get('best_val_loss', 'N/A')
        # Convert tensor to float if needed
        if hasattr(raw_val_loss, 'item'):
            val_loss = raw_val_loss.item()
        elif isinstance(raw_val_loss, (int, float)):
            val_loss = float(raw_val_loss)
        else:
            val_loss = raw_val_loss

        with torch.no_grad():
            att_maps = get_transformer_attention_maps(model, x)

        entropy = compute_attention_entropy(att_maps)
        all_results[cond_name] = {
            'entropy': entropy,
            'val_loss': val_loss,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
        }

        val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        print(f"\n  {cond_name} (val_loss={val_str}):")
        for layer_idx in sorted(entropy.keys()):
            d = entropy[layer_idx]
            head_str = ' '.join(f'{v:.3f}' for v in d['per_head'])
            print(f"    Layer {layer_idx}: [{head_str}]  mean={d['layer_mean']:.3f}")

    # Comparison table
    if 'Standard' in all_results and 'LIF' in all_results:
        print(f"\n  --- Depth Hierarchy Comparison ---")
        print(f"  {'Layer':>6} {'Standard':>10} {'LIF':>10} {'Delta':>10} {'Delta%':>10}")
        print(f"  {'-'*50}")

        std_entropies = []
        lif_entropies = []
        n_layer = all_results['Standard']['n_layer']
        for l in range(n_layer):
            se = all_results['Standard']['entropy'][l]['layer_mean']
            le = all_results['LIF']['entropy'][l]['layer_mean']
            std_entropies.append(se)
            lif_entropies.append(le)
            delta = le - se
            pct = delta / se * 100 if se != 0 else 0
            print(f"  L{l:>4d} {se:>10.4f} {le:>10.4f} {delta:>+10.4f} {pct:>+9.2f}%")

        # Range
        std_range = max(std_entropies) - min(std_entropies)
        lif_range = max(lif_entropies) - min(lif_entropies)
        print(f"\n  Entropy range (depth spread):")
        print(f"    Standard: {std_range:.4f} (L0={std_entropies[0]:.3f} → L{n_layer-1}={std_entropies[-1]:.3f})")
        print(f"    LIF:      {lif_range:.4f} (L0={lif_entropies[0]:.3f} → L{n_layer-1}={lif_entropies[-1]:.3f})")
        print(f"    LIF creates {lif_range/std_range:.1f}x wider hierarchy" if std_range > 0 else "")

    return all_results


# ---------------------------------------------------------------------------
# 2. CfC text — neuron firing entropy
# ---------------------------------------------------------------------------

def load_liquid_model(path, device='cpu'):
    """Load Liquid Ember checkpoint."""
    from liquid_ember import LiquidEmber, LiquidEmberConfig, LIFGate
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = LiquidEmber(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config, ckpt


def extract_cfc_lif_activations(model, x):
    """Extract LIF gate fire_mask per layer (from analyze_liquid.py)."""
    from liquid_ember import LIFGate
    activations = []
    hooks = []

    for block in model.blocks:
        layer_data = {}
        if isinstance(block.lif_gate, LIFGate):
            def make_hook(ld, lif_gate):
                def hook_fn(module, input, output):
                    x_in = input[0]
                    threshold = torch.abs(lif_gate.threshold) * 0.1
                    steepness = F.softplus(lif_gate.steepness)
                    potential = torch.abs(x_in)
                    fire_mask = torch.sigmoid(steepness * (potential - threshold))
                    ld['fire_mask'] = fire_mask.detach().cpu()
                return hook_fn
            h = block.lif_gate.register_forward_hook(make_hook(layer_data, block.lif_gate))
        else:
            def make_identity_hook(ld):
                def hook_fn(module, input, output):
                    x_in = input[0]
                    ld['fire_mask'] = torch.ones_like(x_in).cpu()
                return hook_fn
            h = block.lif_gate.register_forward_hook(make_identity_hook(layer_data))
        hooks.append(h)
        activations.append(layer_data)

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
    return activations


def compute_neuron_entropy(activations, n_bins=10):
    """Compute neuron firing entropy per layer (from analyze_liquid.py).

    For each neuron, discretize fire_mask into bins and compute entropy.
    Low entropy = specialist, High entropy = unpredictable.
    """
    results = {}
    for layer_idx, data in enumerate(activations):
        fire_mask = data['fire_mask']  # [B, T, D]
        B, T, D = fire_mask.shape
        flat = fire_mask.reshape(-1, D)

        entropies = []
        for d in range(D):
            vals = flat[:, d].numpy()
            hist, _ = np.histogram(vals, bins=n_bins, range=(0, 1), density=True)
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            ent = -(hist * np.log(hist + 1e-10)).sum()
            entropies.append(ent)

        entropies = np.array(entropies)
        fire_rate = (fire_mask > 0.5).float().mean(dim=(0, 1)).numpy()

        results[layer_idx] = {
            'entropy_mean': float(entropies.mean()),
            'entropy_std': float(entropies.std()),
            'fire_rate_mean': float(fire_rate.mean()),
            'fire_rate_std': float(fire_rate.std()),
        }
    return results


def analyze_cfc_text():
    """Analyze CfC text (Liquid Ember) with cross-seed statistics."""
    print("\n" + "=" * 70)
    print("2. CfC TEXT (LIQUID EMBER) — NEURON FIRING ENTROPY")
    print("=" * 70)

    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    seeds = [42, 668, 1337]
    conditions = ['base', 'lif']
    n_batches = 10
    batch_size = 32
    block_size = 256

    all_seed_results = {}

    for seed in seeds:
        for cond in conditions:
            tag = f"liquid_scaled_{cond}_s{seed}"
            path = os.path.join(OUT_DIR, f'{tag}.pt')
            if not os.path.exists(path):
                print(f"  Skipping {tag}: not found")
                continue

            model, config, ckpt = load_liquid_model(path)
            n_layer = config.n_layer
            n_embd = config.n_embd

            # Accumulate activations over multiple batches
            torch.manual_seed(seed + 1000)  # Different from training seed
            all_acts = None
            for batch_idx in range(n_batches):
                ix = torch.randint(len(val_data) - block_size, (batch_size,))
                x = torch.stack([torch.from_numpy(
                    (val_data[i:i+block_size]).astype(np.int64)) for i in ix])
                acts = extract_cfc_lif_activations(model, x)
                if all_acts is None:
                    all_acts = acts
                else:
                    for li in range(len(acts)):
                        for k in acts[li]:
                            all_acts[li][k] = torch.cat([all_acts[li][k], acts[li][k]], dim=0)

            metrics = compute_neuron_entropy(all_acts)
            key = f"{cond}_s{seed}"
            all_seed_results[key] = metrics

            # Print per-seed results
            ent_str = " | ".join([f"L{l}={metrics[l]['entropy_mean']:.4f}" for l in range(n_layer)])
            fr_str = " | ".join([f"L{l}={metrics[l]['fire_rate_mean']:.4f}" for l in range(n_layer)])
            print(f"  {key:20s} entropy: {ent_str}")
            print(f"  {'':20s} fire_rt: {fr_str}")

    # Cross-seed statistics
    n_layer = 4  # CfC text has 4 layers
    cross_seed = {}
    for cond in conditions:
        per_layer_ent = {l: [] for l in range(n_layer)}
        per_layer_fr = {l: [] for l in range(n_layer)}

        for seed in seeds:
            key = f"{cond}_s{seed}"
            if key not in all_seed_results:
                continue
            for l in range(n_layer):
                per_layer_ent[l].append(all_seed_results[key][l]['entropy_mean'])
                per_layer_fr[l].append(all_seed_results[key][l]['fire_rate_mean'])

        if not per_layer_ent[0]:
            continue

        cond_stats = {}
        for l in range(n_layer):
            ent_arr = np.array(per_layer_ent[l])
            fr_arr = np.array(per_layer_fr[l])
            cond_stats[l] = {
                'entropy_mean': float(ent_arr.mean()),
                'entropy_std': float(ent_arr.std()),
                'fire_rate_mean': float(fr_arr.mean()),
                'fire_rate_std': float(fr_arr.std()),
                'n_seeds': len(ent_arr),
            }
        cross_seed[cond] = cond_stats

    # Print cross-seed summary
    print(f"\n  --- Cross-Seed Statistics (N={len(seeds)} seeds) ---")
    print(f"  {'Layer':>6} {'Base ent':>12} {'LIF ent':>12} {'Cohen d':>10} {'p-value':>10}")
    print(f"  {'-'*56}")

    for l in range(n_layer):
        base_ents = [all_seed_results[f'base_s{s}'][l]['entropy_mean'] for s in seeds
                     if f'base_s{s}' in all_seed_results]
        lif_ents = [all_seed_results[f'lif_s{s}'][l]['entropy_mean'] for s in seeds
                    if f'lif_s{s}' in all_seed_results]

        if len(base_ents) >= 2 and len(lif_ents) >= 2:
            base_m, base_s = np.mean(base_ents), np.std(base_ents)
            lif_m, lif_s = np.mean(lif_ents), np.std(lif_ents)

            # Cohen's d
            pooled_std = np.sqrt((base_s**2 + lif_s**2) / 2) if (base_s + lif_s) > 0 else 1e-10
            cohens_d = (lif_m - base_m) / pooled_std if pooled_std > 0 else float('inf')

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(lif_ents, base_ents)

            print(f"  L{l:>4d} {base_m:>8.4f}±{base_s:.4f} {lif_m:>8.4f}±{lif_s:.4f} "
                  f"{cohens_d:>+10.3f} {p_val:>10.4f}")
        else:
            base_m = np.mean(base_ents) if base_ents else 0
            lif_m = np.mean(lif_ents) if lif_ents else 0
            print(f"  L{l:>4d} {base_m:>12.4f} {lif_m:>12.4f}     N/A        N/A")

    # Verify base fire rate = 1.0 (critical claim)
    print(f"\n  --- Base Model Fire Rate Verification ---")
    print(f"  (Claim: base model fire rate should be 1.0 → entropy 0)")
    for seed in seeds:
        key = f"base_s{seed}"
        if key in all_seed_results:
            frs = [all_seed_results[key][l]['fire_rate_mean'] for l in range(n_layer)]
            ents = [all_seed_results[key][l]['entropy_mean'] for l in range(n_layer)]
            fr_str = " ".join(f"L{l}={frs[l]:.4f}" for l in range(n_layer))
            ent_str = " ".join(f"L{l}={ents[l]:.4f}" for l in range(n_layer))
            print(f"  seed {seed}: fire_rate=[{fr_str}]")
            print(f"  {'':>10}  entropy =[{ent_str}]")

    return all_seed_results, cross_seed


# ---------------------------------------------------------------------------
# 3. CfC audio — neuron firing entropy
# ---------------------------------------------------------------------------

def analyze_cfc_audio():
    """Analyze Audio CfC with cross-seed statistics."""
    print("\n" + "=" * 70)
    print("3. CfC AUDIO (AUDIO LIQUID EMBER) — NEURON FIRING ENTROPY")
    print("=" * 70)

    from audio_liquid_ember import AudioLiquidEmber, AudioLiquidEmberConfig
    from liquid_ember import LIFGate
    from train_audio import SpeechCommandsDataset, LABELS

    seeds = [42, 668, 1337, 2024, 314, 777]
    conditions = ['base', 'lif']
    n_batches = 10
    batch_size = 32
    data_dir = os.path.join(EMBER_DIR, 'data')

    all_seed_results = {}
    val_ds_cache = {}

    for seed in seeds:
        for cond in conditions:
            tag = f"audio_{cond}_s{seed}"
            path = os.path.join(OUT_DIR, f'{tag}.pt')
            if not os.path.exists(path):
                print(f"  Skipping {tag}: not found")
                continue

            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            config = ckpt['config']
            model = AudioLiquidEmber(config)
            model.load_state_dict(ckpt['model'])
            model.eval()
            val_acc = ckpt.get('val_acc', 'N/A')

            # Cache validation dataset (same across seeds)
            cache_key = f"{cond}"
            if cache_key not in val_ds_cache:
                val_ds_cache[cache_key] = SpeechCommandsDataset(data_dir, 'validation', config)
            val_ds = val_ds_cache[cache_key]

            # Extract activations
            activations = []
            hooks = []
            for block in model.blocks:
                layer_data = {}
                if isinstance(block.lif_gate, LIFGate):
                    def make_hook(ld, lif_gate):
                        def hook_fn(module, input, output):
                            x_in = input[0]
                            threshold = torch.abs(lif_gate.threshold) * 0.1
                            steepness = F.softplus(lif_gate.steepness)
                            potential = torch.abs(x_in)
                            fire_mask = torch.sigmoid(steepness * (potential - threshold))
                            ld['fire_mask'] = fire_mask.detach().cpu()
                        return hook_fn
                    h = block.lif_gate.register_forward_hook(make_hook(layer_data, block.lif_gate))
                else:
                    def make_identity_hook(ld):
                        def hook_fn(module, input, output):
                            x_in = input[0]
                            ld['fire_mask'] = torch.ones_like(x_in).cpu()
                        return hook_fn
                    h = block.lif_gate.register_forward_hook(make_identity_hook(layer_data))
                hooks.append(h)
                activations.append(layer_data)

            all_acts = [ld for ld in activations]
            first_run = True
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                if start >= len(val_ds):
                    break
                end = min(start + batch_size, len(val_ds))
                mels = torch.stack([val_ds[i][0] for i in range(start, end)])

                # Reset activations list for each batch
                for ld in activations:
                    ld.clear()

                with torch.no_grad():
                    model(mels)

                if first_run:
                    # Initialize accumulated activations
                    all_acts = [{k: v.clone() for k, v in ld.items()} for ld in activations]
                    first_run = False
                else:
                    for li in range(len(activations)):
                        for k in activations[li]:
                            all_acts[li][k] = torch.cat([all_acts[li][k], activations[li][k]], dim=0)

            for h in hooks:
                h.remove()

            n_layer = len(all_acts)
            metrics = compute_neuron_entropy(all_acts)
            all_seed_results[f"{cond}_s{seed}"] = {
                'metrics': metrics,
                'val_acc': float(val_acc) if isinstance(val_acc, (int, float)) else val_acc,
            }

            ent_str = " | ".join([f"L{l}={metrics[l]['entropy_mean']:.4f}" for l in range(n_layer)])
            fr_str = " | ".join([f"L{l}={metrics[l]['fire_rate_mean']:.4f}" for l in range(n_layer)])
            print(f"  {tag:20s} [acc={val_acc:.4f}] entropy: {ent_str}")
            print(f"  {'':20s}                fire_rt: {fr_str}")

    # Cross-seed statistics
    n_layer = 4
    cross_seed = {}

    print(f"\n  --- Cross-Seed Statistics (N={len(seeds)} seeds) ---")
    print(f"  {'Layer':>6} {'Base ent':>12} {'LIF ent':>12} {'Cohen d':>10} {'p-value':>10}")
    print(f"  {'-'*56}")

    for cond in conditions:
        per_layer_ent = {l: [] for l in range(n_layer)}
        per_layer_fr = {l: [] for l in range(n_layer)}
        accs = []

        for seed in seeds:
            key = f"{cond}_s{seed}"
            if key not in all_seed_results:
                continue
            metrics = all_seed_results[key]['metrics']
            for l in range(n_layer):
                per_layer_ent[l].append(metrics[l]['entropy_mean'])
                per_layer_fr[l].append(metrics[l]['fire_rate_mean'])
            accs.append(all_seed_results[key]['val_acc'])

        cond_stats = {}
        for l in range(n_layer):
            ent_arr = np.array(per_layer_ent[l])
            fr_arr = np.array(per_layer_fr[l])
            cond_stats[l] = {
                'entropy_mean': float(ent_arr.mean()),
                'entropy_std': float(ent_arr.std()),
                'fire_rate_mean': float(fr_arr.mean()),
                'fire_rate_std': float(fr_arr.std()),
                'n_seeds': len(ent_arr),
            }
        if accs:
            cond_stats['val_acc_mean'] = float(np.mean(accs))
            cond_stats['val_acc_std'] = float(np.std(accs))
        cross_seed[cond] = cond_stats

    for l in range(n_layer):
        base_ents = [all_seed_results[f'base_s{s}']['metrics'][l]['entropy_mean'] for s in seeds
                     if f'base_s{s}' in all_seed_results]
        lif_ents = [all_seed_results[f'lif_s{s}']['metrics'][l]['entropy_mean'] for s in seeds
                    if f'lif_s{s}' in all_seed_results]

        if len(base_ents) >= 2 and len(lif_ents) >= 2:
            base_m, base_s = np.mean(base_ents), np.std(base_ents)
            lif_m, lif_s = np.mean(lif_ents), np.std(lif_ents)

            pooled_std = np.sqrt((base_s**2 + lif_s**2) / 2) if (base_s + lif_s) > 0 else 1e-10
            cohens_d = (lif_m - base_m) / pooled_std if pooled_std > 0 else float('inf')

            t_stat, p_val = stats.ttest_rel(lif_ents, base_ents)

            print(f"  L{l:>4d} {base_m:>8.4f}±{base_s:.4f} {lif_m:>8.4f}±{lif_s:.4f} "
                  f"{cohens_d:>+10.3f} {p_val:>10.4f}")

    # Accuracy comparison
    if 'base' in cross_seed and 'lif' in cross_seed:
        base_acc = cross_seed['base'].get('val_acc_mean', 0)
        base_std = cross_seed['base'].get('val_acc_std', 0)
        lif_acc = cross_seed['lif'].get('val_acc_mean', 0)
        lif_std = cross_seed['lif'].get('val_acc_std', 0)
        print(f"\n  Accuracy: Base={base_acc:.4f}±{base_std:.4f}  LIF={lif_acc:.4f}±{lif_std:.4f}")

    # Base fire rate verification
    print(f"\n  --- Base Model Fire Rate Verification ---")
    for seed in seeds:
        key = f"base_s{seed}"
        if key in all_seed_results:
            metrics = all_seed_results[key]['metrics']
            frs = [metrics[l]['fire_rate_mean'] for l in range(n_layer)]
            print(f"  seed {seed}: fire_rate=[" + " ".join(f"L{l}={frs[l]:.4f}" for l in range(n_layer)) + "]")

    return all_seed_results, cross_seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EMBER UNIFIED ENTROPY ANALYSIS — Cross-Seed Statistics")
    print("=" * 70)

    # 1. Transformer
    transformer_results = analyze_transformer()

    # 2. CfC text
    cfc_text_seeds, cfc_text_cross = analyze_cfc_text()

    # 3. CfC audio
    cfc_audio_seeds, cfc_audio_cross = analyze_cfc_audio()

    # ---------------------------------------------------------------------------
    # Summary for paper
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PUBLICATION-READY SUMMARY")
    print("=" * 70)

    # Transformer summary
    print("\n--- Transformer (Attention Entropy) ---")
    if 'Standard' in transformer_results and 'LIF' in transformer_results:
        n_layer = transformer_results['Standard']['n_layer']
        print(f"  {'Layer':>6} {'Standard':>12} {'LIF':>12} {'Delta':>10}")
        print(f"  {'-'*42}")
        std_ents = []
        lif_ents = []
        for l in range(n_layer):
            se = transformer_results['Standard']['entropy'][l]['layer_mean']
            le = transformer_results['LIF']['entropy'][l]['layer_mean']
            std_ents.append(se)
            lif_ents.append(le)
            print(f"  L{l:>4d} {se:>12.4f} {le:>12.4f} {le-se:>+10.4f}")

        std_range = max(std_ents) - min(std_ents)
        lif_range = max(lif_ents) - min(lif_ents)
        print(f"\n  Hierarchy ratio: LIF range / Std range = {lif_range:.4f} / {std_range:.4f} = {lif_range/std_range:.2f}x")
        print(f"  (Note: single checkpoint per condition, no cross-seed stats for Transformer)")

    # CfC text summary
    print("\n--- CfC Text (Neuron Firing Entropy, N=3 seeds) ---")
    if 'base' in cfc_text_cross and 'lif' in cfc_text_cross:
        n_layer = max(max(cfc_text_cross['base'].keys()), max(cfc_text_cross['lif'].keys())) + 1
        print(f"  {'Layer':>6} {'Base':>16} {'LIF':>16}")
        print(f"  {'-'*42}")
        for l in range(n_layer):
            if isinstance(l, int) and l in cfc_text_cross['base'] and l in cfc_text_cross['lif']:
                bm = cfc_text_cross['base'][l]['entropy_mean']
                bs = cfc_text_cross['base'][l]['entropy_std']
                lm = cfc_text_cross['lif'][l]['entropy_mean']
                ls = cfc_text_cross['lif'][l]['entropy_std']
                print(f"  L{l:>4d} {bm:>8.4f}±{bs:.4f} {lm:>8.4f}±{ls:.4f}")

    # CfC audio summary
    print("\n--- CfC Audio (Neuron Firing Entropy, N=6 seeds) ---")
    if 'base' in cfc_audio_cross and 'lif' in cfc_audio_cross:
        n_layer = max(k for k in cfc_audio_cross['lif'].keys() if isinstance(k, int)) + 1
        print(f"  {'Layer':>6} {'Base':>16} {'LIF':>16}")
        print(f"  {'-'*42}")
        for l in range(n_layer):
            if l in cfc_audio_cross['base'] and l in cfc_audio_cross['lif']:
                bm = cfc_audio_cross['base'][l]['entropy_mean']
                bs = cfc_audio_cross['base'][l]['entropy_std']
                lm = cfc_audio_cross['lif'][l]['entropy_mean']
                ls = cfc_audio_cross['lif'][l]['entropy_std']
                print(f"  L{l:>4d} {bm:>8.4f}±{bs:.4f} {lm:>8.4f}±{ls:.4f}")

    # ---------------------------------------------------------------------------
    # Figure data (to replace hardcoded values in plot_paper_figures.py)
    # ---------------------------------------------------------------------------
    print("\n--- Updated Figure Data (for plot_paper_figures.py) ---")

    if 'Standard' in transformer_results and 'LIF' in transformer_results:
        n_layer = transformer_results['Standard']['n_layer']
        std_vals = [transformer_results['Standard']['entropy'][l]['layer_mean'] for l in range(n_layer)]
        lif_vals = [transformer_results['LIF']['entropy'][l]['layer_mean'] for l in range(n_layer)]
        print(f"\n  # Transformer (fig2 left)")
        print(f"  std_entropy_t = {[round(v, 4) for v in std_vals]}")
        print(f"  lif_entropy_t = {[round(v, 4) for v in lif_vals]}")

    if 'base' in cfc_text_cross and 'lif' in cfc_text_cross:
        n_layer_c = max(k for k in cfc_text_cross['lif'].keys() if isinstance(k, int)) + 1
        base_vals = [cfc_text_cross['base'][l]['entropy_mean'] for l in range(n_layer_c)]
        lif_vals = [cfc_text_cross['lif'][l]['entropy_mean'] for l in range(n_layer_c)]
        base_stds = [cfc_text_cross['base'][l]['entropy_std'] for l in range(n_layer_c)]
        lif_stds = [cfc_text_cross['lif'][l]['entropy_std'] for l in range(n_layer_c)]
        print(f"\n  # CfC Text (fig2 right)")
        print(f"  base_entropy_c = {[round(v, 4) for v in base_vals]}")
        print(f"  lif_entropy_c  = {[round(v, 4) for v in lif_vals]}")
        print(f"  base_entropy_c_std = {[round(v, 4) for v in base_stds]}")
        print(f"  lif_entropy_c_std  = {[round(v, 4) for v in lif_stds]}")

    if 'base' in cfc_audio_cross and 'lif' in cfc_audio_cross:
        n_layer_a = max(k for k in cfc_audio_cross['lif'].keys() if isinstance(k, int)) + 1
        base_vals_a = [cfc_audio_cross['base'][l]['entropy_mean'] for l in range(n_layer_a)]
        lif_vals_a = [cfc_audio_cross['lif'][l]['entropy_mean'] for l in range(n_layer_a)]
        base_stds_a = [cfc_audio_cross['base'][l]['entropy_std'] for l in range(n_layer_a)]
        lif_stds_a = [cfc_audio_cross['lif'][l]['entropy_std'] for l in range(n_layer_a)]
        print(f"\n  # CfC Audio")
        print(f"  base_entropy_audio = {[round(v, 4) for v in base_vals_a]}")
        print(f"  lif_entropy_audio  = {[round(v, 4) for v in lif_vals_a]}")
        print(f"  base_entropy_audio_std = {[round(v, 4) for v in base_stds_a]}")
        print(f"  lif_entropy_audio_std  = {[round(v, 4) for v in lif_stds_a]}")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    output = {
        'transformer': {},
        'cfc_text': {},
        'cfc_audio': {},
    }

    # Transformer
    for cond_name, data in transformer_results.items():
        n_layer = data['n_layer']
        output['transformer'][cond_name] = {
            'val_loss': data['val_loss'],
            'per_layer_entropy': [data['entropy'][l]['layer_mean'] for l in range(n_layer)],
            'per_layer_per_head': [data['entropy'][l]['per_head'].tolist() for l in range(n_layer)],
        }

    # CfC text
    for cond in ['base', 'lif']:
        if cond in cfc_text_cross:
            n_layer = max(k for k in cfc_text_cross[cond].keys() if isinstance(k, int)) + 1
            output['cfc_text'][cond] = {
                'per_layer': {
                    str(l): cfc_text_cross[cond][l] for l in range(n_layer)
                },
                'seeds_used': [s for s in [42, 668, 1337] if f'{cond}_s{s}' in cfc_text_seeds],
            }

    # CfC audio
    for cond in ['base', 'lif']:
        if cond in cfc_audio_cross:
            n_layer = max(k for k in cfc_audio_cross[cond].keys() if isinstance(k, int)) + 1
            per_layer = {}
            for l in range(n_layer):
                if l in cfc_audio_cross[cond]:
                    per_layer[str(l)] = cfc_audio_cross[cond][l]
            entry = {
                'per_layer': per_layer,
                'seeds_used': [s for s in [42, 668, 1337, 2024, 314, 777]
                               if f'{cond}_s{s}' in cfc_audio_seeds],
            }
            if 'val_acc_mean' in cfc_audio_cross[cond]:
                entry['val_acc_mean'] = cfc_audio_cross[cond]['val_acc_mean']
                entry['val_acc_std'] = cfc_audio_cross[cond]['val_acc_std']
            output['cfc_audio'][cond] = entry

    json_path = os.path.join(RESULTS_DIR, 'entropy_cross_seed_stats.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
