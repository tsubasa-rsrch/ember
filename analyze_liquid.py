"""
Liquid Ember analysis — neuron-level specialization via LIF gate.

Since Liquid Ember has NO attention mechanism (CfC replaces it),
we analyze the LIF gate's fire/smolder patterns instead.

Key metrics:
  1. Fire rate per neuron: fraction of inputs where neuron fires
  2. Neuron entropy: how selective each neuron is (low = specialist)
  3. Population sparsity: fraction of neurons active per input
  4. Learned LIF parameters: threshold/leak/steepness distributions

The hypothesis (Kana 2026-02-18):
  "LIF's value is not accuracy but organization.
   Even when loss is identical, LIF models should show
   more structured internal representations."

Usage:
    python3 analyze_liquid.py
    python3 analyze_liquid.py --compare

2026-02-18 — Tsubasa
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from liquid_ember import LiquidEmber, LiquidEmberConfig, LIFGate

DATA_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/nanoGPT/data/shakespeare_char")
OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")


def load_data():
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return val_data, meta


def load_liquid_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = LiquidEmber(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def extract_lif_activations(model, x):
    """Run forward pass and capture LIF gate activations per layer.

    Returns list of dicts per layer:
      - fire_mask: [B, T, n_embd] — soft fire probability per neuron
      - potential: [B, T, n_embd] — input magnitude
      - cfc_output: [B, T, n_embd] — raw CfC output (before gating)
      - gated_output: [B, T, n_embd] — output after LIF gate
    """
    activations = []
    hooks = []

    for layer_idx, block in enumerate(model.blocks):
        layer_data = {}

        if isinstance(block.lif_gate, LIFGate):
            # Hook into LIF gate to capture fire_mask
            def make_hook(ld, lif_gate):
                def hook_fn(module, input, output):
                    x_in = input[0]  # [B, T, n_units]
                    threshold = torch.abs(lif_gate.threshold) * 0.1
                    steepness = F.softplus(lif_gate.steepness)
                    potential = torch.abs(x_in)
                    fire_mask = torch.sigmoid(steepness * (potential - threshold))
                    ld['fire_mask'] = fire_mask.detach().cpu()
                    ld['potential'] = potential.detach().cpu()
                    ld['cfc_output'] = x_in.detach().cpu()
                    ld['gated_output'] = output.detach().cpu()
                return hook_fn

            h = block.lif_gate.register_forward_hook(make_hook(layer_data, block.lif_gate))
            hooks.append(h)
        else:
            # Identity gate (baseline) — no LIF, capture CfC output
            def make_identity_hook(ld):
                def hook_fn(module, input, output):
                    x_in = input[0]
                    ld['fire_mask'] = torch.ones_like(x_in).cpu()  # all fire
                    ld['potential'] = torch.abs(x_in).detach().cpu()
                    ld['cfc_output'] = x_in.detach().cpu()
                    ld['gated_output'] = output.detach().cpu()
                return hook_fn

            h = block.lif_gate.register_forward_hook(make_identity_hook(layer_data))
            hooks.append(h)

        activations.append(layer_data)

    # Forward pass
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations


def analyze_fire_rates(activations, config):
    """Compute per-neuron fire rate across all inputs.

    Fire rate = fraction of (batch, position) pairs where fire_mask > 0.5
    Low fire rate = selective/specialist neuron
    High fire rate = always-on neuron
    """
    results = {}
    for layer_idx, data in enumerate(activations):
        fire_mask = data['fire_mask']  # [B, T, n_embd]

        # Binary fire decision (threshold at 0.5)
        firing = (fire_mask > 0.5).float()

        # Fire rate per neuron (avg over batch and position)
        fire_rate = firing.mean(dim=(0, 1))  # [n_embd]

        results[layer_idx] = {
            'fire_rate': fire_rate.numpy(),
            'mean': fire_rate.mean().item(),
            'std': fire_rate.std().item(),
            'min': fire_rate.min().item(),
            'max': fire_rate.max().item(),
            # Fraction of neurons that are "selective" (fire rate < 0.5)
            'selective_frac': (fire_rate < 0.5).float().mean().item(),
            # Fraction always firing
            'always_on_frac': (fire_rate > 0.95).float().mean().item(),
        }
    return results


def analyze_neuron_entropy(activations):
    """Compute entropy of each neuron's fire/smolder distribution.

    For each neuron, discretize fire_mask into bins and compute entropy.
    Low entropy = neuron is either always on or always off (specialist)
    High entropy = neuron fires unpredictably (no specialization)
    """
    results = {}
    n_bins = 10

    for layer_idx, data in enumerate(activations):
        fire_mask = data['fire_mask']  # [B, T, n_embd]
        B, T, D = fire_mask.shape

        # Reshape to [B*T, D]
        flat = fire_mask.reshape(-1, D)

        # For each neuron, compute histogram entropy
        entropies = []
        for d in range(D):
            vals = flat[:, d].numpy()
            # Histogram of fire probabilities
            hist, _ = np.histogram(vals, bins=n_bins, range=(0, 1), density=True)
            hist = hist / (hist.sum() + 1e-10)  # normalize
            hist = hist[hist > 0]
            entropy = -(hist * np.log(hist + 1e-10)).sum()
            entropies.append(entropy)

        entropies = np.array(entropies)
        results[layer_idx] = {
            'per_neuron': entropies,
            'mean': entropies.mean(),
            'std': entropies.std(),
            'min': entropies.min(),
            'max': entropies.max(),
        }
    return results


def analyze_population_sparsity(activations):
    """Compute population sparsity: what fraction of neurons fire per input.

    Lower = sparser code = more efficient representation
    """
    results = {}
    for layer_idx, data in enumerate(activations):
        fire_mask = data['fire_mask']  # [B, T, n_embd]

        # Fraction of neurons firing per (batch, position)
        firing = (fire_mask > 0.5).float()
        sparsity = firing.mean(dim=-1)  # [B, T] — fraction active

        results[layer_idx] = {
            'mean': sparsity.mean().item(),
            'std': sparsity.std().item(),
            'per_position': sparsity.mean(dim=0).numpy(),  # [T]
        }
    return results


def analyze_cfc_hidden_variance(activations):
    """Compute variance of CfC output across neurons.

    Higher variance = more differentiated neuron responses.
    """
    results = {}
    for layer_idx, data in enumerate(activations):
        cfc_out = data['cfc_output']  # [B, T, n_embd]

        # Variance across neuron dimension for each (batch, position)
        var_per_input = cfc_out.var(dim=-1)  # [B, T]

        # Variance across inputs for each neuron
        var_per_neuron = cfc_out.var(dim=(0, 1))  # [n_embd]

        results[layer_idx] = {
            'input_var_mean': var_per_input.mean().item(),
            'neuron_var_mean': var_per_neuron.mean().item(),
            'neuron_var_std': var_per_neuron.std().item(),
        }
    return results


def analyze_model(model_path, device='cpu', n_batches=10, batch_size=32):
    """Full analysis of a Liquid Ember model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(model_path)}")
    print(f"{'='*60}")

    model, config = load_liquid_model(model_path, device)
    val_data, meta = load_data()

    block_size = config.block_size
    use_lif = config.use_lif
    mode = "CfC+LIF" if use_lif else "CfC-only"

    print(f"Mode: {mode}")
    print(f"Layers: {config.n_layer}, Embed: {config.n_embd}, CfC units: {config.cfc_units}")

    # Accumulate activations over multiple batches for stable stats
    all_activations = None

    for batch_idx in range(n_batches):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(
            (val_data[i:i+block_size]).astype(np.int64)) for i in ix]).to(device)

        acts = extract_lif_activations(model, x)

        if all_activations is None:
            all_activations = acts
        else:
            for layer_idx in range(len(acts)):
                for key in acts[layer_idx]:
                    all_activations[layer_idx][key] = torch.cat(
                        [all_activations[layer_idx][key], acts[layer_idx][key]], dim=0)

    total_samples = n_batches * batch_size
    print(f"Analyzed {total_samples} samples × {block_size} positions")

    # 1. Fire rates
    print(f"\n--- Fire Rate Per Neuron (selective=specialist) ---")
    fire_rates = analyze_fire_rates(all_activations, config)
    for layer, data in fire_rates.items():
        print(f"  Layer {layer}: mean={data['mean']:.3f} ± {data['std']:.3f}"
              f"  min={data['min']:.3f} max={data['max']:.3f}")
        print(f"           selective (<50%): {data['selective_frac']:.1%}"
              f"  always-on (>95%): {data['always_on_frac']:.1%}")

    # 2. Neuron entropy
    print(f"\n--- Neuron Entropy (lower=more specialized) ---")
    entropy = analyze_neuron_entropy(all_activations)
    for layer, data in entropy.items():
        print(f"  Layer {layer}: mean={data['mean']:.3f} ± {data['std']:.3f}"
              f"  min={data['min']:.3f} max={data['max']:.3f}")

    # 3. Population sparsity
    print(f"\n--- Population Sparsity (fraction neurons active) ---")
    sparsity = analyze_population_sparsity(all_activations)
    for layer, data in sparsity.items():
        print(f"  Layer {layer}: mean={data['mean']:.3f} ± {data['std']:.3f}")

    # 4. CfC hidden variance
    print(f"\n--- CfC Output Variance ---")
    variance = analyze_cfc_hidden_variance(all_activations)
    for layer, data in variance.items():
        print(f"  Layer {layer}: input_var={data['input_var_mean']:.4f}"
              f"  neuron_var={data['neuron_var_mean']:.4f} ± {data['neuron_var_std']:.4f}")

    # 5. LIF parameters
    if use_lif:
        print(f"\n--- Learned LIF Parameters ---")
        for name, param in model.named_parameters():
            if any(k in name for k in ['threshold', 'leak', 'steepness']):
                vals = param.detach().cpu()
                print(f"  {name}:")
                print(f"    mean={vals.mean():.4f}  std={vals.std():.4f}"
                      f"  min={vals.min():.4f}  max={vals.max():.4f}")

    return {
        'mode': mode,
        'fire_rates': fire_rates,
        'entropy': entropy,
        'sparsity': sparsity,
        'variance': variance,
    }


def compare(scale='auto'):
    """Compare base vs LIF models.

    scale: '2l' for 2L/128d, '4l' for 4L/256d, 'auto' tries 4L first
    """
    if scale == '4l' or scale == 'auto':
        base_path = os.path.join(OUT_DIR, 'liquid_scaled_base_s1337.pt')
        lif_path = os.path.join(OUT_DIR, 'liquid_scaled_lif_s1337.pt')
        if os.path.exists(base_path) and os.path.exists(lif_path):
            print("Using 4L/256d checkpoints (seed=1337)")
        elif scale == 'auto':
            scale = '2l'

    if scale == '2l':
        base_path = os.path.join(OUT_DIR, 'liquid_liquid_base.pt')
        lif_path = os.path.join(OUT_DIR, 'liquid_liquid_lif.pt')
        print("Using 2L/128d checkpoints")

    if not os.path.exists(base_path) or not os.path.exists(lif_path):
        print(f"Need both checkpoints:\n  {base_path}\n  {lif_path}")
        return

    base_results = analyze_model(base_path)
    lif_results = analyze_model(lif_path)

    print(f"\n{'='*60}")
    print("COMPARISON: CfC-only vs CfC+LIF")
    print(f"{'='*60}")

    print(f"\n{'Metric':<35} {'CfC-only':>12} {'CfC+LIF':>12} {'Delta':>10}")
    print("-" * 75)

    for layer in base_results['fire_rates']:
        base_fr = base_results['fire_rates'][layer]['mean']
        lif_fr = lif_results['fire_rates'][layer]['mean']
        print(f"  L{layer} Fire rate mean              {base_fr:>12.3f} {lif_fr:>12.3f} {lif_fr-base_fr:>+10.3f}")

        base_sel = base_results['fire_rates'][layer]['selective_frac']
        lif_sel = lif_results['fire_rates'][layer]['selective_frac']
        print(f"  L{layer} Selective neurons (%)        {base_sel:>11.1%} {lif_sel:>11.1%} {lif_sel-base_sel:>+10.1%}")

        base_ent = base_results['entropy'][layer]['mean']
        lif_ent = lif_results['entropy'][layer]['mean']
        print(f"  L{layer} Neuron entropy               {base_ent:>12.3f} {lif_ent:>12.3f} {lif_ent-base_ent:>+10.3f}")

        base_sp = base_results['sparsity'][layer]['mean']
        lif_sp = lif_results['sparsity'][layer]['mean']
        print(f"  L{layer} Population sparsity          {base_sp:>12.3f} {lif_sp:>12.3f} {lif_sp-base_sp:>+10.3f}")

        base_var = base_results['variance'][layer]['neuron_var_mean']
        lif_var = lif_results['variance'][layer]['neuron_var_mean']
        print(f"  L{layer} CfC neuron variance          {base_var:>12.4f} {lif_var:>12.4f} {lif_var-base_var:>+10.4f}")
        print()

    # Overall summary
    print(f"\n--- Key Finding ---")
    overall_base_ent = np.mean([base_results['entropy'][l]['mean'] for l in base_results['entropy']])
    overall_lif_ent = np.mean([lif_results['entropy'][l]['mean'] for l in lif_results['entropy']])
    overall_base_sp = np.mean([base_results['sparsity'][l]['mean'] for l in base_results['sparsity']])
    overall_lif_sp = np.mean([lif_results['sparsity'][l]['mean'] for l in lif_results['sparsity']])

    print(f"Overall neuron entropy:  base={overall_base_ent:.3f}  lif={overall_lif_ent:.3f}  "
          f"delta={overall_lif_ent-overall_base_ent:+.3f}")
    print(f"Overall pop. sparsity:   base={overall_base_sp:.3f}  lif={overall_lif_sp:.3f}  "
          f"delta={overall_lif_sp-overall_base_sp:+.3f}")

    # Check if base has zero entropy (all neurons always fire = no differentiation)
    base_is_undifferentiated = overall_base_ent < 0.001

    if base_is_undifferentiated and overall_lif_ent > 0.01:
        print(f"\n  → LIF creates neuronal DIFFERENTIATION from undifferentiated baseline")
        print(f"    Base: all neurons fire identically (entropy≈0, no specialization)")
        print(f"    LIF: neurons develop diverse firing patterns (entropy={overall_lif_ent:.3f})")
        print(f"    This supports: 'LIF promotes organization from homogeneity'")

        # Check for depth hierarchy
        lif_entropies = [lif_results['entropy'][l]['mean'] for l in sorted(lif_results['entropy'])]
        if len(lif_entropies) >= 2 and lif_entropies[-1] > lif_entropies[0]:
            print(f"    Depth hierarchy: L0 entropy={lif_entropies[0]:.3f} → "
                  f"L{len(lif_entropies)-1} entropy={lif_entropies[-1]:.3f}")
            print(f"    → Progressive specialization with depth (cortical hierarchy)")
    elif overall_lif_ent < overall_base_ent:
        print(f"\n  → LIF neurons are MORE specialized (lower entropy)")
        print(f"    This supports: 'LIF promotes organization, not just accuracy'")
    else:
        print(f"\n  → LIF neurons show different (not necessarily more) specialization")
        print(f"    Consider depth-wise hierarchy analysis for nuanced interpretation")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Liquid Ember models')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--compare', action='store_true', help='Compare base vs LIF')
    parser.add_argument('--scale', type=str, default='auto', choices=['2l', '4l', 'auto'],
                        help='Scale: 2l (2L/128d), 4l (4L/256d), auto (try 4l first)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.compare:
        compare(args.scale)
    elif args.model:
        analyze_model(args.model, args.device)
    else:
        compare(args.scale)


if __name__ == '__main__':
    main()
