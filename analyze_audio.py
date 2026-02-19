"""
Analyze Audio Liquid Ember internal organization.

Extract neuron firing rates and entropy per layer,
comparing Base (CfC-only) vs LIF (CfC+LIF).
Same methodology as text Liquid Ember analysis.

Usage:
    python analyze_audio.py

2026-02-19 — Tsubasa
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F

from audio_liquid_ember import AudioLiquidEmber, AudioLiquidEmberConfig
from liquid_ember import LIFGate
from train_audio import SpeechCommandsDataset, LABELS


OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def extract_lif_activations(model, mel_batch):
    """Hook into LIF gates to capture fire_mask per layer."""
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
        model(mel_batch)

    for h in hooks:
        h.remove()

    return activations


def compute_metrics(activations):
    """Fire rate and entropy per layer."""
    results = {}
    for layer_idx, data in enumerate(activations):
        fm = data['fire_mask']
        B, T, D = fm.shape
        firing = (fm > 0.5).float()
        fire_rate = firing.mean(dim=(0, 1))

        # Entropy per neuron
        flat = fm.reshape(-1, D)
        entropies = []
        for d in range(D):
            vals = flat[:, d].numpy()
            hist, _ = np.histogram(vals, bins=10, range=(0, 1), density=True)
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            ent = -(hist * np.log(hist + 1e-10)).sum()
            entropies.append(ent)
        entropies = np.array(entropies)

        results[layer_idx] = {
            'fire_rate_mean': fire_rate.mean().item(),
            'fire_rate_std': fire_rate.std().item(),
            'entropy_mean': entropies.mean(),
            'entropy_std': entropies.std(),
            'always_on_frac': (fire_rate > 0.95).float().mean().item(),
            'selective_frac': (fire_rate < 0.5).float().mean().item(),
        }
    return results


def main():
    device = 'cpu'  # Analysis on CPU for reproducibility
    n_batches = 10
    batch_size = 32

    print("=" * 70)
    print("AUDIO LIQUID EMBER — INTERNAL ORGANIZATION ANALYSIS")
    print("=" * 70)

    seeds = [42, 668, 1337]
    conditions = ['base', 'lif']
    all_results = {}

    for seed in seeds:
        for cond in conditions:
            tag = f"{cond}_s{seed}"
            path = os.path.join(OUT_DIR, f'audio_{tag}.pt')
            if not os.path.exists(path):
                continue

            print(f"\nLoading {tag}...")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            config = ckpt['config']
            model = AudioLiquidEmber(config)
            model.load_state_dict(ckpt['model'])
            model.eval()

            # Load validation data
            val_ds = SpeechCommandsDataset(DATA_DIR, 'validation', config)

            # Accumulate activations
            all_acts = None
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                if start >= len(val_ds):
                    break
                end = min(start + batch_size, len(val_ds))
                mels = torch.stack([val_ds[i][0] for i in range(start, end)])

                acts = extract_lif_activations(model, mels)
                if all_acts is None:
                    all_acts = acts
                else:
                    for li in range(len(acts)):
                        for k in acts[li]:
                            all_acts[li][k] = torch.cat([all_acts[li][k], acts[li][k]], dim=0)

            metrics = compute_metrics(all_acts)
            all_results[tag] = metrics

            mode = "CfC+LIF" if config.use_lif else "CfC-only"
            n_layers = len(metrics)
            ent_str = " | ".join([f"L{l}={metrics[l]['entropy_mean']:.3f}" for l in range(n_layers)])
            fr_str = " | ".join([f"L{l}={metrics[l]['fire_rate_mean']:.3f}" for l in range(n_layers)])
            print(f"  {tag:15s} [{mode:8s}] entropy: {ent_str}")
            print(f"  {'':15s}            fire_rt: {fr_str}")

            # LIF threshold values
            if config.use_lif:
                for i, block in enumerate(model.blocks):
                    if hasattr(block.lif_gate, 'threshold'):
                        t = torch.abs(block.lif_gate.threshold) * 0.1
                        print(f"  L{i} threshold: mean={t.mean():.4f}, std={t.std():.4f}")

    # Cross-seed summary
    if len(all_results) >= 2:
        n_layers = 4
        print(f"\n{'=' * 70}")
        print("CROSS-SEED SUMMARY")
        print(f"{'=' * 70}")

        for cond in conditions:
            ent_per_layer = {l: [] for l in range(n_layers)}
            fr_per_layer = {l: [] for l in range(n_layers)}
            accs = []
            for seed in seeds:
                tag = f"{cond}_s{seed}"
                if tag not in all_results:
                    continue
                for l in range(n_layers):
                    ent_per_layer[l].append(all_results[tag][l]['entropy_mean'])
                    fr_per_layer[l].append(all_results[tag][l]['fire_rate_mean'])
                # Load accuracy
                ckpt_path = os.path.join(OUT_DIR, f'audio_{tag}.pt')
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    accs.append(ckpt.get('val_acc', 0))

            if not ent_per_layer[0]:
                continue

            ent_means = [np.mean(ent_per_layer[l]) for l in range(n_layers)]
            ent_stds = [np.std(ent_per_layer[l]) for l in range(n_layers)]
            fr_means = [np.mean(fr_per_layer[l]) for l in range(n_layers)]
            acc_mean = np.mean(accs) if accs else 0
            acc_std = np.std(accs) if accs else 0

            cond_label = "CfC+LIF" if cond == 'lif' else "CfC-only"
            print(f"\n  {cond_label} (mean ± std across {len(ent_per_layer[0])} seeds):")
            print(f"    Val Acc: {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"    Entropy: " + " | ".join(
                [f"L{l}={ent_means[l]:.3f}±{ent_stds[l]:.3f}" for l in range(n_layers)]))
            print(f"    Fire rt: " + " | ".join(
                [f"L{l}={fr_means[l]:.3f}" for l in range(n_layers)]))

        # Hierarchy pattern
        print(f"\n--- Hierarchy Pattern ---")
        print(f"{'Condition':>12} {'L0 entropy':>12} {'L3 entropy':>12} {'Trend':>8}")
        print("-" * 50)
        for cond in conditions:
            ents = []
            for seed in seeds:
                tag = f"{cond}_s{seed}"
                if tag in all_results:
                    ents.append([all_results[tag][l]['entropy_mean'] for l in range(n_layers)])
            if ents:
                avg = np.mean(ents, axis=0)
                trend = "↑" if avg[-1] > avg[0] else "↓"
                label = "CfC+LIF" if cond == 'lif' else "CfC-only"
                print(f"{label:>12} {avg[0]:>12.3f} {avg[-1]:>12.3f} {trend:>8}")


if __name__ == '__main__':
    main()
