"""
Analyze learned LIF parameters from Ember checkpoints.
Shows how each attention head specialized its threshold/leak/steepness.
"""

import os
import sys
import torch
import numpy as np

OUT_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/out")


def analyze_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    state = ckpt['model']

    print(f"Checkpoint: {path}")
    print(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
    print(f"  Iteration: {ckpt['iter']}")
    print(f"  Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    print()

    # Collect LIF parameters
    for layer_idx in range(config.n_layer):
        prefix = f"transformer.h.{layer_idx}.attn"

        threshold_key = f"{prefix}.threshold"
        leak_key = f"{prefix}.leak"
        steepness_key = f"{prefix}.steepness"

        if threshold_key not in state:
            print(f"  Layer {layer_idx}: No LIF parameters (standard attention?)")
            continue

        threshold = state[threshold_key]
        leak = state[leak_key]
        steepness = state[steepness_key]

        # Compute effective values
        eff_threshold = torch.abs(threshold) * 0.1
        eff_leak = torch.sigmoid(leak)
        eff_steepness = torch.nn.functional.softplus(steepness)

        print(f"  Layer {layer_idx}:")
        for h in range(config.n_head):
            t = eff_threshold[h].item()
            l = eff_leak[h].item()
            s = eff_steepness[h].item()

            # Characterize head behavior
            if t > 0.02 and l < 0.85:
                mode = "SELECTIVE"  # high threshold, low leak = sharp filter
            elif t > 0.02 and l > 0.85:
                mode = "mild-filter"  # high threshold but high leak = mild
            elif t < 0.005:
                mode = "pass-through"  # very low threshold = nearly identity
            else:
                mode = "moderate"

            print(f"    Head {h}: thresh={t:.4f} leak={l:.3f} steep={s:.2f}  [{mode}]")

    print()


def compare_checkpoints():
    lif_path = os.path.join(OUT_DIR, "ember_lif.pt")
    std_path = os.path.join(OUT_DIR, "ember_std.pt")

    if os.path.exists(std_path):
        print("=" * 60)
        print("STANDARD ATTENTION BASELINE")
        print("=" * 60)
        ckpt = torch.load(std_path, map_location='cpu', weights_only=False)
        print(f"  Best val loss: {ckpt['best_val_loss']:.4f}")
        print(f"  Iteration: {ckpt['iter']}")
        print()

    if os.path.exists(lif_path):
        print("=" * 60)
        print("LIF ATTENTION")
        print("=" * 60)
        analyze_checkpoint(lif_path)
    else:
        print("No LIF checkpoint found. Run training first.")


if __name__ == '__main__':
    compare_checkpoints()
