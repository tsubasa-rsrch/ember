"""
collect_lif_params.py
各スケール(xs/small/medium/mid/wide)でLIF training(1000 iters, seed 42)を実行し、
lif_paramsをJSONに保存する。→ analyze_pca_width.py のインプット。

Usage:
    python3 collect_lif_params.py
"""

import sys
import os
import json

EMBER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EMBER_DIR)

import train_tiny

# Short training for parameter collection (convergence-sufficient)
train_tiny.MAX_ITERS = 1500
train_tiny.EVAL_INTERVAL = 500
train_tiny.EVAL_ITERS = 20


def main():
    device = train_tiny.get_device()
    print(f"Device: {device}")
    train_data, val_data, meta = train_tiny.load_data()

    scales = ['xs', 'small', 'medium', 'mid', 'wide']
    all_params = {}

    for scale in scales:
        print(f"\n{'='*60}")
        print(f"Scale: {scale}  (n_embd={train_tiny.SCALES[scale]['n_embd']})")
        print(f"{'='*60}")

        result = train_tiny.train_one(
            'lif', seed=42, device=device,
            train_data=train_data, val_data=val_data,
            meta=meta, scale=scale,
        )

        if 'lif_params' not in result or not result['lif_params']:
            print(f"  WARNING: No lif_params for scale={scale}")
            continue

        lp = result['lif_params']
        n_values = sum(len(v) if isinstance(v, list) else 1 for v in lp.values())

        s = train_tiny.SCALES[scale]
        all_params[scale] = {
            'best_val_loss': result['best_val_loss'],
            'n_params': result['n_params'],
            'n_lif_values': n_values,
            'lif_params': lp,
            'config': {
                'n_layer': s.get('n_layer', s.get('cfc_layers')),
                'n_head': s.get('n_head', 1),
                'n_embd': s.get('n_embd'),
            },
        }
        print(f"  best_val={result['best_val_loss']:.4f} | {n_values} LIF param values saved")

    out_path = os.path.join(EMBER_DIR, 'results', 'lif_params_all_scales.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_params, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(all_params)} scales → {out_path}")
    for sc, data in all_params.items():
        cfg = data['config']
        print(f"  {sc:8s} ({cfg['n_embd']:3d}d): val={data['best_val_loss']:.4f} | {data['n_lif_values']} LIF values")


if __name__ == '__main__':
    main()
