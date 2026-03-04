"""
Analyze Ember-Tiny cross-scale results.
Loads all results files and produces a comparison table.

Usage:
    python3 analyze_tiny.py              # Auto-detect and analyze all results
    python3 analyze_tiny.py --latest     # Only show the latest run per scale
"""

import os
import json
import glob
import argparse
import numpy as np

RESULTS_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/results")


def load_all_tiny_results():
    """Load all tiny_*.json result files."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "tiny_*.json")))
    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        for r in data:
            r['_file'] = os.path.basename(f)
        all_results.extend(data)
    return all_results


def infer_scale(n_params):
    """Infer scale from param count."""
    if n_params < 600_000:
        return 'xs'
    elif n_params < 3_000_000:
        return 'small'
    elif n_params < 8_000_000:
        return 'medium'
    elif n_params < 15_000_000:
        return 'wide'
    else:
        return 'large'


def analyze_results(results, min_iters=2000):
    """Group by scale and model_type, compute stats.
    Filter out runs with fewer than min_iters iterations.
    Deduplicate by keeping only the best run per (scale, model_type, seed).
    """
    # Add scale info and max iter
    for r in results:
        r['scale'] = infer_scale(r['n_params'])
        r['max_iter'] = max(h['iter'] for h in r['history']) if r.get('history') else 0

    # Filter by min iterations
    filtered = [r for r in results if r['max_iter'] >= min_iters]
    print(f"Filtered: {len(results)} -> {len(filtered)} runs (min_iters={min_iters})")

    # Deduplicate: keep best per (scale, model_type, seed)
    best = {}
    for r in filtered:
        key = (r['scale'], r['model_type'], r['seed'])
        if key not in best or r['best_val_loss'] < best[key]['best_val_loss']:
            best[key] = r

    deduped = list(best.values())
    print(f"Deduplicated: {len(filtered)} -> {len(deduped)} runs")

    # Group by (scale, model_type)
    groups = {}
    for r in deduped:
        key = (r['scale'], r['model_type'])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    return groups


def print_cross_scale_table(groups):
    """Print a comprehensive cross-scale comparison table."""
    scales = sorted(set(k[0] for k in groups.keys()),
                    key=lambda s: {'xs': 0, 'small': 1, 'medium': 2, 'wide': 3, 'large': 4}.get(s, 99))
    model_types = ['standard', 'lif', 'cfc_base', 'cfc']

    print("\n" + "=" * 90)
    print("EMBER-TINY CROSS-SCALE COMPARISON")
    print("=" * 90)

    for scale in scales:
        print(f"\n--- Scale: {scale.upper()} ---")

        standard_mean = None
        for mt in model_types:
            key = (scale, mt)
            if key not in groups:
                continue

            runs = groups[key]
            vals = [r['best_val_loss'] for r in runs]
            seeds = [r['seed'] for r in runs]
            n_params = runs[0]['n_params']
            mode_str = runs[0]['mode_str']

            mean = np.mean(vals)
            std = np.std(vals) if len(vals) > 1 else 0.0

            if mt == 'standard':
                standard_mean = mean

            # Per-seed details
            seed_str = " | ".join(f"s{s}={v:.4f}" for s, v in sorted(zip(seeds, vals)))

            # Delta vs standard
            if mt == 'standard':
                delta_str = "baseline"
            elif standard_mean is not None:
                delta = (mean - standard_mean) / standard_mean * 100
                wins = sum(1 for v, sv in zip(
                    sorted(vals), sorted([r['best_val_loss'] for r in groups.get((scale, 'standard'), [])]))
                    if v < sv) if (scale, 'standard') in groups else '?'
                delta_str = f"{delta:+.3f}%"
            else:
                delta_str = "no baseline"

            print(f"  {mode_str:<12} | {n_params/1e6:.3f}M | {mean:.4f}±{std:.4f} | {delta_str}")
            print(f"    {seed_str}")

    # Cross-scale summary
    print("\n" + "=" * 90)
    print("CROSS-SCALE SUMMARY: LIF effect by scale")
    print("=" * 90)
    print(f"{'Scale':<10} {'Params':<10} {'Standard':<18} {'LIF':<18} {'Delta':<10} {'Wins'}")
    print("-" * 80)

    for scale in scales:
        std_key = (scale, 'standard')
        lif_key = (scale, 'lif')
        if std_key not in groups or lif_key not in groups:
            continue

        std_vals = [r['best_val_loss'] for r in groups[std_key]]
        lif_vals = [r['best_val_loss'] for r in groups[lif_key]]
        n_params = groups[std_key][0]['n_params']

        std_mean = np.mean(std_vals)
        std_std = np.std(std_vals) if len(std_vals) > 1 else 0.0
        lif_mean = np.mean(lif_vals)
        lif_std = np.std(lif_vals) if len(lif_vals) > 1 else 0.0

        delta = (lif_mean - std_mean) / std_mean * 100

        # Per-seed wins
        std_sorted = sorted(zip([r['seed'] for r in groups[std_key]], std_vals))
        lif_sorted = sorted(zip([r['seed'] for r in groups[lif_key]], lif_vals))
        wins = sum(1 for (_, lv), (_, sv) in zip(lif_sorted, std_sorted) if lv < sv)
        total = min(len(std_vals), len(lif_vals))

        print(f"{scale:<10} {n_params/1e6:.3f}M    {std_mean:.4f}±{std_std:.4f}  {lif_mean:.4f}±{lif_std:.4f}  {delta:+.3f}%   {wins}/{total}")

    # CfC summary if available
    cfc_scales = [s for s in scales if (s, 'cfc') in groups or (s, 'cfc_base') in groups]
    if cfc_scales:
        print(f"\n--- CfC Architecture Comparison ---")
        print(f"{'Scale':<8} {'CfC-only':<20} {'CfC+LIF':<20} {'Delta (LIF vs Base)':<22} {'vs Transformer'}")
        print("-" * 90)
        for scale in cfc_scales:
            base_runs = groups.get((scale, 'cfc_base'), [])
            lif_runs = groups.get((scale, 'cfc'), [])
            std_mean = np.mean([r['best_val_loss'] for r in groups.get((scale, 'standard'), [])]) if (scale, 'standard') in groups else None

            base_str = f"{np.mean([r['best_val_loss'] for r in base_runs]):.4f}±{np.std([r['best_val_loss'] for r in base_runs]):.4f}" if base_runs else "N/A"
            lif_str = f"{np.mean([r['best_val_loss'] for r in lif_runs]):.4f}±{np.std([r['best_val_loss'] for r in lif_runs]):.4f}" if lif_runs else "N/A"

            if base_runs and lif_runs:
                base_mean = np.mean([r['best_val_loss'] for r in base_runs])
                lif_mean = np.mean([r['best_val_loss'] for r in lif_runs])
                delta = (lif_mean - base_mean) / base_mean * 100
                delta_str = f"{delta:+.3f}% ({'✓LIF wins' if delta < 0 else '✗LIF loses'})"
                vs_trans = f"{(lif_mean - std_mean) / std_mean * 100:+.3f}%" if std_mean else "N/A"
            else:
                delta_str = "incomplete"
                vs_trans = "N/A"

            print(f"  {scale:<6}  {base_str:<20} {lif_str:<20} {delta_str:<22} {vs_trans}")

        # Architecture independence check
        print(f"\n--- Architecture Independence (Paper 1 claim) ---")
        for scale in cfc_scales:
            base_runs = groups.get((scale, 'cfc_base'), [])
            lif_runs = groups.get((scale, 'cfc'), [])
            trans_std = groups.get((scale, 'standard'), [])
            trans_lif = groups.get((scale, 'lif'), [])
            if base_runs and lif_runs and trans_std and trans_lif:
                cfc_delta = (np.mean([r['best_val_loss'] for r in lif_runs]) - np.mean([r['best_val_loss'] for r in base_runs])) / np.mean([r['best_val_loss'] for r in base_runs]) * 100
                trans_delta = (np.mean([r['best_val_loss'] for r in trans_lif]) - np.mean([r['best_val_loss'] for r in trans_std])) / np.mean([r['best_val_loss'] for r in trans_std]) * 100
                both_improve = cfc_delta < 0 and trans_delta < 0
                print(f"  {scale}: Transformer {trans_delta:+.3f}% | CfC {cfc_delta:+.3f}% => {'✓ BOTH ARCHITECTURES IMPROVE' if both_improve else '✗ mixed results'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latest', action='store_true')
    args = parser.parse_args()

    results = load_all_tiny_results()
    if not results:
        print("No results found!")
        exit(1)

    print(f"Loaded {len(results)} runs from {len(set(r['_file'] for r in results))} files")

    groups = analyze_results(results)
    print_cross_scale_table(groups)
