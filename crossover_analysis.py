"""
Crossover Analysis: When does LIF overtake Standard during training?

Hypothesis: Crossover point is depth-dependent.
- xs (2L): No crossover, LIF leads from start
- small (4L): Intermediate crossover
- medium (6L): Late crossover (~iter 1600, matching 10M Ember)

Usage:
    python3 crossover_analysis.py
"""

import os
import json
import glob
import numpy as np

RESULTS_DIR = os.path.expanduser("~/Documents/TsubasaWorkspace/ember/results")


def load_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "tiny_*.json")))
    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        all_results.extend(data)
    return all_results


def infer_scale(n_params):
    if n_params < 600_000:
        return 'xs'
    elif n_params < 3_000_000:
        return 'small'
    elif n_params < 8_000_000:
        return 'medium'
    else:
        return 'large'


def find_crossover(std_history, lif_history):
    """Find iteration where LIF first becomes better than Standard.
    Returns (crossover_iter, None) or (None, 'lif_always_better') or (None, 'std_always_better').
    """
    if not std_history or not lif_history:
        return None, 'no_data'

    # Align by iteration
    std_by_iter = {h['iter']: h['val_loss'] for h in std_history}
    lif_by_iter = {h['iter']: h['val_loss'] for h in lif_history}

    common_iters = sorted(set(std_by_iter.keys()) & set(lif_by_iter.keys()))
    if not common_iters:
        return None, 'no_common_iters'

    # Find first iter where LIF < Standard
    lif_leads_from = None
    for it in common_iters:
        if lif_by_iter[it] < std_by_iter[it]:
            lif_leads_from = it
            break

    if lif_leads_from is None:
        return None, 'std_always_better'

    # Check if LIF was always better
    if lif_leads_from == common_iters[0] or lif_leads_from <= 200:
        return lif_leads_from, 'lif_always_better'

    return lif_leads_from, 'crossover'


def analyze_crossover_by_scale(results):
    """Group by (scale, seed) and find crossover for each pair."""
    # Filter to full runs only (>= 2800 iters)
    results = [r for r in results if r.get('history') and max(h['iter'] for h in r['history']) >= 2800]

    # Group by (scale, model_type, seed)
    grouped = {}
    for r in results:
        scale = infer_scale(r['n_params'])
        key = (scale, r['model_type'], r['seed'])
        if key not in grouped or r['best_val_loss'] < grouped[key]['best_val_loss']:
            grouped[key] = r

    # Find crossover for each (scale, seed) pair
    scales = sorted(set(k[0] for k in grouped.keys()),
                    key=lambda s: {'xs': 0, 'small': 1, 'medium': 2, 'large': 3}.get(s, 99))
    seeds = sorted(set(k[2] for k in grouped.keys()))

    print("=" * 80)
    print("CROSSOVER ANALYSIS: When does LIF overtake Standard?")
    print("=" * 80)

    for scale in scales:
        n_layers = {'xs': 2, 'small': 4, 'medium': 6}.get(scale, '?')
        print(f"\n--- Scale: {scale.upper()} ({n_layers} layers) ---")

        for seed in seeds:
            std_key = (scale, 'standard', seed)
            lif_key = (scale, 'lif', seed)

            if std_key not in grouped or lif_key not in grouped:
                continue

            std_hist = grouped[std_key]['history']
            lif_hist = grouped[lif_key]['history']

            crossover_iter, status = find_crossover(std_hist, lif_hist)

            # Also compute per-eval-point deltas
            std_by_iter = {h['iter']: h['val_loss'] for h in std_hist}
            lif_by_iter = {h['iter']: h['val_loss'] for h in lif_hist}
            common = sorted(set(std_by_iter.keys()) & set(lif_by_iter.keys()))

            delta_str = ""
            for it in common:
                d = lif_by_iter[it] - std_by_iter[it]
                marker = "  " if d >= 0 else "* "
                delta_str += f"    {marker}iter {it:5d}: std={std_by_iter[it]:.4f} lif={lif_by_iter[it]:.4f} delta={d:+.4f}\n"

            if status == 'lif_always_better':
                print(f"  seed={seed}: LIF leads from iter {crossover_iter} (no crossover)")
            elif status == 'crossover':
                print(f"  seed={seed}: CROSSOVER at iter {crossover_iter}")
            elif status == 'std_always_better':
                print(f"  seed={seed}: Standard always better (no crossover)")
            else:
                print(f"  seed={seed}: {status}")

            print(delta_str.rstrip())

    # Summary
    print("\n" + "=" * 80)
    print("DEPTH vs CROSSOVER SUMMARY")
    print("=" * 80)
    print(f"{'Scale':<10} {'Layers':<8} {'Crossover Pattern'}")
    print("-" * 50)

    for scale in scales:
        n_layers = {'xs': 2, 'small': 4, 'medium': 6}.get(scale, '?')
        crossovers = []
        for seed in seeds:
            std_key = (scale, 'standard', seed)
            lif_key = (scale, 'lif', seed)
            if std_key in grouped and lif_key in grouped:
                ci, status = find_crossover(grouped[std_key]['history'], grouped[lif_key]['history'])
                crossovers.append((ci, status))

        if not crossovers:
            continue

        pattern_strs = []
        for ci, status in crossovers:
            if status == 'lif_always_better':
                pattern_strs.append(f"LIF from {ci}")
            elif status == 'crossover':
                pattern_strs.append(f"cross@{ci}")
            elif status == 'std_always_better':
                pattern_strs.append("no cross")
            else:
                pattern_strs.append(status)

        print(f"{scale:<10} {str(n_layers):<8} {' | '.join(pattern_strs)}")

    print("\nReference: 10M Ember (6L/12H/768d) crossover at ~iter 1600")


if __name__ == '__main__':
    results = load_results()
    if not results:
        print("No results found!")
        exit(1)
    analyze_crossover_by_scale(results)
