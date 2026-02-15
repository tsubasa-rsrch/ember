#!/usr/bin/env python3
"""Analyze multi-seed ablation results for Ember v2.5."""

import re
import sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def parse_ablation_log(filepath):
    """Extract final val_loss per condition from an ablation log."""
    results = {}
    with open(filepath) as f:
        for line in f:
            # Match lines like: "  Standard              val_loss=1.4923  time=2562.8s  diff=+0.00%"
            m = re.match(r'\s+(\S+)\s+val_loss=(\d+\.\d+)\s+time=(\d+\.\d+)s', line)
            if m:
                condition = m.group(1)
                val_loss = float(m.group(2))
                time_s = float(m.group(3))
                results[condition] = {"val_loss": val_loss, "time": time_s}
    return results

def main():
    # Find all seeded result files
    seed_files = sorted(RESULTS_DIR.glob("ablation_v25_seed*_*.log"))

    if not seed_files:
        print("No seeded result files found in", RESULTS_DIR)
        sys.exit(1)

    print(f"Found {len(seed_files)} seed files:")
    all_results = {}

    for f in seed_files:
        seed_match = re.search(r'seed(\d+)', f.name)
        seed = seed_match.group(1) if seed_match else "?"
        results = parse_ablation_log(f)
        all_results[seed] = results
        print(f"  Seed {seed}: {f.name}")
        for cond, data in sorted(results.items()):
            print(f"    {cond:20s} val_loss={data['val_loss']:.4f}  time={data['time']:.1f}s")

    # Also check for unseeded
    noseed_files = list(RESULTS_DIR.glob("ablation_v25_noseed_*.log"))
    if noseed_files:
        noseed_results = parse_ablation_log(noseed_files[0])
        print(f"\n  No-seed (reference): {noseed_files[0].name}")
        for cond, data in sorted(noseed_results.items()):
            print(f"    {cond:20s} val_loss={data['val_loss']:.4f}")

    # Compute mean ± std across seeds
    conditions = set()
    for seed_results in all_results.values():
        conditions.update(seed_results.keys())
    # Put Standard first, then sort the rest
    conditions = sorted(conditions)
    if "Standard" in conditions:
        conditions.remove("Standard")
        conditions.insert(0, "Standard")

    print("\n" + "=" * 70)
    print("MULTI-SEED ANALYSIS")
    print("=" * 70)
    print(f"\nSeeds: {', '.join(sorted(all_results.keys()))}")
    print(f"{'Condition':20s} {'Mean':>8s} {'± Std':>8s} {'Min':>8s} {'Max':>8s} {'N':>4s} {'vs Standard':>12s}")
    print("-" * 70)

    standard_mean = None
    for cond in conditions:
        vals = [all_results[s][cond]["val_loss"] for s in all_results if cond in all_results[s]]
        if len(vals) == 0:
            continue
        mean = np.mean(vals)
        std = np.std(vals) if len(vals) > 1 else 0
        mn = min(vals)
        mx = max(vals)

        if cond == "Standard":
            standard_mean = mean
            diff_str = "baseline"
        elif standard_mean is not None:
            diff_pct = (mean - standard_mean) / standard_mean * 100
            diff_str = f"{diff_pct:+.2f}%"
        else:
            diff_str = "—"

        print(f"{cond:20s} {mean:8.4f} {std:8.4f} {mn:8.4f} {mx:8.4f} {len(vals):4d} {diff_str:>12s}")

    # Markdown table for DESIGN.md
    print("\n\nMarkdown table for DESIGN.md:")
    print("| Condition | Mean | ± Std | Min | Max | N | vs Standard |")
    print("|-----------|------|-------|-----|-----|---|-------------|")

    standard_mean = None
    for cond in conditions:
        vals = [all_results[s][cond]["val_loss"] for s in all_results if cond in all_results[s]]
        if not vals:
            continue
        mean = np.mean(vals)
        std = np.std(vals) if len(vals) > 1 else 0

        if cond == "Standard":
            standard_mean = mean
            diff_str = "baseline"
        elif standard_mean:
            diff_pct = (mean - standard_mean) / standard_mean * 100
            diff_str = f"{diff_pct:+.2f}%"
        else:
            diff_str = "—"

        print(f"| {cond} | {mean:.4f} | {std:.4f} | {min(vals):.4f} | {max(vals):.4f} | {len(vals)} | {diff_str} |")

    # Training time analysis
    print("\n\nTraining time (seconds):")
    print(f"{'Condition':20s} {'Mean':>8s} {'Overhead':>10s}")
    print("-" * 40)

    standard_time = None
    for cond in conditions:
        times = [all_results[s][cond]["time"] for s in all_results if cond in all_results[s]]
        if not times:
            continue
        mean_t = np.mean(times)

        if cond == "Standard":
            standard_time = mean_t
            overhead_str = "baseline"
        elif standard_time:
            overhead_pct = (mean_t - standard_time) / standard_time * 100
            overhead_str = f"+{overhead_pct:.1f}%"
        else:
            overhead_str = "—"

        print(f"{cond:20s} {mean_t:8.1f} {overhead_str:>10s}")

if __name__ == "__main__":
    main()
