"""Aggregate v3.5 ablation results across 3 seeds from checkpoint files.

Usage: python3 aggregate_v35.py
Run after all 3 seeds complete (s42, s668, s1337).
"""
import torch
import os

CONDITIONS = [
    ("Standard",      "ember_standard.pt"),
    ("LIF-fixed",     "ember_lif_fixed.pt"),
    ("LIF-learnable", "ember_lif.pt"),
    ("LIF-refractory","ember_lif_refrac.pt"),
    ("Head-persist",  "ember_head_persist.pt"),
    ("Refrac+Head",   "ember_refrac+head.pt"),
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

def load_results():
    """Load best_val_loss from each checkpoint."""
    results = {}
    for name, fname in CONDITIONS:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            loss = ckpt.get("best_val_loss", None)
            if loss is not None:
                results[name] = float(loss)
            else:
                results[name] = None
        else:
            results[name] = None
    return results

def main():
    results = load_results()

    print("=" * 60)
    print("Ember v3.5 Ablation Results (current checkpoint data)")
    print("=" * 60)

    # Find best
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("No results found!")
        return

    best_name = min(valid, key=valid.get)
    best_loss = valid[best_name]
    std_loss = valid.get("Standard", None)

    print(f"\n{'Condition':<20} {'best_val_loss':>14} {'vs Standard':>12} {'vs Best':>10}")
    print("-" * 60)

    for name, _ in CONDITIONS:
        loss = results.get(name)
        if loss is None:
            print(f"{name:<20} {'N/A':>14}")
            continue

        vs_std = ""
        if std_loss is not None and name != "Standard":
            delta = (loss - std_loss) / std_loss * 100
            vs_std = f"{delta:+.2f}%"

        vs_best = ""
        if name != best_name:
            delta = (loss - best_loss) / best_loss * 100
            vs_best = f"{delta:+.2f}%"

        marker = " ← BEST" if name == best_name else ""
        print(f"{name:<20} {loss:>14.4f} {vs_std:>12} {vs_best:>10}{marker}")

    print(f"\nBest: {best_name} ({best_loss:.4f})")
    if std_loss:
        print(f"Standard: {std_loss:.4f}")
        print(f"Best vs Standard: {(best_loss - std_loss) / std_loss * 100:+.2f}%")

if __name__ == "__main__":
    main()
