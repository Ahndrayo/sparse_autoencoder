# viz_analysis/viz.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    repo_root = Path(__file__).resolve().parent.parent
    lat_path = repo_root / "analysis_data" / "latent_activations.npy"
    if not lat_path.exists():
        raise SystemExit(f"Missing {lat_path}. Run main.py first.")
    lat = np.load(lat_path)   # shape [T, M]
    if lat.ndim != 2:
        raise SystemExit(f"Expected 2D array [T, M], got {lat.shape}")
    T, M = lat.shape
    print(f"Loaded latents: T={T} tokens, M={M} features")

    # --- Sparsity across features (fraction > 0 for each feature) ---
    frac_active = (lat > 0).mean(axis=0)  # [M]
    plt.figure()
    plt.hist(frac_active, bins=50)
    plt.title("Feature sparsity (fraction > 0)")
    plt.xlabel("Fraction of tokens with activation > 0")
    plt.ylabel("Number of features")
    plt.show(block=False)

    # --- Top-K features by mean activation over tokens ---
    mean_per_feat = lat.mean(axis=0)      # [M]
    max_per_feat  = lat.max(axis=0)       # [M]
    nz_frac       = (lat > 0).mean(axis=0)

    k = min(args.topk, M)
    top_idx = np.argsort(mean_per_feat)[-k:][::-1]
    print(f"\nTop {k} features by MEAN activation (over {T} tokens)")
    print(f"{'rank':>4}  {'feat':>6}  {'mean_act':>12}  {'max_act':>10}  {'frac>0':>8}")
    for r, j in enumerate(top_idx, 1):
        print(f"{r:>4}  {j:>6}  {mean_per_feat[j]:>12.6f}  {max_per_feat[j]:>10.6f}  {nz_frac[j]:>8.3f}")

    plt.figure()
    plt.bar(range(k), mean_per_feat[top_idx])
    plt.title(f"Top-{k} features by mean activation")
    plt.ylabel("Mean activation")
    plt.xticks(range(k), top_idx, rotation=45, ha='right')  # use the same list as the plotted bars
    plt.xlabel("Feature ID")
    plt.show(block=False)

    # --- Optional: per-token view ---
    if args.token is not None:
        t = args.token
        if not (0 <= t < T):
            raise SystemExit(f"--token must be in [0, {T-1}]")
        row = lat[t]  # [M]
        top_tok_idx = np.argsort(row)[-k:][::-1]
        print(f"\nTop {k} features at token {t} / {T-1}")
        print(f"{'rank':>4}  {'feat':>6}  {'act':>10}")
        for r, j in enumerate(top_tok_idx, 1):
            print(f"{r:>4}  {j:>6}  {row[j]:>10.6f}")

        plt.figure()
        plt.bar(range(k), row[top_tok_idx])
        plt.title(f"Top-{k} feature activations at token {t}")
        plt.ylabel("Activation Check")
        plt.xticks(range(k), top_tok_idx, rotation=45, ha='right')  # use the same list as the plotted bars
        plt.xlabel("Feature ID")
        plt.show(block=False)



    plt.show()  # keep figures open

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=20, help="How many features to show in top-k plots")
    ap.add_argument("--token", type=int, default=None, help="Inspect a specific token index")
    args = ap.parse_args()
    main(args)
