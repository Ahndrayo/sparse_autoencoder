# viz_analysis/viz.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# path to the folder with index.json
root = Path("../viewer/gpt2-small/resid_post_mlp/blocks.6.hook_resid_post")

# load all atoms
atoms_dir = root / "atoms"
atoms = []
for file in atoms_dir.glob("*.json"):
    with open(file) as f:
        atoms.append(json.load(f))

mean_act = np.array([a["mean_activation"] for a in atoms])
max_act  = np.array([a["max_activation"] for a in atoms])
sparsity = np.array([a["nonzero_frac"] for a in atoms])

# Plot sparsity histogram
plt.figure()
plt.hist(sparsity, bins=50)
plt.title("Feature sparsity (fraction > 0)")
plt.xlabel("Fraction active")
plt.ylabel("Count")
plt.show()

# Top 20 by mean activation
topk = np.argsort(mean_act)[-20:]
plt.figure()
plt.bar(range(20), mean_act[topk])
plt.title("Top-20 features by mean activation")
plt.xlabel("Feature rank (top-k)")
plt.ylabel("Mean activation")
plt.show()
