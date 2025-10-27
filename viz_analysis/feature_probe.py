# viz_analysis/feature_probe.py
import numpy as np
from transformer_lens import HookedTransformer

# Load saved data
lat = np.load("../analysis_data/latent_activations.npy")   # [T, M]
tok_ids = np.load("../analysis_data/tokens.npy")
with open("../analysis_data/prompt.txt") as f:
    prompt = f.read().strip()

# Rebuild model only to decode tokens (no need for SAE)
model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)

# Convert token IDs to readable strings
tok_strs = model.to_str_tokens(prompt)

# Pick feature and top-k
feat_id = 1925
top_k = 10

vals = lat[:, feat_id]
top_idx = np.argsort(vals)[::-1][:top_k]

print(f"\nTop {top_k} tokens activating feature {feat_id}:")
for i in top_idx:
    print(f"  Token {i:>3}: '{tok_strs[i]}'  act={vals[i]:.4f}")
