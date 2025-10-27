# viz_analysis/feature_probe.py
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from utils.run_picker import get_latest_run

# Load saved data
repo_root = Path(__file__).resolve().parent.parent
analysis_dir = repo_root / "analysis_data"
latest_run = get_latest_run(analysis_dir)
print("Loading data from: ", latest_run)

lat = np.load(latest_run / "latent_activations.npy")  # [T, M]
tok_ids = np.load(latest_run / "tokens.npy")
with open(latest_run / "prompt.txt", "r") as f:
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
