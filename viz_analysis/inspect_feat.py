# feature_probe.py
import numpy as np
from transformer_lens import HookedTransformer

# Assume you've run main.py and have latent_activations + tokens saved
# For demonstration, load them back if saved; otherwise reuse in same session
feat_id = 18   # change this to whichever bar was tallest
top_k = 5      # how many strongest activations to see

lat = latent_activations.detach().cpu().numpy()   # [T, M]
tok_strs = model.to_str_tokens(prompt)            # tokens -> text list

vals = lat[:, feat_id]
top_idx = np.argsort(vals)[::-1][:top_k]

print(f"\nTop {top_k} tokens activating feature {feat_id}:")
for i in top_idx:
    print(f"  Token {i:>3}: '{tok_strs[i]}'  act={vals[i]:.4f}")
