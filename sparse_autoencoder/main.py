import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder
import os, json
import numpy as np

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 6
location = "resid_post_mlp"

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

input_tensor = activation_cache[transformer_lens_loc]

input_tensor_ln = input_tensor

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor_ln)
    reconstructed_activations = autoencoder.decode(latent_activations, info)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print("This is my output:")
print(location, normalized_mse)


# Map 'location' -> final hook name suffix to keep folder names consistent
hook_suffix = {
    "mlp_post_act":   f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn":  f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp":  f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp":   f"blocks.{layer_index}.hook_resid_post",
}[location]

# Where to write files (repo root /viewer/...)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_dir = os.path.join(
    repo_root, "viewer", "gpt2-small", location, hook_suffix
)
atoms_dir = os.path.join(out_dir, "atoms")
os.makedirs(atoms_dir, exist_ok=True)

# Convert latents to numpy
lat = latent_activations
if hasattr(lat, "detach"): lat = lat.detach()
if hasattr(lat, "cpu"):    lat = lat.cpu()
lat_np = lat.numpy()  # shape [T, M]
T, M = lat_np.shape

# --- index.json (write once) ---
index_payload = {
    "n_features": int(M),
    "layer": int(layer_index),
    "location": str(location),
    "subject": "gpt2-small",
}
with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2)

# --- atoms/*.json: per-feature summaries ---
num_to_dump = min(512, M)
for j in range(num_to_dump):
    feat = lat_np[:, j]  # activations over tokens for feature j
    atom_payload = {
        "feature_id": int(j),
        "mean_activation": float(feat.mean()),
        "max_activation": float(feat.max()),
        "nonzero_frac": float((feat > 0).mean()),
        # You can add more later (e.g., top token strings, histograms, thresholds, etc.)
    }
    with open(os.path.join(atoms_dir, f"{j}.json"), "w", encoding="utf-8") as f:
        json.dump(atom_payload, f, indent=2)

print("Wrote viewer files to:", out_dir)