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

# ----- where to write files -----
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_dir = os.path.join(
    repo_root,
    "viewer",
    "gpt2-small",                   # subject (you can rename if you like)
    location,                       # e.g., "resid_post_mlp"
    f"blocks.{layer_index}.hook_resid_post"  # matches transformer_lens_loc last segment
)
atoms_dir = os.path.join(out_dir, "atoms")
os.makedirs(atoms_dir, exist_ok=True)

# ----- write a tiny index.json -----
index_payload = {
    "n_features": int(latent_activations.shape[-1]) if latent_activations.ndim > 0 else 0,
    "layer": layer_index,
    "location": location,
    "subject": "gpt2-small",
}
with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
    json.dump(index_payload, f, indent=2)

# ----- write a few atom jsons (demo) -----
# Here we dump just the normalized_mse for each "feature id". You can extend with more fields later.
# figure out how many entries normalized_mse actually has
if hasattr(normalized_mse, "shape"):
    n_items = int(normalized_mse.shape[0])
else:
    n_items = len(normalized_mse)

num_to_dump = min(32, n_items)  # don't loop past the end
for i in range(num_to_dump):
    val = normalized_mse[i]
    val = float(val.item()) if hasattr(val, "item") else float(val)
    atom_payload = {
        "feature_id": i,
        "normalized_mse": val,
    }
    with open(os.path.join(atoms_dir, f"{i}.json"), "w", encoding="utf-8") as f:
        json.dump(atom_payload, f, indent=2)

for i in range(num_to_dump):
    atom_payload = {
        "feature_id": i,
        "normalized_mse": float(normalized_mse[i].item()) if hasattr(normalized_mse[i], "item") else float(normalized_mse[i]),
        # add more fields later if you want (e.g., top_examples, histograms, etc.)
    }
    with open(os.path.join(atoms_dir, f"{i}.json"), "w", encoding="utf-8") as f:
        json.dump(atom_payload, f, indent=2)

print("Wrote viewer files to:", out_dir)

