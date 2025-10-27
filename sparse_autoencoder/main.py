import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder
import os, json
import numpy as np
from utils.run_dirs import make_analysis_run_dir

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


# Where to write files (repo root /viewer/...)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Save data for inpsecting meaning of features
run_dir = make_analysis_run_dir(repo_root)

np.save(run_dir / "latent_activations.npy",
        latent_activations.detach().cpu().numpy())
np.save(run_dir / "tokens.npy",
        tokens.detach().cpu().numpy())

with open(run_dir / "prompt.txt", "w") as f:
    f.write(prompt)

print("Saved analysis data to:", str(run_dir))