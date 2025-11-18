import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder
import os, json
import numpy as np
import argparse
from pathlib import Path
from utils.run_dirs import make_analysis_run_dir
import pandas as pd

import kagglehub

# Download latest version (cached by kagglehub, so this is cheap on reruns)
path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
print("Path to dataset files:", path)
dataset_dir = Path(path)
dataset_csv = dataset_dir / "analyst_ratings_processed.csv"
KAGGLE_TEXT_COLUMN = "title"

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

# Get prompt
parser = argparse.ArgumentParser()
parser.add_argument("--prompt-file", type=str, default=None,
                    help="Path to a text file containing the input prompt.")
parser.add_argument("--kaggle-row", type=int, default=0,
                    help="Row index from the Kaggle dataset to use when no prompt file is provided.")
args = parser.parse_args()

if args.prompt_file:
    prompt_path = Path(args.prompt_file)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    print(f"Loaded prompt from {prompt_path}")
else:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Expected dataset CSV at {dataset_csv} but it was not found.")
    df = pd.read_csv(dataset_csv)
    if KAGGLE_TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{KAGGLE_TEXT_COLUMN}' not found in {dataset_csv.name}")
    if args.kaggle_row < 0 or args.kaggle_row >= len(df):
        raise ValueError(f"--kaggle-row must be between 0 and {len(df) - 1}")
    prompt = str(df.iloc[args.kaggle_row][KAGGLE_TEXT_COLUMN]).strip()
    print(f"Loaded prompt from Kaggle dataset row {args.kaggle_row} ({dataset_csv.name})")

tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

layer_index = 8
location = "resid_post_mlp"
# location = "mlp-post-act"

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