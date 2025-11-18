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
parser.add_argument("--kaggle-row", type=int, default=None,
                    help="Optional row index from the Kaggle dataset to process instead of all rows.")
# parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on the number of Kaggle rows to process.")
args = parser.parse_args()

if args.prompt_file:
    prompt_path = Path(args.prompt_file)
    single_prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_entries: list[tuple[str | int, str]] = [("prompt_file", single_prompt)]
    print(f"Loaded prompt from {prompt_path}")
else:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Expected dataset CSV at {dataset_csv} but it was not found.")
    df = pd.read_csv(dataset_csv)
    if KAGGLE_TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{KAGGLE_TEXT_COLUMN}' not found in {dataset_csv.name}")
    kaggle_series = df[KAGGLE_TEXT_COLUMN].dropna().astype(str)
    if args.kaggle_row is not None:
        if args.kaggle_row < 0 or args.kaggle_row >= len(df):
            raise ValueError(f"--kaggle-row must be between 0 and {len(df) - 1}")
        kaggle_series = kaggle_series.iloc[[args.kaggle_row]]
    #if args.max_rows is not None:kaggle_series = kaggle_series.head(args.max_rows)
    prompt_entries = list(kaggle_series.items())
    print(f"Loaded {len(prompt_entries)} prompt(s) from Kaggle dataset ({dataset_csv.name})")

if len(prompt_entries) == 0:
    raise ValueError("No prompts available to process.")

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
    state_dict = torch.load(f, map_location=device)
autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
autoencoder.to(device)
autoencoder.eval()

MAX_SEQUENCE_LENGTH = 64
pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
if pad_token_id is None:
    pad_token_id = getattr(model.tokenizer, "eos_token_id", 50256)

all_latents: list[torch.Tensor] = []
all_tokens: list[torch.Tensor] = []
all_mse: list[torch.Tensor] = []
processed_prompts: list[dict[str, str | int]] = []

for row_idx, text in prompt_entries:
    prompt = text.strip()
    if not prompt:
        continue

    tokens = model.to_tokens(prompt, truncate=False)
    tokens = tokens[:, :MAX_SEQUENCE_LENGTH]
    if tokens.shape[1] == 0:
        continue

    seq_len = tokens.shape[1]
    tokens = tokens.to(device)

    with torch.no_grad():
        _, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
        input_tensor = activation_cache[transformer_lens_loc]
        latent_activations, info = autoencoder.encode(input_tensor)
        reconstructed_activations = autoencoder.decode(latent_activations, info)
        normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (
            (input_tensor).pow(2).sum(dim=1)
        )

    latents_cpu = latent_activations.detach().cpu()
    tokens_cpu = tokens.detach().cpu()
    mse_cpu = normalized_mse.detach().cpu()

    latent_padded = torch.zeros(
        MAX_SEQUENCE_LENGTH, latents_cpu.shape[-1], dtype=latents_cpu.dtype
    )
    latent_padded[:seq_len] = latents_cpu
    token_padded = torch.full((MAX_SEQUENCE_LENGTH,), pad_token_id, dtype=tokens_cpu.dtype)
    token_padded[:seq_len] = tokens_cpu[0]
    mse_padded = torch.zeros(MAX_SEQUENCE_LENGTH, dtype=mse_cpu.dtype)
    mse_padded[:seq_len] = mse_cpu

    all_latents.append(latent_padded)
    all_tokens.append(token_padded)
    all_mse.append(mse_padded)
    processed_prompts.append(
        {
            "row_id": row_idx,
            "seq_len": int(seq_len),
            "prompt": prompt,
        }
    )

if len(all_latents) == 0:
    raise ValueError("No prompts produced valid activations.")

latent_tensor = torch.stack(all_latents)
token_tensor = torch.stack(all_tokens)
mse_tensor = torch.stack(all_mse)

# Where to write files (repo root /viewer/...)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run_dir = make_analysis_run_dir(repo_root)

np.save(run_dir / "latent_activations.npy", latent_tensor.numpy())
np.save(run_dir / "tokens.npy", token_tensor.numpy())
np.save(run_dir / "normalized_mse.npy", mse_tensor.numpy())

with open(run_dir / "prompts.jsonl", "w", encoding="utf-8") as f:
    for entry in processed_prompts:
        json.dump(entry, f)
        f.write("\n")

metadata = {
    "dataset_csv": str(dataset_csv),
    "text_column": KAGGLE_TEXT_COLUMN,
    "layer_index": layer_index,
    "location": location,
    "num_prompts": len(processed_prompts),
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
}
with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved analysis data for {len(processed_prompts)} prompt(s) to: {run_dir}")