# viz_analysis/feature_probe.py
import json
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from utils.run_picker import get_latest_run

# Load saved data
repo_root = Path(__file__).resolve().parent.parent
analysis_dir = repo_root / "analysis_data"
latest_run = get_latest_run(analysis_dir)
print("Loading data from: ", latest_run)

lat_path = latest_run / "latent_activations.npy"
tok_path = latest_run / "tokens.npy"
prompts_jsonl = latest_run / "prompts.jsonl"
prompt_txt = latest_run / "prompt.txt"

lat = np.load(lat_path)
tok_ids = np.load(tok_path)

if prompts_jsonl.exists():
    with open(prompts_jsonl, "r", encoding="utf-8") as f:
        prompt_entries = [json.loads(line) for line in f if line.strip()]
    seq_lens = [int(entry["seq_len"]) for entry in prompt_entries]
else:
    if prompt_txt.exists():
        prompt_text = prompt_txt.read_text(encoding="utf-8").strip()
        prompt_entries = [{"prompt": prompt_text}]
    else:
        raise FileNotFoundError("Expected prompts.jsonl or prompt.txt in the latest run directory.")
    # Derive sequence length from array shapes
    total_tokens = lat.shape[0] if lat.ndim == 2 else tok_ids.shape[-1]
    seq_lens = [int(total_tokens)]

total_tokens = sum(seq_lens)

def flatten_latents(lat_array, seq_lengths):
    if lat_array.ndim == 2 and lat_array.shape[0] == total_tokens:
        return lat_array
    if lat_array.ndim != 3 or lat_array.shape[0] != len(seq_lengths):
        raise ValueError(f"Unexpected latent tensor shape {lat_array.shape} for provided seq lengths.")
    hidden_dim = lat_array.shape[-1]
    flat = np.zeros((total_tokens, hidden_dim), dtype=lat_array.dtype)
    idx = 0
    for prompt_idx, seq_len in enumerate(seq_lengths):
        flat[idx : idx + seq_len] = lat_array[prompt_idx, :seq_len]
        idx += seq_len
    return flat

def flatten_tokens(tok_array, seq_lengths):
    if tok_array.ndim == 1 and tok_array.shape[0] == total_tokens:
        return tok_array
    if tok_array.ndim == 2 and tok_array.shape[0] == len(seq_lengths):
        flat = np.zeros(total_tokens, dtype=tok_array.dtype)
        idx = 0
        for prompt_idx, seq_len in enumerate(seq_lengths):
            flat[idx : idx + seq_len] = tok_array[prompt_idx, :seq_len]
            idx += seq_len
        return flat
    raise ValueError(f"Unexpected token tensor shape {tok_array.shape} for provided seq lengths.")

lat_flat = flatten_latents(lat, seq_lens)
tok_flat = flatten_tokens(tok_ids, seq_lens)

# Rebuild model only to decode tokens (no need for SAE)
model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)

tok_strs: list[str] = []
for entry, seq_len in zip(prompt_entries, seq_lens):
    prompt_text = entry.get("prompt", "")
    prompt_tokens = model.to_str_tokens(prompt_text)
    tok_strs.extend(prompt_tokens[:seq_len])

# Ensure token strings align with flattened arrays
if len(tok_strs) < len(tok_flat):
    tok_strs.extend(["<pad>"] * (len(tok_flat) - len(tok_strs)))
elif len(tok_strs) > len(tok_flat):
    tok_strs = tok_strs[: len(tok_flat)]

# Pick feature and top-k
feat_id = 25906
# feat_id = 3065
top_k = 10

vals = lat_flat[:, feat_id]
top_idx = np.argsort(vals)[::-1][:top_k]

print(f"\nTop {top_k} tokens activating feature {feat_id}:")
for i in top_idx:
    token_display = tok_strs[i] if i < len(tok_strs) else "<unknown>"
    print(f"  Token {i:>3}: '{token_display}'  act={vals[i]:.4f}")
