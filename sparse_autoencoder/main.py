import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder
import os, json
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List
from utils.run_dirs import make_analysis_run_dir
import pandas as pd
import heapq

import kagglehub

# Download latest version (cached by kagglehub, so this is cheap on reruns)
path = kagglehub.dataset_download("miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests")
print("Path to dataset files:", path)
dataset_dir = Path(path)
dataset_csv = dataset_dir / "analyst_ratings_processed.csv"
KAGGLE_TEXT_COLUMN = "title"


@dataclass(frozen=True)
class FeatureMetricDefinition:
    name: str
    description: str
    compute: Callable[["FeatureStatsAggregator"], torch.Tensor]


class FeatureStatsAggregator:
    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.total_tokens: int = 0
        self.sum_activations = torch.zeros(feature_dim, dtype=torch.float64)
        self.max_activations = torch.zeros(feature_dim, dtype=torch.float64)
        self.nonzero_counts = torch.zeros(feature_dim, dtype=torch.float64)
        self.sum_of_squares = torch.zeros(feature_dim, dtype=torch.float64)

    def update(self, latents: torch.Tensor) -> None:
        if latents.numel() == 0:
            return
        latents = latents.to(dtype=torch.float64, copy=False)
        self.total_tokens += latents.shape[0]
        self.sum_activations += latents.sum(dim=0)
        self.max_activations = torch.maximum(self.max_activations, latents.max(dim=0).values)
        self.nonzero_counts += (latents > 0).sum(dim=0).to(torch.float64)
        self.sum_of_squares += (latents ** 2).sum(dim=0)

    def safe_total_tokens(self) -> int:
        return max(self.total_tokens, 1)

    def build_summary(
        self, metric_defs: List[FeatureMetricDefinition], top_k: int
    ) -> dict[str, object]:
        if self.total_tokens == 0:
            raise ValueError("Cannot build summary: no tokens were processed.")

        metric_arrays: Dict[str, np.ndarray] = {}
        for metric in metric_defs:
            values = metric.compute(self)
            metric_arrays[metric.name] = values.detach().cpu().numpy()

        final_top_k = min(top_k, self.feature_dim)
        summary: dict[str, object] = {
            "num_features": self.feature_dim,
            "total_tokens": self.total_tokens,
            "top_feature_count": final_top_k,
            "metrics": {},
        }

        for metric in metric_defs:
            values = metric_arrays[metric.name]
            order = np.argsort(values)[::-1]
            top_indices = order[:final_top_k]
            top_entries: list[dict[str, object]] = []
            for idx in top_indices:
                metrics_snapshot = {
                    name: float(metric_arrays[name][idx]) for name in metric_arrays.keys()
                }
                top_entries.append(
                    {
                        "feature_id": int(idx),
                        "value": float(values[idx]),
                        "metrics": metrics_snapshot,
                    }
                )
            summary["metrics"][metric.name] = {
                "description": metric.description,
                "top_features": top_entries,
            }

        mean_sq = (self.sum_of_squares / self.safe_total_tokens()).detach().cpu().numpy()
        summary["mean_act_squared"] = mean_sq.tolist()

        return summary


class TokenStringCache:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.cache: Dict[int, str] = {}

    def decode(self, token_id: int) -> str:
        token_id = int(token_id)
        if token_id not in self.cache:
            self.cache[token_id] = self.tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
        return self.cache[token_id]


class FeatureTopTokenTracker:
    def __init__(
        self,
        feature_dim: int,
        top_tokens_per_feature: int,
        per_token_feature_scan: int,
        tokenizer,
    ) -> None:
        self.feature_dim = feature_dim
        self.top_tokens_per_feature = top_tokens_per_feature
        self.per_token_feature_scan = per_token_feature_scan
        self.heaps: list[list[tuple[float, dict[str, object]]]] = [
            [] for _ in range(feature_dim)
        ]
        self.token_cache = TokenStringCache(tokenizer)
        self._counter = 0  # <-- tiebreaker

    def _push(self, feature_id: int, activation: float, metadata: dict[str, object]) -> None:
        self._counter += 1
        heap = self.heaps[feature_id]
        # entry: (activation, counter, metadata)
        entry = (activation, self._counter, metadata)
        if len(heap) < self.top_tokens_per_feature:
            heapq.heappush(heap, entry)
        else:
            if activation > heap[0][0]:
                heapq.heappushpop(heap, entry)

    def update(
        self,
        latents: torch.Tensor,
        token_ids: torch.Tensor,
        prompt_index: int,
        row_id: int | str,
        prompt_text: str,
        prompt_tokens: list[str],
    ) -> None:
        if latents.numel() == 0:
            return
        seq_len = latents.shape[0]
        num_features = latents.shape[1]
        k = min(self.per_token_feature_scan, num_features)
        if k <= 0:
            return
        values, indices = torch.topk(latents, k=k, dim=1)
        prompt_snippet = prompt_text[:200]
        for token_pos in range(seq_len):
            token_id = int(token_ids[token_pos])
            token_str = self.token_cache.decode(token_id)
            for j in range(k):
                activation = float(values[token_pos, j])
                if activation <= 0.0:
                    continue
                feature_id = int(indices[token_pos, j])
                metadata = {
                    "activation": activation,
                    "token_str": token_str,
                        "token_id": token_id,
                    "token_position": int(token_pos),
                    "prompt_index": int(prompt_index),
                    "row_id": row_id,
                    "prompt_snippet": prompt_snippet,
                }
                metadata["prompt_tokens"] = prompt_tokens
                self._push(feature_id, activation, metadata)

    def export(self, feature_ids: List[int]) -> dict[str, list[dict[str, object]]]:
        output: dict[str, list[dict[str, object]]] = {}
        for feature_id in feature_ids:
            if feature_id < 0 or feature_id >= self.feature_dim:
                continue
            heap = self.heaps[feature_id]
            if not heap:
                continue
            sorted_entries = sorted(heap, key=lambda x: x[0], reverse=True)
            output[str(feature_id)] = [entry[2] for entry in sorted_entries]
        return output


FEATURE_METRICS: List[FeatureMetricDefinition] = [
    FeatureMetricDefinition(
        name="mean_activation",
        description="Average activation across all tokens.",
        compute=lambda stats: stats.sum_activations / stats.safe_total_tokens(),
    ),
    FeatureMetricDefinition(
        name="max_activation",
        description="Maximum activation observed for each feature.",
        compute=lambda stats: stats.max_activations.clone(),
    ),
    FeatureMetricDefinition(
        name="fraction_active",
        description="Fraction of tokens where the feature activation was > 0.",
        compute=lambda stats: stats.nonzero_counts / stats.safe_total_tokens(),
    ),
]

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
device = next(model.parameters()).device

# Get prompt
parser = argparse.ArgumentParser()
parser.add_argument("--prompt-file", type=str, default=None,
                    help="Path to a text file containing the input prompt.")
parser.add_argument("--kaggle-row", type=int, default=None,
                    help="Optional row index from the Kaggle dataset to process instead of all rows.")
parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on the number of Kaggle rows to process.")
parser.add_argument("--chunk-size", type=int, default=10000, help="Number of rows to process at a time (default: 10000).")
parser.add_argument("--top-feature-count", type=int, default=100,
                    help="Number of top features per metric to store (default: 100).")
parser.add_argument("--top-tokens-per-feature", type=int, default=20,
                    help="Number of top tokens to keep for each feature (default: 20).")
parser.add_argument("--per-token-feature-scan", type=int, default=5,
                    help="How many features to sample per token when tracking top tokens.")
args = parser.parse_args()

if args.prompt_file:
    prompt_path = Path(args.prompt_file)
    single_prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_entries: list[tuple[str | int, str]] = [("prompt_file", single_prompt)]
    print(f"Loaded prompt from {prompt_path}")
else:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Expected dataset CSV at {dataset_csv} but it was not found.")
    
    # Check if single row processing
    if args.kaggle_row is not None:
        df = pd.read_csv(dataset_csv, nrows=args.kaggle_row + 1)
        if KAGGLE_TEXT_COLUMN not in df.columns:
            raise ValueError(f"Column '{KAGGLE_TEXT_COLUMN}' not found in {dataset_csv.name}")
        if args.kaggle_row >= len(df):
            raise ValueError(f"--kaggle-row must be between 0 and {len(df) - 1}")
        kaggle_series = df[KAGGLE_TEXT_COLUMN].dropna().astype(str)
        kaggle_series = kaggle_series.iloc[[args.kaggle_row]]
        prompt_entries = list(kaggle_series.items())
        print(f"Loaded {len(prompt_entries)} prompt(s) from Kaggle dataset ({dataset_csv.name})")
        use_chunking = False
    else:
        # Will process in chunks - set flag
        use_chunking = True
        prompt_entries = None  # Will be processed chunk by chunk

layer_index = 6
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

# Where to write files (repo root /viewer/...)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run_dir = make_analysis_run_dir(repo_root)
feature_dim = autoencoder.encoder.weight.shape[0]
feature_stats = FeatureStatsAggregator(feature_dim)
top_token_tracker = FeatureTopTokenTracker(
    feature_dim,
    args.top_tokens_per_feature,
    args.per_token_feature_scan,
    model.tokenizer,
)


def process_prompt_batch(prompt_entries_batch, next_prompt_index: int):
    """Process a batch of prompts, updating stats and returning processed metadata."""
    batch_processed_prompts: list[dict[str, str | int]] = []
    prompt_index = next_prompt_index

    for row_idx, text in prompt_entries_batch:
        prompt = text.strip()
        if not prompt:
            continue

        tokens = model.to_tokens(prompt, truncate=False)
        tokens = tokens[:, :MAX_SEQUENCE_LENGTH]
        if tokens.shape[1] == 0:
            continue

        seq_len = tokens.shape[1]
        tokens = tokens.to(device)

        with torch.inference_mode():
            _, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
            input_tensor = activation_cache[transformer_lens_loc]
            latent_activations, info = autoencoder.encode(input_tensor)

        latents_cpu = latent_activations.detach().cpu()
        tokens_cpu = tokens.detach().cpu()
        prompt_tokens_list = model.to_str_tokens(prompt)

        feature_stats.update(latents_cpu)
        top_token_tracker.update(
            latents_cpu,
            tokens_cpu[0],
            prompt_index=prompt_index,
            row_id=row_idx,
            prompt_text=prompt,
            prompt_tokens=prompt_tokens_list,
        )

        batch_processed_prompts.append(
            {
                "row_id": row_idx,
                "seq_len": int(seq_len),
                "prompt": prompt,
            }
        )
        prompt_index += 1

    return batch_processed_prompts, prompt_index

# Process prompts
total_processed = 0
next_prompt_index = 0
prompts_file_path = run_dir / "prompts.jsonl"

if use_chunking:
    print(f"Processing dataset of {args.max_rows} in chunks of {args.chunk_size} rows...")
    df_header = pd.read_csv(dataset_csv, nrows=0)
    if KAGGLE_TEXT_COLUMN not in df_header.columns:
        raise ValueError(f"Column '{KAGGLE_TEXT_COLUMN}' not found in {dataset_csv.name}")

    chunk_iter = pd.read_csv(dataset_csv, chunksize=args.chunk_size, usecols=[KAGGLE_TEXT_COLUMN])
    rows_processed = 0

    for chunk_idx, chunk_df in enumerate(chunk_iter):
        if args.max_rows is not None and rows_processed >= args.max_rows:
            break

        if args.max_rows is not None:
            remaining = args.max_rows - rows_processed
            if remaining < len(chunk_df):
                chunk_df = chunk_df.head(remaining)

        kaggle_series = chunk_df[KAGGLE_TEXT_COLUMN].dropna().astype(str)
        prompt_entries_batch = list(kaggle_series.items())
        if len(prompt_entries_batch) == 0:
            continue

        batch_prompts, next_prompt_index = process_prompt_batch(
            prompt_entries_batch, next_prompt_index
        )

        if len(batch_prompts) == 0:
            continue

        with open(prompts_file_path, "a" if total_processed > 0 else "w", encoding="utf-8") as f:
            for entry in batch_prompts:
                json.dump(entry, f)
                f.write("\n")

        batch_count = len(batch_prompts)
        total_processed += batch_count
        rows_processed += len(chunk_df)
        print(f"Processed chunk {chunk_idx + 1}: {batch_count} prompts (total: {total_processed})")

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if args.max_rows is not None and rows_processed >= args.max_rows:
            break

    if total_processed == 0:
        raise ValueError("No prompts produced valid activations.")

else:
    if prompt_entries is None or len(prompt_entries) == 0:
        raise ValueError("No prompts available to process.")

    batch_prompts, next_prompt_index = process_prompt_batch(prompt_entries, next_prompt_index)
    if len(batch_prompts) == 0:
        raise ValueError("No prompts produced valid activations.")

    with open(prompts_file_path, "w", encoding="utf-8") as f:
        for entry in batch_prompts:
            json.dump(entry, f)
            f.write("\n")

    total_processed = len(batch_prompts)

metadata = {
    "dataset_csv": str(dataset_csv),
    "text_column": KAGGLE_TEXT_COLUMN,
    "layer_index": layer_index,
    "location": location,
    "num_prompts": total_processed,
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "top_feature_count": args.top_feature_count,
    "top_tokens_per_feature": args.top_tokens_per_feature,
}
with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

feature_stats_summary = feature_stats.build_summary(FEATURE_METRICS, args.top_feature_count)
with open(run_dir / "feature_stats.json", "w", encoding="utf-8") as f:
    json.dump(feature_stats_summary, f, indent=2)

selected_feature_ids: set[int] = set()
for metric_data in feature_stats_summary["metrics"].values():
    for entry in metric_data["top_features"]:
        selected_feature_ids.add(int(entry["feature_id"]))

feature_tokens = top_token_tracker.export(sorted(selected_feature_ids))
with open(run_dir / "feature_tokens.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "top_tokens_per_feature": args.top_tokens_per_feature,
            "features": feature_tokens,
        },
        f,
        indent=2,
    )

print(f"Saved analysis data for {total_processed} prompt(s) to: {run_dir}")