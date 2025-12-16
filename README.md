# sparse_autoencoder_openai

This repo extracts sparse autoencoder activations from GPT-2, saves lightweight summaries (top features plus their highest-activation tokens), and exposes a local React viewer (`sae-viewer/`) that visualizes those summaries instead of relying on the hosted OpenAI viewer.

## Workflow

### 1. Install dependencies

- Install the Python requirements listed in `pyproject.toml`:
  ```bash
  python -m pip install -r requirements.txt
  ```
- Install the React viewer dependencies (Parcel + Tailwind) in `sae-viewer/`:
  ```bash
  cd sae-viewer
  npm install
  ```

### 2. Generate top-feature summaries

```bash
python sparse_autoencoder/main.py --max-rows 100 --chunk-size 50 --top-feature-count 100 --top-tokens-per-feature 20
```

Flags:

- `--max-rows`: how many Kaggle rows to process.  
- `--chunk-size`: how many rows to load per chunk (keeps memory usage low).  
- `--top-feature-count`: how many features per metric to persist in `feature_stats.json`.  
- `--top-tokens-per-feature`: how many token activations per feature to store in `feature_tokens.json`.

Outputs are placed under `analysis_data/<timestamp>_run-XXX/` (metadata, prompts, feature stats/tokens).

### 3. Serve the summaries locally

```bash
python viz_analysis/feature_probe_server.py --host 127.0.0.1 --port 8765
```

The server loads the newest run and exposes:

- `GET /api/features`: the saved features per metric (means/maxes/fraction > 0).  
- `GET /api/feature_info?id=<feature_id>`: a `FeatureInfo` payload (density, histogram, top/random sequences) that the React viewer renders.

It prints the run directory it loaded so you can confirm which dataset you are visualizing.

### 4. Launch the SAE viewer

```bash
cd sae-viewer
npm start
```

Parcel serves the UI (default `http://localhost:1234`). The React app fetches from the Python server, lets you switch metrics, and shows each feature’s histogram + token snippets (via the re-used `FeatureInfo` component).

### 5. Notes

- Only saved features are available in the viewer. Re-run `main.py` with larger limits to include more neurons.  
- Add your own ranking metrics in `FEATURE_METRICS` inside `sparse_autoencoder/main.py`; they will appear automatically in the viewer’s metric dropdown.  
- Legacy helpers in `viz_analysis/` (`feature_probe.py`, `viz.py`, etc.) remain if you want to switch back to storing the full `.npy` activations.
- Monitor storage in `analysis_data/`—each run adds a timestamped folder.

## Additional references

- The original SAE viewer is documented in `sae-viewer/README.md` and hosted publicly [here](https://openaipublic.blob.core.windows.net/sparse-autoencoder/sae-viewer/index.html).  
- Explore `sparse_autoencoder/model.py`, `train.py`, and `paths.py` for details on the autoencoder architecture and available checkpoints.
# sparse_autoencoder_openai

This repo extracts sparse autoencoder activations from the GPT‑2 transformer, persists the results, and ships a lightweight React viewer to explore the top features and tokens.  
Use it to run on a subset of the pre-downloaded Kaggle dataset and then inspect the saved features locally.

## Workflow

### 1. Install dependencies

- Python dependencies are defined in `pyproject.toml`. Use poetry/pip to install them, e.g.
  ```bash
  python -m pip install -r requirements.txt
  ```
  (If `requirements.txt` is missing, build it from `pyproject.toml` or use `poetry install`.)
- The React viewer lives under `viz_analysis/`; it uses only vanilla JS/React shipped via CDN, so no extra build step is required.

### 2. Run `main.py` on a small slice

`main.py` downloads the Kaggle dataset via `kagglehub`, tokenizes prompts, runs them through the sparse autoencoder, and writes several summary files (`metadata.json`, `prompts.jsonl`, `feature_stats.json`, `feature_tokens.json`). To limit memory/disk use when you are experimenting:

```bash
python sparse_autoencoder/main.py --max-rows 100 --chunk-size 50 --top-feature-count 100 --top-tokens-per-feature 10
```

Key flags:

- `--max-rows`: cap the number of dataset rows processed.  
- `--chunk-size`: how many rows are loaded per chunk (keeps RAM small).  
- `--top-feature-count`: only the top `N` features per metric are saved to `feature_stats.json`.  
- `--top-tokens-per-feature`: how many token activations are cached per feature in `feature_tokens.json`.

The script writes run outputs to `analysis_data/<timestamp>_run-XXX/`. Each run writes metadata plus the trimmed summaries above; it no longer keeps the huge `latent_activations.npy`.

### 3. Explore with the React viewer

The viewer reads the latest run folder and exposes:

- `/api/features`: lists the saved top features per metric.  
- `/api/feature?id=...`: returns the saved tokens/snippets for that feature.  
- A static React UI (`feature_probe_frontend.html` + `feature_probe_app.js`) that lets you choose the metric, browse features, and drill into tokens.

Run it with:

```bash
python viz_analysis/feature_probe_server.py --host 127.0.0.1 --port 8765
```

The server prints the run folder it loads, so you can confirm which summary you are viewing. Open `http://127.0.0.1:8765/` in your browser, pick a metric, click feature rows, and the token list will update.

### 4. Legacy scripts / troubleshooting

- `viz_analysis/feature_probe.py` and `viz_analysis/viz.py` still exist for earlier workflows that relied on the raw numpy arrays; they can be used if you ever rerun `main.py` in “full save” mode again.  
- If you need to regenerate the feature summaries with different metrics, add them to `FEATURE_METRICS` in `sparse_autoencoder/main.py`.
- Monitor disk usage in `analysis_data/`; each run creates a timestamped directory with the saved JSON files.

## Summary
1. Run `main.py` with small arguments (e.g., `--max-rows 100`) to generate summaries without huge `.npy` files.  
2. Start the viewer via `feature_probe_server.py` and browse the React UI at `http://localhost:8765/`.  
3. Adjust metrics/limits as you need; rerun `main.py` if you want a fresh summary.  

Let me know if you want a docker script or automatic dataset download/update workflow.
# Sparse autoencoders

This repository hosts:
- sparse autoencoders trained on the GPT2-small model's activations.
- a visualizer for the autoencoders' features

### Install

```sh
pip install git+https://github.com/openai/sparse_autoencoder.git
```

### Code structure

See [sae-viewer](./sae-viewer/README.md) to see the visualizer code, hosted publicly [here](https://openaipublic.blob.core.windows.net/sparse-autoencoder/sae-viewer/index.html).

See [model.py](./sparse_autoencoder/model.py) for details on the autoencoder model architecture.
See [train.py](./sparse_autoencoder/train.py) for autoencoder training code.
See [paths.py](./sparse_autoencoder/paths.py) for more details on the available autoencoders.

### Example usage

```py
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

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
print(location, normalized_mse)
```
