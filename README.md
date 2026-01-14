# FinBERT Sparse Autoencoder Explorer

This repository trains sparse autoencoders (SAEs) on FinBERT activations for sentiment analysis and provides an interactive viewer to explore feature activations at both the feature and headline level.

## Overview

- **Model**: Fine-tuned FinBERT for financial sentiment classification (Bearish/Bullish/Neutral)
- **SAE Training**: Multiple SAE sizes (4k, 8k, 16k, 32k features) trained on FinBERT layer activations
- **Dual Viewer**: Browse by headlines (see which features fire per prediction) or by features (see which tokens activate each feature)
- **Dataset**: Twitter Financial News Sentiment dataset

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PyTorch with CUDA support must be installed separately for GPU acceleration:

```bash
# For CUDA 11.8 (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check available CUDA versions at: https://pytorch.org/get-started/locally/

### 2. Install React Viewer Dependencies

```bash
cd sae-viewer
npm install
```

## Workflow

### 1. Train Sparse Autoencoders

Open `Sentiment_Classification.ipynb` and run the SAE training cell. This will:
- Fine-tune FinBERT on the financial sentiment dataset
- Extract layer-8 activations from all training samples
- Train SAEs with 4 different sizes: **4k, 8k, 16k, and 32k** features
- Save each model to `./finbert_sae/layer_8_Xk.pt`

Training parameters:
```python
LAYER_TO_EXTRACT = 8         # Which BERT layer to extract
LATENT_DIMS = [4096, 8192, 16384, 32768]  # SAE feature counts
L1_COEFFICIENT = 1e-3         # Sparsity penalty
NUM_EPOCHS = 3                # Training epochs
```

**Sparsity results** (layer 8, 32k SAE):
- ~3.8% of features active per token (~1,250 features)
- But only top 10-20 features have strong activations

### 2. Run Inference and Generate Feature Data

In the same notebook, run the **inference cell** to:
- Load a trained SAE (choose size: `SAE_SIZE = "32k"`)
- Process validation samples through FinBERT + SAE
- Generate three data files:
  - `feature_stats.json` - Statistics for all features
  - `feature_tokens.json` - Top 20 token examples for top 100 features per metric
  - `headline_features.json` - Top 10 features per headline with activating tokens
- Save to `./analysis_data/<timestamp>_run-XXX/`

Key parameters:
```python
SAE_SIZE = "32k"              # Which SAE to use: "4k", "8k", "16k", or "32k"
MAX_SAMPLES = 100             # Number of validation samples to process
TOP_FEATURES = 100            # Top features to save detailed token data for
TOP_TOKENS_PER_FEATURE = 20   # Token examples per feature
```

### 3. Start the Backend Server

On a terminal, run the following command:
```bash
python viz_analysis/feature_probe_server.py --host 127.0.0.1 --port 8765
```

The server automatically loads the latest run from `analysis_data/` and exposes:
- `GET /api/headlines` - List all headlines with top features
- `GET /api/features` - Top features by metric
- `GET /api/feature_info?id=<id>` - Detailed token activations for a feature

Optional: Load a specific run:
```bash
python viz_analysis/feature_probe_server.py --run-id 34  # Loads run-034
```

### 4. Launch the Viewer

On a separate terminal from the server, run the following command:
```bash
cd sae-viewer
npm start
```

Opens at `http://localhost:1234` with two tabs:

#### **Headlines Tab** (Default)
- View all processed headlines
- See prediction vs. true label (green = correct, red = wrong)
- Top 3 features shown as chips: `"Feature 1234: surge"` 
- Hover over chips to see activation values
- Click "Show All" to see all 10 features
- Click any feature chip to jump to Features tab

#### **Features Tab**
- Browse top 100 features by metric (mean activation, max activation, fraction active)
- Search by feature ID using the search box
- View top 20 token activations for each feature with context
- See predictions and labels for each example

## Understanding Feature Tracking

### Three Levels of Data

1. **Statistics for ALL features** (all 32k)
   - Mean activation, max activation, fraction active
   - Computed but not saved per-token

2. **Top 100 features per metric** (300 unique features)
   - Saves 20 top-activating token examples each
   - Viewable in Features tab

3. **Top 10 features per headline** (per sample)
   - Which features fired strongest for each prediction
   - Shows the max-activating token per feature
   - Viewable in Headlines tab

### Sparsity Breakdown

With the 32k SAE at 3.82% sparsity:
- ~1,250 features have non-zero activation per token
- Most are weak (< 0.1 activation)
- Only ~10-50 are meaningfully strong (> 1.0)
- Top 5-10 dominate (> 5.0)

## Files Structure

```
.
├── Sentiment_Classification.ipynb   # Main notebook: training + inference
├── finbert_sae/                     # Saved SAE models
│   ├── layer_8_4k.pt
│   ├── layer_8_8k.pt
│   ├── layer_8_16k.pt
│   └── layer_8_32k.pt
├── analysis_data/                   # Inference results
│   └── <timestamp>_run-XXX/
│       ├── feature_stats.json       # All features' statistics
│       ├── feature_tokens.json      # Top 100 features' token examples
│       ├── headline_features.json   # Per-headline top features
│       ├── prompts.jsonl            # Sample metadata
│       └── metadata.json            # Run configuration
├── viz_analysis/
│   └── feature_probe_server.py      # Backend API server
└── sae-viewer/                      # React viewer
    └── src/
        ├── App.tsx                  # Tab router
        ├── components/
        │   ├── HeadlinesView.tsx    # Headlines tab
        │   └── FeaturesView.tsx     # Features tab
        └── interpAPI.ts             # API client
```

## Switching Between SAE Sizes

To compare different SAE sizes, just change `SAE_SIZE` in the inference cell:

```python
SAE_SIZE = "4k"   # 4096 features, ~30% sparsity
SAE_SIZE = "8k"   # 8192 features, ~15% sparsity
SAE_SIZE = "16k"  # 16384 features, ~7.7% sparsity
SAE_SIZE = "32k"  # 32768 features, ~3.8% sparsity (most granular)
```

Then rerun inference, restart the server, and refresh the viewer.

## Customization

### Add More Metrics
Edit the inference cell to add custom feature metrics (e.g., entropy, kurtosis) - they'll appear automatically in the viewer's metric dropdown.

### Increase Feature Coverage
Change `TOP_FEATURES = 100` to track more features with token examples (increases storage).

### Process More Samples
Change `MAX_SAMPLES = 100` to analyze more validation examples (increases compute time).

## Notes

- Only the top 100 features per metric have detailed token examples
- All 32k features have summary statistics and can be searched
- Each run creates a new timestamped folder in `analysis_data/`
- The viewer always loads the latest run unless `--run-id` is specified
- GPU with CUDA strongly recommended for SAE training and FinBERT finetuning

## References

- Original SAE viewer from OpenAI: [sae-viewer README](./sae-viewer/README.md)
- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Dataset: [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
