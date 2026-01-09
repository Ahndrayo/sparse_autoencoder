# FinBERT + Sparse Autoencoder Guide

This guide explains how to use the modified codebase with FinancialBERT models instead of GPT-2.

## Overview

The codebase now supports two modes:
1. **GPT-2 mode** (original): Uses transformer_lens + OpenAI's pretrained SAE
2. **FinBERT mode** (new): Uses your fine-tuned FinancialBERT + custom-trained SAE

## Step-by-Step Workflow

### 1. Fine-tune FinancialBERT (if not done already)

Run Cell 2 in `Sentiment_Classification.ipynb`:
```python
# This trains FinancialBERT on the Twitter sentiment dataset
# Saves to: ./finbert_twitter_ft/best/
```

### 2. Train Sparse Autoencoder on FinBERT

Run Cell 3 (newly added) in `Sentiment_Classification.ipynb`:
```python
# This trains an SAE on FinBERT's layer 6 activations
# Saves to: ./finbert_sae/layer_6_32k.pt
```

**Training configuration:**
- Input dimension: 768 (BERT hidden size)
- Latent dimension: 32,768 sparse features
- L1 coefficient: 0.001 (sparsity penalty)
- Epochs: 10
- Expected sparsity: ~5-15% of features active per token

**What the SAE learns:**
The SAE decomposes BERT's 768-dim activations into ~32k interpretable features. Each feature should correspond to a semantic concept (e.g., "mentions stock price", "negative sentiment words", "financial terms").

### 3. Run Analysis with FinBERT + SAE

Use `main.py` with the `--use-bert` flag:

```bash
python sparse_autoencoder/main.py \
  --use-bert \
  --bert-model ./finbert_twitter_ft/best \
  --sae-path ./finbert_sae/layer_6_32k.pt \
  --max-rows 100 \
  --chunk-size 50 \
  --top-feature-count 100 \
  --top-tokens-per-feature 20
```

**Arguments:**
- `--use-bert`: Enable FinBERT mode (required)
- `--bert-model`: Path to your fine-tuned FinBERT
- `--sae-path`: Path to your trained SAE checkpoint
- `--bert-layer`: Which BERT layer to extract (default: 6)
- Other args same as before

### 4. View Results

Start the viewer as usual:

```bash
# Terminal 1: Start backend
python viz_analysis/feature_probe_server.py

# Terminal 2: Start frontend
cd sae-viewer
npm start
```

Open `http://localhost:1234` in your browser.

## Differences from GPT-2 Mode

### Tokenization
- **GPT-2**: Byte-level BPE (tokens like " hello", "Ġworld")
- **FinBERT**: WordPiece (tokens like "[CLS]", "hello", "##world", "[SEP]")

The frontend now filters out special tokens and handles both properly.

### Architecture
- **GPT-2**: Decoder-only, causal attention
- **FinBERT**: Encoder-only, bidirectional attention

### Features
- **GPT-2 SAE**: Trained on general language
- **FinBERT SAE**: Trained on financial sentiment → features should be domain-specific

## Comparing GPT-2 vs FinBERT

You can run both modes and compare:

```bash
# GPT-2 mode (default)
python sparse_autoencoder/main.py --max-rows 100

# FinBERT mode
python sparse_autoencoder/main.py --use-bert --max-rows 100
```

Each creates a timestamped run in `analysis_data/`. The viewer loads the latest by default.

## Troubleshooting

### "SAE checkpoint not found"
- Make sure you ran Cell 3 in the notebook to train the SAE
- Check that `./finbert_sae/layer_6_32k.pt` exists

### "Model not found"
- Make sure you ran Cell 2 to fine-tune FinBERT
- Check that `./finbert_twitter_ft/best/` exists

### High reconstruction error
If the SAE has high error (>20% normalized MSE):
- Train for more epochs
- Adjust L1 coefficient (lower = less sparse, better reconstruction)
- Try different BERT layers

### Viewer shows wrong tokens
- The frontend was updated to handle BERT tokens
- If you still see issues, make sure you've refreshed the browser
- The viewer now shows the original prompt text instead of reconstructed tokens

## Advanced: Training Better SAEs

To improve SAE quality:

1. **More training data**: Use full training set instead of subset
2. **Tune L1 coefficient**: Balance sparsity vs reconstruction
   - Higher L1 → sparser but worse reconstruction
   - Lower L1 → better reconstruction but less sparse
3. **Different layers**: Try layers 3-9 (middle layers often best)
4. **Larger latent dim**: Try 65k features for more granularity
5. **Learning rate schedule**: Add cosine annealing for better convergence

Example modified training config:
```python
LATENT_DIM = 65536  # More features
L1_COEFFICIENT = 5e-4  # Less sparsity penalty
NUM_EPOCHS = 20  # More training
LEARNING_RATE = 1e-3
```

## Next Steps

1. Analyze which features activate for different sentiment classes
2. Find "steering vectors" by averaging feature activations
3. Ablate features to see impact on predictions
4. Compare feature activations between BERT and GPT-2

Happy analyzing! 🎉

