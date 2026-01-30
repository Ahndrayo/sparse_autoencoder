"""FinBERT-specific Sparse Autoencoder implementation."""

import torch
import torch.nn as nn
from pathlib import Path


class SparseAutoencoder(nn.Module):
    """
    Simplified Sparse Autoencoder for FinBERT (compatible with OpenAI's architecture).
    
    Encodes 768-dimensional BERT activations into a sparse latent space (e.g., 32k dimensions).
    """
    
    def __init__(self, input_dim=768, latent_dim=32768):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input -> latent
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        
        # Decoder: latent -> reconstruction
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
        # Initialize decoder with unit norm columns (standard for SAEs)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def encode(self, x):
        """Encode to sparse latent representation."""
        latent = self.encoder(x)
        latent = nn.functional.relu(latent)  # ReLU for sparsity
        return latent
    
    def decode(self, latent):
        """Decode from latent representation."""
        return self.decoder(latent)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def get_feature_activations(self, x):
        """Get sparse feature activations (for analysis)."""
        with torch.no_grad():
            return self.encode(x)


def load_sae(layer: int = 8, latent_size: str = "32k", device: torch.device = None):
    """
    Load a trained SAE model.
    
    Args:
        layer: The layer number (default: 8)
        latent_size: Size of latent dimension as string: "4k", "8k", "16k", or "32k"
        device: Device to load model to (default: uses CUDA if available)
    
    Returns:
        sae: The loaded SAE model
        config: Configuration dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sae_path = f"./finbert_sae/layer_{layer}_{latent_size}.pt"
    
    # Load checkpoint
    checkpoint = torch.load(sae_path, map_location=device)
    
    # Create SAE model
    config = checkpoint['config']
    sae = SparseAutoencoder(input_dim=config['input_dim'], latent_dim=config['latent_dim'])
    
    # Load weights
    sae.encoder.weight.data = checkpoint['encoder_weight']
    sae.encoder.bias.data = checkpoint['encoder_bias']
    sae.decoder.weight.data = checkpoint['decoder_weight']
    sae.decoder.bias.data = checkpoint['decoder_bias']
    
    sae.to(device)
    sae.eval()
    
    print(f"✓ Loaded SAE from {sae_path}")
    print(f"  Layer: {config['layer']}")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Latent dim: {config['latent_dim']}")
    
    return sae, config


__all__ = ["SparseAutoencoder", "load_sae"]
