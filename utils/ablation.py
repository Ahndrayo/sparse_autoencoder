"""Ablation experiment utilities for SAE feature intervention."""

import torch
from typing import List, Dict


def create_intervention_hook(sae, features_to_ablate: List[int], device, current_sample_data: Dict = None):
    """
    Create a hook that intercepts layer outputs, encodes through SAE, ablates features, and decodes back.
    
    Args:
        sae: Sparse autoencoder model
        features_to_ablate: List of feature IDs to zero out
        device: Device to use for computations
        current_sample_data: Optional dictionary to store SAE features for tracking
        
    Returns:
        intervention_hook: Function that can be registered as a forward hook
    """
    
    def intervention_hook(module, input_tuple, output):
        """
        Hook function that modifies the output of the target layer.
        This replaces the normal forward pass with: encode -> ablate -> decode
        """
        if isinstance(output, tuple):
            hidden_states = output[0]  # BERT outputs tuple (hidden_states, ...)
        else:
            hidden_states = output
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape to [batch * seq_len, hidden_dim] for SAE processing
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # Encode through SAE: [batch*seq_len, latent_dim]
        sae_features = sae.encode(hidden_flat)
        
        # Store SAE features for tracking (before ablation, but we'll track after)
        # We'll store the ablated version for consistency
        if current_sample_data is not None:
            sae_features_ablated = sae_features.clone()
            sae_features_ablated[:, features_to_ablate] = 0.0
            
            # Store for later tracking (will be processed after forward pass)
            if batch_size == 1:  # Single sample
                current_sample_data["sae_features"] = sae_features_ablated.detach()
        
        # Zero out ablated features
        sae_features[:, features_to_ablate] = 0.0
        
        # Decode back to activation space: [batch*seq_len, hidden_dim]
        modified_activations = sae.decode(sae_features)
        
        # Reshape back to [batch, seq_len, hidden_dim]
        modified_hidden = modified_activations.view(batch_size, seq_len, hidden_dim)
        
        # Return modified output (preserve tuple structure if original was tuple)
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        else:
            return modified_hidden
    
    return intervention_hook


__all__ = ["create_intervention_hook"]
