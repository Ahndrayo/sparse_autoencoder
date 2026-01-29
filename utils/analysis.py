"""Feature analysis utilities for SAE feature tracking and aggregation."""

import numpy as np
import heapq
from typing import List


class FeatureStatsAggregator:
    """Track per-token feature statistics across samples."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.total_tokens = 0
        self.sum_activations = np.zeros(feature_dim, dtype=np.float64)
        self.max_activations = np.zeros(feature_dim, dtype=np.float64)
        self.nonzero_counts = np.zeros(feature_dim, dtype=np.float64)
        self.sum_of_squares = np.zeros(feature_dim, dtype=np.float64)
    
    def update(self, token_activations: np.ndarray):
        """Update with activations from tokens [num_tokens, feature_dim]."""
        self.total_tokens += token_activations.shape[0]
        self.sum_activations += token_activations.sum(axis=0)
        self.max_activations = np.maximum(self.max_activations, token_activations.max(axis=0))
        self.nonzero_counts += (token_activations > 0).sum(axis=0)
        self.sum_of_squares += (token_activations ** 2).sum(axis=0)
    
    def get_stats(self):
        """Get aggregated statistics across all tokens."""
        mean_act = self.sum_activations / max(self.total_tokens, 1)
        frac_active = self.nonzero_counts / max(self.total_tokens, 1)
        mean_act_squared = self.sum_of_squares / max(self.total_tokens, 1)
        return {
            "mean_activation": mean_act,
            "max_activation": self.max_activations,
            "fraction_active": frac_active,
            "mean_act_squared": mean_act_squared
        }


class FeatureTopTokenTracker:
    """Track top activating tokens for each feature."""
    
    def __init__(self, feature_dim: int, top_k: int):
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.heaps = [[] for _ in range(feature_dim)]
    
    def update(self, token_activations: np.ndarray, token_ids: List[int], 
               prompt_idx: int, prompt_text: str, prompt_tokens: List[str],
               predicted_label: str = None, true_label: str = None):
        """Update with tokens from one prompt."""
        for token_pos, (act_vec, token_id) in enumerate(zip(token_activations, token_ids)):
            top_features = np.argsort(act_vec)[-5:]
            for feat_id in top_features:
                activation = float(act_vec[feat_id])
                if activation <= 0:
                    continue
                heap = self.heaps[feat_id]
                token_str = prompt_tokens[token_pos] if token_pos < len(prompt_tokens) else f"[{token_id}]"
                metadata = {
                    "activation": activation,
                    "token_str": token_str,
                    "token_id": int(token_id),
                    "token_position": int(token_pos),
                    "prompt_index": int(prompt_idx),
                    "row_id": int(prompt_idx),
                    "prompt_snippet": prompt_text[:160],
                    "prompt": prompt_text,
                    "prompt_tokens": prompt_tokens,
                    "predicted_label": predicted_label,
                    "true_label": true_label,
                }
                if len(heap) < self.top_k:
                    heapq.heappush(heap, (activation, metadata))
                elif activation > heap[0][0]:
                    heapq.heapreplace(heap, (activation, metadata))
    
    def export(self):
        """Export top tokens for each feature."""
        result = {}
        for feat_id in range(self.feature_dim):
            sorted_tokens = sorted(self.heaps[feat_id], key=lambda x: -x[0])
            result[str(feat_id)] = [meta for _, meta in sorted_tokens]
        return result


class HeadlineFeatureAggregator:
    """Aggregate top features per headline/sample."""
    
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.headlines = []
    
    def add_headline(self, prompt_idx: int, prompt_text: str,
                     token_activations: np.ndarray,
                     token_ids: List[int],
                     token_strings: List[str],
                     predicted_label: str, true_label: str,
                     confidence: float = None):
        """Aggregate features across all tokens in a headline."""
        if token_activations.size == 0:
            return
        max_token_idx_per_feature = token_activations.argmax(axis=0)
        max_activation_per_feature = token_activations.max(axis=0)
        top_feature_ids = np.argsort(max_activation_per_feature)[-self.top_k:][::-1]
        features = [
            {
                "feature_id": int(fid),
                "max_activation": float(max_activation_per_feature[fid]),
                "token_position": int(max_token_idx_per_feature[fid]),
                "token_id": int(token_ids[max_token_idx_per_feature[fid]]),
                "token_str": token_strings[max_token_idx_per_feature[fid]],
            }
            for fid in top_feature_ids if max_activation_per_feature[fid] > 0
        ]
        self.headlines.append({
            "row_id": int(prompt_idx),
            "prompt": prompt_text,
            "predicted_label": predicted_label,
            "confidence": float(confidence) if confidence is not None else None,
            "true_label": true_label,
            "correct": predicted_label == true_label,
            "num_tokens": int(token_activations.shape[0]),
            "features": features
        })
    
    def add_headline_with_ablation_metrics(
        self, 
        prompt_idx: int, 
        prompt_text: str,
        token_activations: np.ndarray,
        token_ids: List[int],
        token_strings: List[str],
        predicted_label: str, 
        true_label: str,
        confidence: float,
        baseline_features: dict,
        features_to_ablate: List[int]
    ):
        """
        Aggregate features with ablation comparison metrics.
        
        Args:
            baseline_features: Dict with 'top_features' (list of {feature_id, activation}) 
                              and 'total_activation' (sum of top-10 activations)
            features_to_ablate: List of feature IDs that were ablated
        """
        if token_activations.size == 0:
            return
        
        # Compute ablated features (existing logic)
        max_token_idx_per_feature = token_activations.argmax(axis=0)
        max_activation_per_feature = token_activations.max(axis=0)
        top_feature_ids = np.argsort(max_activation_per_feature)[-self.top_k:][::-1]
        features = [
            {
                "feature_id": int(fid),
                "max_activation": float(max_activation_per_feature[fid]),
                "token_position": int(max_token_idx_per_feature[fid]),
                "token_id": int(token_ids[max_token_idx_per_feature[fid]]),
                "token_str": token_strings[max_token_idx_per_feature[fid]],
            }
            for fid in top_feature_ids if max_activation_per_feature[fid] > 0
        ]
        
        # Compute ablation metrics
        baseline_top_features = baseline_features.get('top_features', [])
        baseline_feature_ids = {feat['feature_id'] for feat in baseline_top_features}
        ablated_set = set(features_to_ablate)
        
        # Count how many baseline top features were ablated
        num_ablated_features = len(baseline_feature_ids & ablated_set)
        total_baseline_features = len(baseline_top_features)
        
        # Calculate activation fraction ablated
        ablated_activation_sum = sum(
            feat['activation'] 
            for feat in baseline_top_features 
            if feat['feature_id'] in ablated_set
        )
        baseline_total_activation = baseline_features.get('total_activation', 0.0)
        
        if baseline_total_activation > 0:
            ablation_fraction = ablated_activation_sum / baseline_total_activation
        else:
            ablation_fraction = 0.0
        
        self.headlines.append({
            "row_id": int(prompt_idx),
            "prompt": prompt_text,
            "predicted_label": predicted_label,
            "confidence": float(confidence) if confidence is not None else None,
            "true_label": true_label,
            "correct": predicted_label == true_label,
            "num_tokens": int(token_activations.shape[0]),
            "features": features,
            "num_ablated_features": int(num_ablated_features),
            "total_baseline_features": int(total_baseline_features),
            "ablation_fraction": float(ablation_fraction)
        })
    
    def export(self):
        """Export all headline data."""
        return self.headlines


__all__ = [
    "FeatureStatsAggregator",
    "FeatureTopTokenTracker",
    "HeadlineFeatureAggregator"
]
