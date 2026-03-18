"""Feature analysis utilities for SAE feature tracking and aggregation."""

import json
import random
from collections import Counter
from pathlib import Path

import heapq
import numpy as np
from typing import Dict, List, Optional, Sequence, Set, Union


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


class InterpretabilityTokenRecorder:
    """
    Per-token (activation, snippet) traces for selected SAE features during baseline inference.
    Uses reservoir sampling per feature to cap memory. Export format matches SAEInterpretabilityEvaluator.
    """

    def __init__(
        self,
        feature_ids: Sequence[int],
        max_rows_per_feature: int = 50_000,
        context_radius: int = 3,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.feature_ids: Set[int] = {int(x) for x in feature_ids}
        self.max_rows = int(max_rows_per_feature)
        self.context_radius = int(context_radius)
        self._reservoir: Dict[int, List[tuple]] = {fid: [] for fid in self.feature_ids}
        self._seen: Dict[int, int] = {fid: 0 for fid in self.feature_ids}

    def _make_snippet(self, tokens: List[str], pos: int) -> str:
        r = self.context_radius
        lo = max(0, pos - r)
        hi = min(len(tokens), pos + r + 1)
        return " ".join(tokens[lo:hi])

    def update(self, sae_features_filtered: np.ndarray, filtered_prompt_tokens: List[str]) -> None:
        """Append reservoir samples for each tracked feature. sae_features_filtered: [T, latent_dim]."""
        if not self.feature_ids or sae_features_filtered.size == 0:
            return
        ntok = sae_features_filtered.shape[0]
        for t in range(ntok):
            for fid in self.feature_ids:
                if fid >= sae_features_filtered.shape[1]:
                    continue
                self._seen[fid] += 1
                k = self._seen[fid]
                act = float(sae_features_filtered[t, fid])
                snippet = self._make_snippet(filtered_prompt_tokens, t)
                center_tok = filtered_prompt_tokens[t] if t < len(filtered_prompt_tokens) else ""
                bucket = self._reservoir[fid]
                row = (act, snippet, center_tok)
                if len(bucket) < self.max_rows:
                    bucket.append(row)
                else:
                    j = random.randint(1, k)
                    if j <= self.max_rows:
                        bucket[random.randint(0, self.max_rows - 1)] = row

    def export_dict(self) -> Dict[str, dict]:
        """Build JSON-serializable payload: feature_id -> activations, snippets, top_token_str."""
        out: Dict[str, dict] = {}
        for fid, bucket in self._reservoir.items():
            if not bucket:
                continue
            acts = np.array([b[0] for b in bucket], dtype=np.float64)
            snippets = [b[1] for b in bucket]
            center_tokens = [b[2] for b in bucket]
            pos = acts > 0
            if pos.any():
                thr = float(np.percentile(acts[pos], 90))
            else:
                thr = 0.0
            top_toks = [tok for a, tok in zip(acts, center_tokens) if a >= thr and tok]
            if top_toks:
                top_token_str = Counter(top_toks).most_common(1)[0][0]
            else:
                top_token_str = ""
            out[str(fid)] = {
                "activations": acts.tolist(),
                "snippets": snippets,
                "top_token_str": top_token_str,
            }
        return out

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_dict(), f, indent=2)


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
        features_to_ablate: List[int],
        baseline_prediction: str,
        baseline_confidence: float
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
        
        # Compute transition and confidence delta
        baseline_correct = baseline_prediction == true_label
        ablated_correct = predicted_label == true_label
        if baseline_correct and not ablated_correct:
            transition = "C -> W"
        elif not baseline_correct and ablated_correct:
            transition = "W -> C"
        else:
            transition = None
        confidence_delta = float(confidence) - float(baseline_confidence)

        # Compute ablation metrics
        baseline_top_features = baseline_features.get('top_features', [])
        baseline_feature_ids = {feat['feature_id'] for feat in baseline_top_features}
        ablated_set = set(features_to_ablate or [])
        
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
            "baseline_prediction": baseline_prediction,
            "baseline_confidence": float(baseline_confidence) if baseline_confidence is not None else None,
            "confidence_delta": float(confidence_delta),
            "transition": transition,
            "true_label": true_label,
            "correct": ablated_correct,
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
    "HeadlineFeatureAggregator",
    "InterpretabilityTokenRecorder",
]
