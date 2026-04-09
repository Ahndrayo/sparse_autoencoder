"""Ablation experiment utilities for SAE feature intervention."""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sklearn.model_selection import StratifiedKFold

# How many top SAE latents (by max activation over tokens) to store per headline in baseline_headlines.
HEADLINE_SAE_TOP_K = 64


def top_sae_latents_by_headline(max_activations_per_feature: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Per headline: max activation per latent over sequence positions; return top k, descending."""
    vec = np.asarray(max_activations_per_feature, dtype=np.float64)
    if vec.size == 0:
        return []
    k = min(int(k), vec.shape[0])
    top_idx = np.argsort(vec)[-k:][::-1]
    return [{"feature_id": int(fid), "activation": float(vec[fid])} for fid in top_idx]


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


def validate_feature_ids(feature_ids: List[int], latent_dim: int, context: str):
    """Validate that feature IDs are within valid range."""
    if not feature_ids:
        return
    invalid = [fid for fid in feature_ids if fid < 0 or fid >= latent_dim]
    if invalid:
        raise ValueError(f"Invalid feature IDs in {context} (0-{latent_dim - 1}): {invalid}")


def normalize_decoder_weights(sae, device: torch.device):
    """Normalize SAE decoder weight columns for cosine similarity."""
    decoder = sae.decoder.weight.detach()
    if decoder.device != device:
        decoder = decoder.to(device)
    norms = torch.norm(decoder, dim=0, keepdim=True).clamp_min(1e-8)
    return decoder / norms


def expand_features_with_similarity(
    feature_ids: List[int],
    normalized_decoder: torch.Tensor,
    top_m: int,
    cache: Dict[int, List[int]],
) -> List[int]:
    """Expand features using cosine similarity between decoder columns."""
    if not feature_ids:
        return []
    top_m = min(int(top_m), normalized_decoder.shape[1])
    expanded = set()
    for fid in feature_ids:
        fid = int(fid)
        if fid in cache and len(cache[fid]) >= top_m:
            similar = cache[fid][:top_m]
        else:
            seed_vec = normalized_decoder[:, fid]
            sims = torch.matmul(seed_vec, normalized_decoder)
            topk = torch.topk(sims, k=top_m).indices.tolist()
            cache[fid] = topk
            similar = topk
        expanded.update(similar)
    return sorted(expanded)


def inference_indices_random(dataset_len: int, max_samples: int, seed: int) -> List[int]:
    """Deterministic subset of row indices: same (len, max_samples, seed) always returns the same list."""
    n = int(dataset_len)
    k = min(int(max_samples), n)
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.choice(n, size=k, replace=False)]


def resolve_inference_sample_indices(
    dataset_len: int,
    max_samples: int,
    sample_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Build ordered list of dataset row indices for inference. None => first min(max_samples, len) rows."""
    n = int(dataset_len)
    if sample_indices is None:
        return list(range(min(int(max_samples), n)))
    out = [int(i) for i in sample_indices]
    seen = set()
    for i in out:
        if i < 0 or i >= n:
            raise ValueError(f"sample index {i} out of range for dataset length {n}")
        if i in seen:
            raise ValueError("sample_indices must be unique")
        seen.add(i)
    return out


def stratified_k_fold_indices(
    labels: Sequence[Union[int, float]],
    n_splits: int,
    *,
    shuffle: bool = True,
    random_state: int,
) -> List[List[int]]:
    """Stratified K-fold row indices. Reproducible for fixed (labels, n_splits, shuffle, random_state)."""
    y = np.asarray(labels)
    n = int(y.shape[0])
    k = int(n_splits)
    if k < 2:
        raise ValueError("n_splits must be >= 2")
    if n < k:
        raise ValueError(f"Need at least n_splits={k} samples, got n={n}")
    X = np.zeros((n, 1), dtype=np.float32)
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    return [[int(i) for i in test_idx] for _, test_idx in skf.split(X, y)]


def select_features_to_ablate(
    ablation_config: Dict[str, Any],
    baseline_features_map: Dict[int, Any],
    sae_latent_dim: int,
    manual_features: List[int],
    similarity_enabled: bool,
    normalized_decoder: Optional[torch.Tensor],
    similarity_top_m: int,
    similarity_cache: Dict[int, List[int]],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Global feature list after baseline (same logic as the notebook Feature Selection cell)."""
    mode = ablation_config["mode"]
    if mode == "manual":
        features = list(manual_features)
        original = list(manual_features)
        validate_feature_ids(features, sae_latent_dim, "manual features")
        if similarity_enabled:
            if normalized_decoder is None:
                raise ValueError("similarity_enabled requires normalized_decoder")
            features = expand_features_with_similarity(
                features, normalized_decoder, similarity_top_m, similarity_cache
            )
        return features, original
    if mode == "union_top_k":
        feature_set = set()
        kk = int(ablation_config["k"])
        for pos in baseline_features_map:
            top_k_ids = [
                f["feature_id"] for f in baseline_features_map[pos]["top_features"][:kk]
            ]
            feature_set.update(top_k_ids)
        features = sorted(feature_set)
        validate_feature_ids(features, sae_latent_dim, "union_top_k features")
        if similarity_enabled:
            if normalized_decoder is None:
                raise ValueError("similarity_enabled requires normalized_decoder")
            features = expand_features_with_similarity(
                features, normalized_decoder, similarity_top_m, similarity_cache
            )
        return features, None
    if mode == "per_sample_top_k":
        return None, None
    raise ValueError(f"Unknown ablation mode: {mode}")


def run_baseline_inference(
    model,
    tokenizer,
    test_ds,
    device: torch.device,
    sae,
    layer_to_extract: int,
    max_samples: int,
    max_seq_length: int,
    interpretability_recorder=None,
    feature_stats_baseline=None,
    top_token_tracker_baseline=None,
    sample_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict], Dict, float, List[Dict[str, Any]], np.ndarray]:
    """Run baseline inference without ablation.

    Returns:
        baseline_sae_max_full: float32 array of shape (n_samples, latent_dim). Per sample, each
        entry is max SAE activation over non-special tokens for that latent (same as top_sae_features).
    """
    baseline_results = []
    baseline_features_map = {}
    baseline_headlines = []
    baseline_sae_max_rows: List[np.ndarray] = []
    latent_dim = int(getattr(sae, "latent_dim", 0))
    if latent_dim <= 0:
        enc_w = getattr(getattr(sae, "encoder", None), "weight", None)
        if enc_w is not None:
            latent_dim = int(enc_w.shape[0])
    target_layer = model.bert.encoder.layer[layer_to_extract]
    iter_indices = resolve_inference_sample_indices(len(test_ds), max_samples, sample_indices)
    n_iter = len(iter_indices)

    with torch.no_grad():
        for pos, sample_idx in enumerate(iter_indices):
            sample = test_ds[sample_idx]

            text = sample["text"]
            true_label = sample["label"]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
            token_ids = inputs["input_ids"][0].tolist()
            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt_tokens = [tok[2:] if tok.startswith("##") else tok for tok in raw_tokens]
            inputs = inputs.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            pred_id = logits.argmax(dim=-1).item()
            pred_label = model.config.id2label[pred_id]
            confidence = probs[0, pred_id].item()

            baseline_results.append(
                {
                    "sample_idx": sample_idx,
                    "text": text,
                    "true_label": model.config.id2label[true_label],
                    "predicted_label": pred_label,
                    "predicted_id": pred_id,
                    "confidence": confidence,
                    "logits": logits.cpu().numpy(),
                    "probs": probs.cpu().numpy(),
                }
            )

            captured_acts = []

            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    captured_acts.append(output[0].detach())
                else:
                    captured_acts.append(output.detach())

            temp_hook = target_layer.register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(**inputs)
            temp_hook.remove()

            if captured_acts:
                bert_activation = captured_acts[0].squeeze(0)

                attention_mask = inputs["attention_mask"].squeeze(0).bool()
                token_ids_tensor = inputs["input_ids"].squeeze(0)
                special_ids = set(tokenizer.all_special_ids)
                not_special = torch.tensor(
                    [tid.item() not in special_ids for tid in token_ids_tensor],
                    dtype=torch.bool,
                    device=device,
                )
                valid_mask = attention_mask & not_special
                bert_activation = bert_activation[valid_mask]
                filtered_token_ids = [tid for tid, valid in zip(token_ids, valid_mask.tolist()) if valid]
                filtered_prompt_tokens = [tok for tok, valid in zip(prompt_tokens, valid_mask.tolist()) if valid]

                if bert_activation.shape[0] > 0:
                    sae_features = sae.encode(bert_activation)
                    sae_features_cpu = sae_features.detach().cpu().numpy()
                    if feature_stats_baseline is not None:
                        feature_stats_baseline.update(sae_features_cpu)
                    if top_token_tracker_baseline is not None:
                        top_token_tracker_baseline.update(
                            sae_features_cpu,
                            filtered_token_ids,
                            prompt_idx=pos,
                            prompt_text=text,
                            prompt_tokens=filtered_prompt_tokens,
                            predicted_label=pred_label,
                            true_label=model.config.id2label[true_label],
                        )
                    max_activations_per_feature = sae_features_cpu.max(axis=0)
                    top_10_indices = np.argsort(max_activations_per_feature)[-10:][::-1]
                    top_features = [
                        {
                            "feature_id": int(fid),
                            "activation": float(max_activations_per_feature[fid]),
                        }
                        for fid in top_10_indices
                    ]
                    total_activation = sum(feat["activation"] for feat in top_features)
                    baseline_features_map[pos] = {
                        "top_features": top_features,
                        "total_activation": total_activation,
                    }
                    max_token_idx_per_feature = sae_features_cpu.argmax(axis=0)
                    baseline_headline_features = [
                        {
                            "feature_id": int(fid),
                            "max_activation": float(max_activations_per_feature[fid]),
                            "token_position": int(max_token_idx_per_feature[fid]),
                            "token_id": int(filtered_token_ids[max_token_idx_per_feature[fid]]),
                            "token_str": filtered_prompt_tokens[max_token_idx_per_feature[fid]],
                        }
                        for fid in top_10_indices
                        if max_activations_per_feature[fid] > 0
                    ]
                    baseline_headlines.append(
                        {
                            "row_id": int(pos),
                            "prompt": text,
                            "predicted_label": pred_label,
                            "confidence": float(confidence),
                            "true_label": model.config.id2label[true_label],
                            "correct": pred_label == model.config.id2label[true_label],
                            "num_tokens": int(len(filtered_token_ids)),
                            "features": baseline_headline_features,
                            "top_sae_features": top_sae_latents_by_headline(
                                max_activations_per_feature, HEADLINE_SAE_TOP_K
                            ),
                        }
                    )
                    baseline_sae_max_rows.append(max_activations_per_feature.astype(np.float32))
                    if interpretability_recorder is not None:
                        interpretability_recorder.update(sae_features_cpu, filtered_token_ids)
                else:
                    baseline_features_map[pos] = {"top_features": [], "total_activation": 0.0}
                    baseline_headlines.append(
                        {
                            "row_id": int(pos),
                            "prompt": text,
                            "predicted_label": pred_label,
                            "confidence": float(confidence),
                            "true_label": model.config.id2label[true_label],
                            "correct": pred_label == model.config.id2label[true_label],
                            "num_tokens": 0,
                            "features": [],
                            "top_sae_features": [],
                        }
                    )
                    baseline_sae_max_rows.append(
                        np.zeros(latent_dim, dtype=np.float32) if latent_dim > 0 else np.array([], dtype=np.float32)
                    )
            else:
                baseline_features_map[pos] = {"top_features": [], "total_activation": 0.0}
                baseline_headlines.append(
                    {
                        "row_id": int(pos),
                        "prompt": text,
                        "predicted_label": pred_label,
                        "confidence": float(confidence),
                        "true_label": model.config.id2label[true_label],
                        "correct": pred_label == model.config.id2label[true_label],
                        "num_tokens": 0,
                        "features": [],
                        "top_sae_features": [],
                    }
                )
                baseline_sae_max_rows.append(
                    np.zeros(latent_dim, dtype=np.float32) if latent_dim > 0 else np.array([], dtype=np.float32)
                )

            if (pos + 1) % 20 == 0:
                print(f"  Baseline: {pos + 1}/{n_iter} samples")

    baseline_accuracy = (
        sum(1 for r in baseline_results if r["predicted_id"] == test_ds[r["sample_idx"]]["label"])
        / len(baseline_results)
    )
    if baseline_sae_max_rows:
        baseline_sae_max_full = np.stack(baseline_sae_max_rows, axis=0)
    elif latent_dim > 0:
        baseline_sae_max_full = np.zeros((0, latent_dim), dtype=np.float32)
    else:
        baseline_sae_max_full = np.zeros((0, 0), dtype=np.float32)
    return baseline_results, baseline_features_map, baseline_accuracy, baseline_headlines, baseline_sae_max_full


def run_ablation_inference(
    model,
    tokenizer,
    test_ds,
    device: torch.device,
    sae,
    layer_to_extract: int,
    max_samples: int,
    max_seq_length: int,
    ablation_config: Dict,
    features_to_ablate: List[int],
    baseline_results: List[Dict],
    baseline_features_map: Dict,
    feature_stats_ablated,
    top_token_tracker_ablated,
    headline_aggregator_ablated,
    current_sample_data: Dict,
    similarity_enabled: bool,
    normalized_decoder: torch.Tensor,
    similarity_top_m: int,
    similarity_cache: Dict[int, List[int]],
    parent_features_to_ablate: Optional[List[int]] = None,
    sample_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, List[int]], set, float]:
    """Run ablation inference with feature intervention."""
    skip_hooks = ablation_config.get("skip_sae_reconstruction", False)
    ablated_results = []
    all_prompt_metadata_ablated = []
    baseline_lookup = {r["sample_idx"]: r for r in baseline_results}
    similarity_stats = {"original_counts": [], "expanded_counts": []}
    all_ablated_features_set = set()
    iter_indices = resolve_inference_sample_indices(len(test_ds), max_samples, sample_indices)
    n_iter = len(iter_indices)

    if not skip_hooks and ablation_config["mode"] != "per_sample_top_k":
        target_layer = model.bert.encoder.layer[layer_to_extract]
        intervention_hook = create_intervention_hook(sae, features_to_ablate, device, current_sample_data)
        hook_handle = target_layer.register_forward_hook(intervention_hook)

    with torch.no_grad():
        for pos, sample_idx in enumerate(iter_indices):
            sample = test_ds[sample_idx]

            text = sample["text"]
            true_label = sample["label"]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length)
            token_ids = inputs["input_ids"][0].tolist()

            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt_tokens = []
            for tok in raw_tokens:
                if tok.startswith("##"):
                    prompt_tokens.append(tok[2:])
                else:
                    prompt_tokens.append(tok)

            inputs = inputs.to(device)
            features_to_ablate_sample = None
            parent_features_to_ablate_sample = None

            if not skip_hooks and ablation_config["mode"] == "per_sample_top_k":
                parent_features_to_ablate_sample = [
                    f["feature_id"]
                    for f in baseline_features_map[pos]["top_features"][: ablation_config["k"]]
                ]
                features_to_ablate_sample = list(parent_features_to_ablate_sample)
                validate_feature_ids(features_to_ablate_sample, sae.latent_dim, "per_sample_top_k features")
                if similarity_enabled:
                    original_count = len(features_to_ablate_sample)
                    features_to_ablate_sample = expand_features_with_similarity(
                        features_to_ablate_sample, normalized_decoder, similarity_top_m, similarity_cache
                    )
                    similarity_stats["original_counts"].append(original_count)
                    similarity_stats["expanded_counts"].append(len(features_to_ablate_sample))

                all_ablated_features_set.update(features_to_ablate_sample)

                target_layer = model.bert.encoder.layer[layer_to_extract]
                intervention_hook = create_intervention_hook(
                    sae, features_to_ablate_sample, device, current_sample_data
                )
                hook_handle = target_layer.register_forward_hook(intervention_hook)

            current_sample_data["sae_features"] = None
            current_sample_data["token_ids"] = token_ids
            current_sample_data["prompt_tokens"] = prompt_tokens
            current_sample_data["text"] = text
            current_sample_data["idx"] = pos

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            if not skip_hooks and ablation_config["mode"] == "per_sample_top_k":
                hook_handle.remove()

            if skip_hooks:
                with torch.no_grad():
                    bert_outputs = model.bert(**inputs, output_hidden_states=True)
                    hidden_states = bert_outputs.hidden_states[layer_to_extract + 1]
                    sae_features = sae.encode(hidden_states.squeeze(0))
                    current_sample_data["sae_features"] = sae_features

            pred_id = logits.argmax(dim=-1).item()
            pred_label = model.config.id2label[pred_id]
            confidence = probs[0, pred_id].item()

            if ablation_config["mode"] == "per_sample_top_k":
                if parent_features_to_ablate_sample is not None:
                    features_ablated_for_this_sample = parent_features_to_ablate_sample
                else:
                    features_ablated_for_this_sample = [
                        f["feature_id"]
                        for f in baseline_features_map[pos]["top_features"][: ablation_config["k"]]
                    ]
            else:
                features_ablated_for_this_sample = (
                    parent_features_to_ablate if parent_features_to_ablate is not None else features_to_ablate
                )

            ablated_results.append(
                {
                    "sample_idx": sample_idx,
                    "text": text,
                    "true_label": model.config.id2label[true_label],
                    "predicted_label": pred_label,
                    "predicted_id": pred_id,
                    "confidence": confidence,
                    "logits": logits.detach().cpu().numpy(),
                    "probs": probs.detach().cpu().numpy(),
                    "ablated_features": features_ablated_for_this_sample,
                }
            )

            if current_sample_data["sae_features"] is not None:
                sae_features_cpu = current_sample_data["sae_features"].cpu().numpy()

                attention_mask = inputs["attention_mask"].squeeze(0).bool().cpu().numpy()
                token_ids_tensor = inputs["input_ids"].squeeze(0).cpu().numpy()
                special_ids = set(tokenizer.all_special_ids)
                not_special = np.array([tid not in special_ids for tid in token_ids_tensor])
                valid_mask = attention_mask & not_special

                sae_features_filtered = sae_features_cpu[valid_mask]
                filtered_token_ids = [tid for tid, valid in zip(token_ids, valid_mask) if valid]
                filtered_prompt_tokens = [tok for tok, valid in zip(prompt_tokens, valid_mask) if valid]

                if sae_features_filtered.shape[0] > 0:
                    seq_len = sae_features_filtered.shape[0]

                    feature_stats_ablated.update(sae_features_filtered)

                    top_token_tracker_ablated.update(
                        sae_features_filtered,
                        filtered_token_ids,
                        prompt_idx=pos,
                        prompt_text=text,
                        prompt_tokens=filtered_prompt_tokens,
                        predicted_label=pred_label,
                        true_label=model.config.id2label[true_label],
                    )

                    baseline_data = baseline_lookup[sample_idx]
                    if ablation_config["mode"] == "per_sample_top_k":
                        features_for_tracking = features_ablated_for_this_sample
                    else:
                        features_for_tracking = (
                            parent_features_to_ablate
                            if parent_features_to_ablate is not None
                            else features_to_ablate
                        )

                    headline_aggregator_ablated.add_headline_with_ablation_metrics(
                        prompt_idx=pos,
                        prompt_text=text,
                        token_activations=sae_features_filtered,
                        token_ids=filtered_token_ids,
                        token_strings=filtered_prompt_tokens,
                        predicted_label=pred_label,
                        true_label=model.config.id2label[true_label],
                        confidence=confidence,
                        baseline_features=baseline_features_map[pos],
                        features_to_ablate=features_for_tracking,
                        baseline_prediction=baseline_data["predicted_label"],
                        baseline_confidence=baseline_data["confidence"],
                    )

                    all_prompt_metadata_ablated.append(
                        {
                            "row_id": pos,
                            "seq_len": seq_len,
                            "prompt": text,
                            "predicted_label": pred_label,
                            "true_label": model.config.id2label[true_label],
                            "correct": pred_id == true_label,
                        }
                    )

            if (pos + 1) % 20 == 0:
                print(f"  Ablated: {pos + 1}/{n_iter} samples")

    if not skip_hooks and ablation_config["mode"] != "per_sample_top_k":
        hook_handle.remove()

    ablated_accuracy = (
        sum(1 for r in ablated_results if r["predicted_id"] == test_ds[r["sample_idx"]]["label"])
        / len(ablated_results)
    )

    return (
        ablated_results,
        all_prompt_metadata_ablated,
        similarity_stats,
        all_ablated_features_set,
        ablated_accuracy,
    )


def find_flipped_predictions(
    model,
    tokenizer,
    device: torch.device,
    sae,
    layer_to_extract: int,
    max_seq_length: int,
    baseline_results: List[Dict],
    ablated_results: List[Dict],
) -> List[Dict]:
    """Compare baseline vs ablated and return flipped prediction details."""
    flipped_samples = []
    target_layer = model.bert.encoder.layer[layer_to_extract]

    for baseline, ablated in zip(baseline_results, ablated_results):
        if baseline["predicted_id"] != ablated["predicted_id"]:
            inputs = tokenizer(baseline["text"], return_tensors="pt", truncation=True, max_length=max_seq_length)
            inputs = inputs.to(device)

            captured_acts = []

            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    captured_acts.append(output[0].detach())
                else:
                    captured_acts.append(output.detach())

            temp_hook = target_layer.register_forward_hook(capture_hook)
            with torch.no_grad():
                _ = model(**inputs)
            temp_hook.remove()

            if captured_acts:
                bert_activation = captured_acts[0].squeeze(0)

                attention_mask = inputs["attention_mask"].squeeze(0).bool()
                token_ids_tensor = inputs["input_ids"].squeeze(0)
                special_ids = set(tokenizer.all_special_ids)
                not_special = torch.tensor(
                    [tid.item() not in special_ids for tid in token_ids_tensor],
                    dtype=torch.bool,
                    device=device,
                )
                valid_mask = attention_mask & not_special
                bert_activation = bert_activation[valid_mask]

                if bert_activation.shape[0] > 0:
                    sae_features = sae.encode(bert_activation)
                    sae_features_cpu = sae_features.detach().cpu().numpy()
                    max_activations_per_feature = sae_features_cpu.max(axis=0)
                    top_10_indices = np.argsort(max_activations_per_feature)[-10:][::-1]
                    ablated_features_for_sample = ablated.get("ablated_features") or []
                    top_features = [
                        {
                            "feature_id": int(fid),
                            "activation": float(max_activations_per_feature[fid]),
                            "ablated": fid in ablated_features_for_sample,
                        }
                        for fid in top_10_indices
                    ]
                else:
                    top_features = []
            else:
                top_features = []

            flipped_samples.append(
                {
                    "sample_idx": baseline["sample_idx"],
                    "text": baseline["text"],
                    "true_label": baseline["true_label"],
                    "baseline_pred": baseline["predicted_label"],
                    "baseline_conf": baseline["confidence"],
                    "ablated_pred": ablated["predicted_label"],
                    "ablated_conf": ablated["confidence"],
                    "top_features": top_features,
                }
            )

    return flipped_samples


__all__ = [
    "create_intervention_hook",
    "validate_feature_ids",
    "normalize_decoder_weights",
    "expand_features_with_similarity",
    "inference_indices_random",
    "resolve_inference_sample_indices",
    "stratified_k_fold_indices",
    "select_features_to_ablate",
    "run_baseline_inference",
    "run_ablation_inference",
    "find_flipped_predictions",
    "HEADLINE_SAE_TOP_K",
    "top_sae_latents_by_headline",
]
