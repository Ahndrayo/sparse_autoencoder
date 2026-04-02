"""Two-phase SAE feature interpretability via local LLM (Ollama) + Spearman validation."""

import hashlib
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ollama
from scipy.stats import spearmanr

try:
    from scipy.stats import ConstantInputWarning
except ImportError:  # older scipy
    ConstantInputWarning = type("ConstantInputWarning", (UserWarning,), {})

DEFAULT_EXPLAIN_PROMPT = """You are a financial interpretability researcher.
Below are snippets where a specific internal feature of a finance model activates.
Each snippet is preceded by an activation score from 0-9 (9 is strongest).

{examples}

TASK: Identify the single, concise conceptual theme that triggers this feature.
- Use the scores to understand the nuance (e.g., why some snippets are 9s and others are 2s).
- Be succinct (under 15 words).
- Do not list specific tokens.
- If the snippets appear random or inconsistent, reply [UNINTERPRETABLE].

EXPLANATION:
"""

DEFAULT_EVAL_PROMPT = """You are simulating a specific AI feature defined as: "{explanation}"

For each snippet below, predict the activation strength (0-9).
0 = The concept is completely absent.
9 = The concept is perfectly and strongly present.

SNIPPETS:
{snippets_block}

TASK: For each snippet in order, output exactly one line per snippet:
Score: [X]
where X is a single digit 0-9. Do not skip any snippet. No other scores on each line.
"""


def _spearman_or_none(
    predicted: Sequence[int], true: Sequence[int]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Spearman ρ is undefined if either series is constant (all tied ranks).
    Returns (rho, None) on success, or (None, reason) when ρ cannot be defined.
    """
    pred_arr = np.asarray(predicted, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    n = pred_arr.size
    if n < 2:
        return None, "need at least 2 evaluation snippets"
    if np.ptp(pred_arr) == 0 and np.ptp(true_arr) == 0:
        return None, "constant predictions and constant ground truth (no rank variation)"
    if np.ptp(pred_arr) == 0:
        return None, "constant model predictions (e.g. same score for every snippet)"
    if np.ptp(true_arr) == 0:
        return None, "constant ground-truth scores (often all 0 after quantization on weak features)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, _ = spearmanr(pred_arr, true_arr)
    if rho is None or (isinstance(rho, float) and rho != rho):
        return None, "Spearman undefined for this pair (degenerate ranks)"
    return float(rho), None


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek-R1 style reasoning blocks from model output."""
    if not text:
        return text
    out = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return out.strip()


def _per_feature_rng_seed(random_seed: int, feature_key: str) -> int:
    """Stable seed for np.random.Generator; avoids salted built-in hash()."""
    try:
        fid = int(feature_key)
    except ValueError:
        digest = hashlib.md5(feature_key.encode("utf-8")).digest()[:8]
        fid = int.from_bytes(digest, "little")
    return (random_seed + fid) % (2**32)


class SAEInterpretabilityEvaluator:
    """
    Phase 1: Explain feature from scored examples (Anthropic-style sampling).
    Phase 2: LLM predicts quantized scores on held-out snippets; Spearman vs truth.
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        model: str = "deepseek-r1:32b",
        explain_prompt_template: Optional[str] = None,
        eval_prompt_template: Optional[str] = None,
        random_seed: Optional[int] = None,
        data_fraction: float = 1.0,
    ):
        if not (0 < data_fraction <= 1.0):
            raise ValueError("data_fraction must be in (0, 1]")
        self.data = self._load_data(json_path)
        subset_seed = random_seed if random_seed is not None else 0
        if data_fraction < 1.0:
            self._subsample_loaded_data(self.data, data_fraction, subset_seed)
        self.model = model
        self.explain_prompt_template = explain_prompt_template or DEFAULT_EXPLAIN_PROMPT
        self.eval_prompt_template = eval_prompt_template or DEFAULT_EVAL_PROMPT
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    @staticmethod
    def _load_data(json_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Interpretability data not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _subsample_loaded_data(data: Dict[str, Any], data_fraction: float, random_seed: int) -> None:
        """In-place: each feature keeps max(1, round(n * data_fraction)) rows, original order preserved."""
        for key, feat in data.items():
            if not isinstance(feat, dict):
                continue
            acts = feat.get("activations")
            snippets = feat.get("snippets")
            if not isinstance(acts, list) or not isinstance(snippets, list):
                continue
            n = len(acts)
            if n != len(snippets) or n == 0:
                continue
            k = max(1, int(round(n * data_fraction)))
            k = min(k, n)
            rng = np.random.default_rng(_per_feature_rng_seed(random_seed, str(key)))
            idx = rng.choice(n, size=k, replace=False)
            idx = np.sort(idx)
            feat["activations"] = [acts[i] for i in idx]
            feat["snippets"] = [snippets[i] for i in idx]

    @staticmethod
    def quantize(activation: float, max_val: float) -> int:
        """Linearly scale activation to a 0-9 integer (0 stays 0)."""
        if max_val <= 0 or activation == 0:
            return 0
        return int(np.clip((activation / max_val) * 9, 0, 9))

    @staticmethod
    def get_sampling_indices(
        activations: np.ndarray,
        snippets: Sequence[str],
        top_token_str: str,
    ) -> Dict[str, List[int]]:
        """
        Anthropic-style buckets: top 10, 12 interval pairs, 5 random, 10 out-of-context.
        """
        n = len(activations)
        if n == 0:
            return {"top": [], "intervals": [], "random": [], "out_of_context": []}

        acts = np.asarray(activations, dtype=np.float64)
        indices: Dict[str, List[int]] = {}

        top_indices = np.argsort(acts)[-10:][::-1].tolist()
        indices["top"] = top_indices

        non_zero_idx = [i for i in np.where(acts > 0)[0].tolist() if i not in top_indices]

        if len(non_zero_idx) > 24:
            sorted_non_zero = sorted(non_zero_idx, key=lambda i: acts[i])
            chunks = np.array_split(sorted_non_zero, 12)
            interval_samples: List[int] = []
            for chunk in chunks:
                ch = list(chunk)
                k = min(2, len(ch))
                if k:
                    interval_samples.extend(random.sample(ch, k))
            indices["intervals"] = interval_samples
        else:
            indices["intervals"] = list(non_zero_idx)

        excluded = set(indices["top"]) | set(indices["intervals"])
        remaining = [i for i in range(n) if i not in excluded]
        k_rand = min(5, len(remaining))
        indices["random"] = random.sample(remaining, k_rand) if k_rand else []

        zero_idx = np.where(acts == 0)[0].tolist()
        tok = top_token_str or ""
        if tok:
            out_of_context = [i for i in zero_idx if tok in str(snippets[i])]
        else:
            out_of_context = []
        k_ooc = min(10, len(out_of_context))
        indices["out_of_context"] = random.sample(out_of_context, k_ooc) if k_ooc else []

        return indices

    def _call_llm(self, prompt: str) -> str:
        """Return cleaned response only (backwards compatible)."""
        _, cleaned = self._call_llm_raw_and_cleaned(prompt)
        return cleaned

    def _call_llm_raw_and_cleaned(self, prompt: str) -> Tuple[str, str]:
        """Return (raw_response, cleaned_response).

        raw_response is the full `ollama.generate(...).response` string.
        cleaned_response is the string with <think>...</think> removed.
        """
        response = ollama.generate(model=self.model, prompt=prompt)
        raw = response.get("response", "") or ""
        cleaned = _strip_thinking(raw)
        return raw, cleaned

    @staticmethod
    def _parse_scores(text: str) -> List[int]:
        """Extract Score: X patterns in order (first digit per line preferred)."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        scores: List[int] = []
        for ln in lines:
            m = re.search(r"Score:\s*(\d)", ln, re.IGNORECASE)
            if m:
                d = int(m.group(1))
                if 0 <= d <= 9:
                    scores.append(d)
        if scores:
            return scores
        return [int(s) for s in re.findall(r"\b([0-9])\b", text) if 0 <= int(s) <= 9]

    def _explain_indices(self, acts: np.ndarray, snippets: List[str], top_token: str) -> List[int]:
        d = self.get_sampling_indices(acts, snippets, top_token)
        return d["top"] + d["intervals"] + d["random"] + d["out_of_context"]

    def _prediction_indices(
        self,
        acts: np.ndarray,
        snippets: List[str],
        top_token: str,
        exclude: set,
        n_target: int = 20,
    ) -> List[int]:
        """Hold-out set disjoint from explain indices when possible."""
        n = len(acts)
        exclude = set(exclude)
        picked: List[int] = []
        for _ in range(30):
            d = self.get_sampling_indices(acts, snippets, top_token)
            parts = d["top"][:6] + d["intervals"] + d["random"][: min(10, len(d["random"]))] + d["out_of_context"][
                : min(20, len(d["out_of_context"]))
            ]
            for i in parts:
                if i not in exclude and i not in picked:
                    picked.append(i)
            if len(picked) >= n_target:
                break
        pool = [i for i in range(n) if i not in exclude and i not in picked]
        random.shuffle(pool)
        while len(picked) < n_target and pool:
            picked.append(pool.pop())
        random.shuffle(picked)
        return picked[:n_target] if len(picked) >= n_target else picked

    def evaluate_feature(self, feature_id: Union[str, int]) -> Dict[str, Any]:
        key = str(feature_id)
        if key not in self.data:
            return {"error": f"Feature {key} not in JSON", "correlation": None}
        feat = self.data[key]
        acts = np.array(feat["activations"], dtype=np.float64)
        snippets = feat["snippets"]
        if len(acts) != len(snippets):
            return {"error": "activations/snippets length mismatch", "correlation": None}

        max_act = float(np.max(acts)) if acts.size else 0.0
        top_token = feat.get("top_token_str") or ""

        expl_idx = self._explain_indices(acts, snippets, top_token)
        explanation_examples: List[Dict[str, Any]] = [
            {
                "example_index": int(i),
                "activation": float(acts[i]),
                "quantized_score": int(self.quantize(float(acts[i]), max_act)),
                "snippet": snippets[i],
            }
            for i in expl_idx
        ]
        expl_lines = [
            f"Score: {ex['quantized_score']} | Text: {ex['snippet']}" for ex in explanation_examples
        ]
        examples_block = "\n".join(expl_lines)
        prompt_1 = self.explain_prompt_template.format(examples=examples_block)
        raw_response_explanation, cleaned_response_explanation = self._call_llm_raw_and_cleaned(prompt_1)
        explanation = cleaned_response_explanation.strip()

        if "UNINTERPRETABLE" in explanation.upper():
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "skipped": True,
                "explanation_prompt_1": prompt_1,
                "raw_response_explanation": raw_response_explanation,
                "cleaned_response_explanation": cleaned_response_explanation,
                "expl_idx": [int(i) for i in expl_idx],
                "explanation_examples": explanation_examples,
            }

        pred_idx = self._prediction_indices(acts, snippets, top_token, set(expl_idx))
        if not pred_idx:
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "error": "No prediction indices available",
                "explanation_prompt_1": prompt_1,
                "raw_response_explanation": raw_response_explanation,
                "cleaned_response_explanation": cleaned_response_explanation,
                "expl_idx": [int(i) for i in expl_idx],
                "explanation_examples": explanation_examples,
                "pred_idx": [],
                "evaluation_prompt_2": None,
                "raw_response_eval": None,
                "evaluation_examples": [],
            }

        evaluation_examples: List[Dict[str, Any]] = [
            {
                "example_index": int(i),
                "activation": float(acts[i]),
                "quantized_true_score": int(self.quantize(float(acts[i]), max_act)),
                "snippet": snippets[i],
            }
            for i in pred_idx
        ]
        true_scores = [ex["quantized_true_score"] for ex in evaluation_examples]
        numbered = "\n".join(f"[{k + 1}] {snippets[i]}" for k, i in enumerate(pred_idx))
        prompt_2 = self.eval_prompt_template.format(explanation=explanation, snippets_block=numbered)
        raw_response_eval, cleaned_response_eval = self._call_llm_raw_and_cleaned(prompt_2)
        predicted_scores = self._parse_scores(cleaned_response_eval)

        if len(predicted_scores) >= len(true_scores):
            predicted_scores = predicted_scores[: len(true_scores)]
        else:
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "error": f"Too few scores: got {len(predicted_scores)}, need {len(true_scores)}",
                "explanation_prompt_1": prompt_1,
                "raw_response_explanation": raw_response_explanation,
                "cleaned_response_explanation": cleaned_response_explanation,
                "expl_idx": [int(i) for i in expl_idx],
                "explanation_examples": explanation_examples,
                "pred_idx": [int(i) for i in pred_idx],
                "evaluation_prompt_2": prompt_2,
                "raw_response_eval": raw_response_eval,
                "cleaned_response_eval": cleaned_response_eval,
                "evaluation_examples": [
                    {
                        **ex,
                        "predicted_score": int(predicted_scores[k])
                        if k < len(predicted_scores)
                        else None,
                    }
                    for k, ex in enumerate(evaluation_examples)
                ],
                "raw_eval_response": raw_response_eval[:2000],
                "true_scores": true_scores,
                "predicted_scores": predicted_scores,
            }

        correlation, corr_note = _spearman_or_none(predicted_scores, true_scores)
        evaluation_examples_with_pred: List[Dict[str, Any]] = []
        for k, ex in enumerate(evaluation_examples):
            evaluation_examples_with_pred.append(
                {
                    **ex,
                    "predicted_score": int(predicted_scores[k])
                    if k < len(predicted_scores)
                    else None,
                }
            )
        out: Dict[str, Any] = {
            "feature_id": key,
            "explanation": explanation,
            "explanation_prompt_1": prompt_1,
            "raw_response_explanation": raw_response_explanation,
            "cleaned_response_explanation": cleaned_response_explanation,
            "expl_idx": [int(i) for i in expl_idx],
            "explanation_examples": explanation_examples,
            "correlation": correlation,
            "n_eval": len(true_scores),
            "evaluation_prompt_2": prompt_2,
            "raw_response_eval": raw_response_eval,
            "cleaned_response_eval": cleaned_response_eval,
            "pred_idx": [int(i) for i in pred_idx],
            "evaluation_examples": evaluation_examples_with_pred,
            "predicted_scores": predicted_scores,
            "true_scores": true_scores,
        }
        if corr_note is not None:
            out["correlation_note"] = corr_note
        return out

    def run_features(
        self,
        feature_ids: Optional[Sequence[Union[str, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run evaluate_feature for all keys in JSON or a subset."""
        keys = [str(k) for k in feature_ids] if feature_ids is not None else list(self.data.keys())
        return [self.evaluate_feature(k) for k in keys]
