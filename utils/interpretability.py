"""Two-phase SAE feature interpretability via local LLM (Ollama) + Spearman validation."""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ollama
from scipy.stats import spearmanr

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


def _strip_thinking(text: str) -> str:
    """Remove DeepSeek-R1 style reasoning blocks from model output."""
    if not text:
        return text
    out = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return out.strip()


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
    ):
        self.data = self._load_data(json_path)
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
        response = ollama.generate(model=self.model, prompt=prompt)
        return _strip_thinking(response.get("response", "") or "")

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
        n_target: int = 60,
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
        expl_lines = [f"Score: {self.quantize(acts[i], max_act)} | Text: {snippets[i]}" for i in expl_idx]
        examples_block = "\n".join(expl_lines)
        prompt_1 = self.explain_prompt_template.format(examples=examples_block)
        explanation = self._call_llm(prompt_1).strip()

        if "UNINTERPRETABLE" in explanation.upper():
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "skipped": True,
            }

        pred_idx = self._prediction_indices(acts, snippets, top_token, set(expl_idx))
        if not pred_idx:
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "error": "No prediction indices available",
            }

        true_scores = [self.quantize(acts[i], max_act) for i in pred_idx]
        numbered = "\n".join(f"[{k + 1}] {snippets[i]}" for k, i in enumerate(pred_idx))
        prompt_2 = self.eval_prompt_template.format(explanation=explanation, snippets_block=numbered)
        raw = self._call_llm(prompt_2)
        predicted_scores = self._parse_scores(raw)

        if len(predicted_scores) >= len(true_scores):
            predicted_scores = predicted_scores[: len(true_scores)]
        else:
            return {
                "feature_id": key,
                "explanation": explanation,
                "correlation": None,
                "error": f"Too few scores: got {len(predicted_scores)}, need {len(true_scores)}",
                "raw_eval_response": raw[:2000],
            }

        correlation, _ = spearmanr(predicted_scores, true_scores)
        return {
            "feature_id": key,
            "explanation": explanation,
            "correlation": float(correlation) if correlation == correlation else None,
            "n_eval": len(true_scores),
            "predicted_scores": predicted_scores,
            "true_scores": true_scores,
        }

    def run_features(
        self,
        feature_ids: Optional[Sequence[Union[str, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run evaluate_feature for all keys in JSON or a subset."""
        keys = [str(k) for k in feature_ids] if feature_ids is not None else list(self.data.keys())
        return [self.evaluate_feature(k) for k in keys]
