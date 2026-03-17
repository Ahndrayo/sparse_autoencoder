import ollama
import numpy as np
import re
import random
from scipy.stats import spearmanr

class SAEInterpretabilityEvaluator:
    def __init__(self, json_path, model="deepseek-r1:32b"):
        # Load your JSON data here
        self.data = self._load_data(json_path)
        self.model = model
    
    def get_sampling_indices(activations, tokens, top_token_id):
        """
        Implements the Anthropic sampling strategy:
        - Top 10
        - 12 Intervals (2 each)
        - 5 Random
        - 10 Out-of-Context
        """
        indices = {}
        
        # 1. Top Activations
        top_indices = np.argsort(activations)[-10:][::-1]
        indices['top'] = top_indices.tolist()

        # 2. 12 Intervals (excluding the Top 10)
        # Get all non-zero activations
        non_zero_idx = np.where(activations > 0)[0]
        non_zero_idx = [i for i in non_zero_idx if i not in top_indices]
        
        if len(non_zero_idx) > 24:
            # Divide non-zero activations into 12 equal-sized quantiles
            sorted_non_zero = sorted(non_zero_idx, key=lambda i: activations[i])
            chunks = np.array_split(sorted_non_zero, 12)
            interval_samples = []
            for chunk in chunks:
                interval_samples.extend(random.sample(list(chunk), min(2, len(chunk))))
            indices['intervals'] = interval_samples
        else:
            indices['intervals'] = non_zero_idx

        # 3. Random Examples, excluding those that had already been selected in the earlier stages
        all_indices = list(range(len(activations)))
        excluded = set(indices['top']) | set(indices['intervals'])
        remaining = [i for i in all_indices if i not in excluded]
        indices['random'] = random.sample(remaining, 5)
    

        # 4. Out-of-Context (The "Hard" Negatives)
        # Find snippets where the top_token exists but activation is 0
        zero_idx = np.where(activations == 0)[0]
        out_of_context = [i for i in zero_idx if top_token_id in tokens[i]]
        indices['out_of_context'] = random.sample(out_of_context, min(10, len(out_of_context)))

        return indices


    def quantize(activation, max_val):
        """Linearly scale activation to a 0-9 integer."""
        if activation == 0: return 0
        # Anthropic uses quantization to make it easier for the LLM to 'score'
        return int(np.clip((activation / max_val) * 9, 0, 9))

    def _call_deepseek(self, prompt):
        """Sends a prompt to Ollama and returns the raw text response."""
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']

    # --- THE FILTER: Extracting numbers from LLM chatter ---
    def _parse_scores(self, text):
        return [int(s) for s in re.findall(r'Score:\s*(\d)', text)]

    def evaluate_feature(self, feature_id):
        feat = self.data[feature_id]
        acts = np.array(feat['activations'])
        snippets = feat['snippets']
        max_act = np.max(acts)
        top_token = feat.get('top_token_str', "")

        # 1. PHASE 1: Generate Explanation (49 examples)
        idx_dict = self._get_sampling_indices(acts, snippets, top_token)
        all_expl_idx = idx_dict['top'] + idx_dict['intervals'] + idx_dict['random'] + idx_dict['out_of_context']
        
        expl_lines = [f"Score: {self._quantize(acts[i], max_act)} | Text: {snippets[i]}" for i in all_expl_idx]
        prompt_1 = f"Explain this feature based on these examples:\n" + "\n".join(expl_lines)
        
        explanation = self._call_deepseek(prompt_1)
        
        # 2. PHASE 2: Prediction (60 examples)
        # We sample again, ideally ensuring no overlap with Phase 1
        idx_dict_2 = self._get_sampling_indices(acts, snippets, top_token)
        all_pred_idx = idx_dict_2['top'][:6] + idx_dict_2['intervals'] + idx_dict_2['random'][:10] + idx_dict_2['out_of_context'][:20]
        random.shuffle(all_pred_idx) # Shuffle so DeepSeek can't see the 'buckets'
        
        pred_lines = [f"Snippet: {snippets[i]}" for i in all_pred_idx]
        true_scores = [self._quantize(acts[i], max_act) for i in all_pred_idx]
        
        prompt_2 = f"Concept: {explanation}\nPredict 0-9 scores for these snippets (Format: 'Score: X'):\n" + "\n".join(pred_lines)
        
        raw_predictions = self._call_deepseek(prompt_2)
        predicted_scores = self._parse_scores(raw_predictions)

        # 3. Calculate Spearman Correlation
        if len(predicted_scores) == len(true_scores):
            correlation, _ = spearmanr(predicted_scores, true_scores)
            return {"explanation": explanation, "correlation": correlation}
        else:
            return {"explanation": explanation, "correlation": None, "error": "Score length mismatch"}