import ollama
import numpy as np
import re
from scipy.stats import spearmanr

class SAEInterpretabiltyPipeline:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_explanation(self, top_examples, range_examples):
        """Phase 1: Generate the hypothesis using 49 examples."""
        prompt = f"""
        You are a financial interpretability researcher. 
        Below are snippets where a specific internal feature of a finance model activates.
        
        TOP ACTIVATING SNIPPETS (Max Strength):
        {top_examples}
        
        LOWER ACTIVATING SNIPPETS (Medium/Low Strength):
        {range_examples}
        
        TASK: Identify the single, concise conceptual theme that triggers this feature.
        - Be succinct (under 15 words).
        - Do not list specific tokens.
        - If the snippets appear random or inconsistent, reply [UNINTERPRETABLE].
        
        EXPLANATION:
        """
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response'].strip()

    def predict_scores(self, explanation, test_snippets):
        """Phase 2: Use the hypothesis to predict activations for 60 unseen tokens."""
        prompt = f"""
        You are simulating a specific AI feature defined as: "{explanation}"
        
        For each snippet below, predict the activation strength on a scale of 0 to 9.
        0 = The concept is completely absent.
        9 = The concept is perfectly and strongly present.
        
        SNIPPETS:
        {test_snippets}
        
        Respond ONLY with the scores in order, one per line, following this format:
        Score: [X]
        """
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response']

    def parse_scores(self, response_text):
        """Extracts integers 0-9 from the LLM response text."""
        scores = re.findall(r'Score:\s*(\d)', response_text)
        return [int(s) for s in scores]

    def run_full_test(self, feature_id, training_data, test_data, true_activations):
        # 1. Get Explanation
        explanation = self.generate_explanation(training_data['top'], training_data['range'])
        print(f"Feature {feature_id} Explanation: {explanation}")
        
        if "[UNINTERPRETABLE]" in explanation:
            return 0.0, explanation

        # 2. Get Predictions
        raw_predictions = self.predict_scores(explanation, test_data)
        predicted_scores = self.parse_scores(raw_predictions)
        
        # 3. Calculate Spearman Correlation
        # We compare the 60 predicted integers vs the 60 real activation values
        if len(predicted_scores) == len(true_activations):
            correlation, _ = spearmanr(predicted_scores, true_activations)
            return correlation, explanation
        else:
            print("Error: Score mismatch. DeepSeek didn't provide exactly 60 scores.")
            return None, explanation