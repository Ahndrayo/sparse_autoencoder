"""FinBERT training utilities."""

import numpy as np
import evaluate

# Load metrics
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 metrics for model evaluation.
    
    Args:
        eval_pred: Tuple of (logits, labels)
        
    Returns:
        Dictionary with accuracy and macro F1 scores
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }


__all__ = ["compute_metrics"]
