import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def tune_threshold(model, X_val, y_val):
    """
    Finds the best threshold for a given model using the validation set.
    Optimizes for F1 Score.
    """
    # Get probabilities for the positive class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        # Fallback 
        return 0.5

    # Calculate Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    
    # Calculate F1 Score for each threshold
    # Note: precision and recall have one extra element (0 and 1) compared to thresholds
    # We use the standard formula: F1 = 2 * (P * R) / (P + R)
    numerator = 2 * recall * precision
    denominator = recall + precision
    
    # Handle division by zero
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    # Find the index of the maximum F1 score
    # We slice f1_scores[:-1] because thresholds is shorter by 1
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Threshold Tuning - Best Threshold: {best_threshold:.4f}, Best F1: {best_f1:.4f}")
    
    return best_threshold
