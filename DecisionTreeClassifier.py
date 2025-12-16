import numpy as np
import pandas as pd
from DecisionTree import DecisionTree, summarize_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os
import gc
from ThresholdTuner import tune_threshold

# load data
print("Loading training data...")
df_train = pd.read_parquet("data\clean500\orange_large_train_500.parquet")
X_train_full = df_train.drop(columns=["churn"]).to_numpy(dtype=np.float32)
y_train_full = df_train["churn"].to_numpy(dtype=np.float32)

del df_train
gc.collect()

# train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_full
)

del X_train_full, y_train_full
gc.collect()

print(f"Training on: {len(X_train)} samples")

# Oversampling (random oversample minority class on TRAIN only)
# Identify indices within the training split
churn_indices = np.where(y_train == 1)[0]
non_churn_indices = np.where(y_train == 0)[0]

print(f"Original Train Class Distribution: Churn={len(churn_indices)}, Non-Churn={len(non_churn_indices)}")

if len(churn_indices) == 0 or len(non_churn_indices) == 0:
    print("Skipping oversampling: only one class present in training split.")
else:
    rng = np.random.default_rng(0)
    churn_indices_oversampled = rng.choice(
        churn_indices, size=len(non_churn_indices), replace=True
    )
    balanced_indices = np.concatenate([non_churn_indices, churn_indices_oversampled])
    rng.shuffle(balanced_indices)

    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]
    print(f"Oversampled Train Size: {len(X_train)} (Balanced)")

# Train tree on BALANCED train, test on REAL imbalanced test
tree = DecisionTree(
    max_depth=10, 
    min_samples_split=8, 
    verbose=True
)

# Fit the tree and summarize
tree.fit(X_train, y_train)
summarize_tree(tree)

# Tune Threshold on Validation Set
print("Tuning threshold on validation set...")
best_threshold = tune_threshold(tree, X_val, y_val)

# Calculate beta for calibration (based on original training distribution)
# We need to adjust the probabilities because the model was trained on balanced data (50/50)
# but the real data has a different prior.
prior_pos = len(churn_indices) / (len(churn_indices) + len(non_churn_indices))
beta = prior_pos / (1 - prior_pos)
print(f"Calculated calibration beta: {beta:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
print("Saving model to models/decision_tree_custom.joblib...")
model_data = {
    "model": tree,
    "threshold": best_threshold,
    "beta": beta
}
joblib.dump(model_data, "models/decision_tree_custom.joblib")

# Evaluate on TEST and print results
y_prob_uncalibrated = tree.predict_proba(X_val)[:, 1]

# Calibration correction for oversampling
# Formula: P_real = (P_model * beta) / (P_model * beta + (1 - P_model))
y_prob_calibrated = (y_prob_uncalibrated * beta) / (y_prob_uncalibrated * beta + (1 - y_prob_uncalibrated))
print(f"Applied calibration correction with beta={beta:.4f}")

# Use uncalibrated for prediction with the tuned threshold (which was tuned on uncalibrated data)
y_pred = (y_prob_uncalibrated >= best_threshold).astype(int)

test_acc = np.mean(y_pred == y_val)
print("Test accuracy of custom tree:", test_acc)
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification report:")
print(classification_report(y_val, y_pred))
print("ROC-AUC Score (Calibrated):", roc_auc_score(y_val, y_prob_calibrated))
print("PR-AUC Score (Calibrated):", average_precision_score(y_val, y_prob_calibrated))