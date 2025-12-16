import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import os
import gc
from ThresholdTuner import tune_threshold

# load data
print("Loading training data...")
df_train = pd.read_parquet("data/clean500/orange_large_train_500.parquet")
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

# Calculate class imbalance weight
num_neg = np.sum(y_train == 0)
num_pos = np.sum(y_train == 1)
scale_weight = num_neg / num_pos
print(f"Class Imbalance Weight: {scale_weight:.2f}")

# Training XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    scale_pos_weight=scale_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    device='gpu',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Train
print("Training XGBoost...")
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Tune Threshold on Validation Set
print("Tuning threshold on validation set...")
best_threshold = tune_threshold(model, X_val, y_val)

# Save model and threshold
os.makedirs("models", exist_ok=True)
print("Saving model to models/xgboost.joblib...")
model_data = {
    "model": model,
    "threshold": best_threshold
}
joblib.dump(model_data, "models/xgboost.joblib")

# Evaluate
print("\n--- Evaluation ---")
y_prob = model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC Score: {average_precision_score(y_val, y_prob):.4f}")