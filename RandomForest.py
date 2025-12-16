import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Train RandomForest with class_weight='balanced' to handle imbalance
rf = RandomForestClassifier(
    n_estimators=1000,          
    max_depth=10,              
    min_samples_leaf=8,        
    max_features='sqrt',
    class_weight='balanced',   # <- handle class imbalance
    random_state=0,
    n_jobs=-1,
    verbose=0
)

rf.fit(X_train, y_train)

# Tune Threshold on Validation Set
print("Tuning threshold on validation set...")
best_threshold = tune_threshold(rf, X_val, y_val)

# Save model
os.makedirs("models", exist_ok=True)
print("Saving model to models/random_forest.joblib...")
model_data = {
    "model": rf,
    "threshold": best_threshold
}
joblib.dump(model_data, "models/random_forest.joblib")

# Evaluate
print("\n--- Evaluation ---")
y_prob = rf.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC Score: {average_precision_score(y_val, y_prob):.4f}")


