import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import gc

# load data
print("Loading training data...")
df_train = pd.read_parquet("data/clean/orange_large_train_clean.parquet", engine='pyarrow')

# Separate y and remove it from df to save memory (df_train becomes X_train)
y_train = df_train["churn"]
df_train.drop(columns=["churn"], inplace=True)

# Force GC
gc.collect()

# feature selection
print("Selecting Top 500 Features (Fitting on Train only)...")

# Select top 500 features based on ANOVA F-value
selector = SelectKBest(f_classif, k=500)
selector.fit(df_train, y_train)

# Get the names of the selected columns
selected_mask = selector.get_support()
selected_columns = df_train.columns[selected_mask]

print(f"Reduced from {df_train.shape[1]} to {len(selected_columns)} features.")

# Filter Train Data to just the selected columns
# We overwrite df_train to save memory
df_train = df_train[selected_columns]
df_train["churn"] = y_train

print("Saving optimized training data...")
df_train.to_parquet("data/clean500/orange_large_train_500.parquet", index=False)

# CLEANUP TRAIN BEFORE LOADING TEST
del df_train, y_train, selector
gc.collect()

# Load Test Data
print("Loading test data...")
df_test = pd.read_parquet("data/clean/orange_large_test_clean.parquet", engine='pyarrow')

# Filter Test Data using the SAME columns
# (We don't need to separate X and y here, just filter columns)
print("Filtering test data to selected features...")
cols_to_keep = list(selected_columns) + ["churn"]
df_test = df_test[cols_to_keep]

print("Saving optimized test data...")
df_test.to_parquet("data/clean500/orange_large_test_500.parquet", index=False)

print("Saved data/clean500/orange_large_train_500.parquet")
print("Saved data/clean500/orange_large_test_500.parquet")

# Clean up
del df_test
gc.collect()