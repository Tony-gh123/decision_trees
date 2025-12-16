import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
import joblib
import os
import gc

# Load test data
def load_test_data():
    print("Loading test data...")
    df = pd.read_parquet("data/clean500/orange_large_test_500.parquet")

    X_test = df.drop(columns=["churn"]).to_numpy(dtype=np.float32)
    y_test = df["churn"].to_numpy(dtype=np.int32)

    del df
    gc.collect()
    return X_test, y_test

# Load models and thresholds (beta for Custom Tree)
def get_models():
    models = {}
    thresholds = {}
    betas = {}
    os.makedirs("models", exist_ok=True)

    def load_model_data(path, name):
        if not os.path.exists(path):
            print(f"{name} model not found at: {path}")
            return

        print(f"Loading {name} from file...")
        data = joblib.load(path)

        # New format
        if isinstance(data, dict) and "model" in data:
            models[name] = data["model"]
            thresholds[name] = data.get("threshold", 0.5)
            if "beta" in data:
                betas[name] = data["beta"]
        else:
            # Old format
            models[name] = data
            thresholds[name] = 0.5

    load_model_data("models/xgboost.joblib", "XGBoost")
    load_model_data("models/random_forest.joblib", "Random Forest")
    load_model_data("models/decision_tree_custom.joblib", "Custom Tree")

    return models, thresholds, betas

# Prior-correction for Custom Tree ONLY
def get_calibrated_proba(model, X, name, beta):
    """
    Returns probabilities.
    For Custom Tree only, apply prior-odds correction because it was trained on
    oversampled (balanced) data.

    P_real = (P_model * beta) / (P_model * beta + (1 - P_model))

    beta should be odds(pi) = pi/(1-pi) where pi is the true prevalence in the population.
    """
    y_prob = model.predict_proba(X)[:, 1].astype(np.float64)

    # clip to keep log_loss/calibration stable
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)

    if name == "Custom Tree":
        num = y_prob * beta
        den = num + (1.0 - y_prob)
        y_prob = num / den
        y_prob = np.clip(y_prob, eps, 1 - eps)

    return y_prob

# Figure 1: Model Comparison Metrics
def plot_results(models, thresholds, betas, X_test, y_test):
    os.makedirs("figures", exist_ok=True)

    default_beta = 1.0

    print("Generating Model Comparison Metrics plot...")
    plt.figure(figsize=(15, 8))

    # ROC Curve (uses adjusted probs for Custom Tree)
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        beta = betas.get(name, default_beta)
        y_prob_adj = get_calibrated_proba(model, X_test, name, beta)

        fpr, tpr, _ = roc_curve(y_test, y_prob_adj)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")

    # Precision-Recall Curve (uses adjusted probs for Custom Tree)
    plt.subplot(1, 2, 2)
    for name, model in models.items():
        beta = betas.get(name, default_beta)
        y_prob_adj = get_calibrated_proba(model, X_test, name, beta)

        precision, recall, _ = precision_recall_curve(y_test, y_prob_adj)
        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # Stats Table
    # AUC / PR-AUC use adjusted probs (ranking & score-quality)
    # LogLoss reported BOTH raw and adjusted (probability-quality)
    # Threshold metrics use RAW probs because thresholds were tuned on raw probs
    # (and for RF/XGB raw==adj anyway)
    stats = []
    print("\n Model Performance Statistics (Validation-Tuned Thresholds)")
    print(
        f"{'Model':<15} {'AUC':<10} {'PR-AUC':<10} {'Accuracy':<10} "
        f"{'Precision':<10} {'Recall':<10} {'F1':<10} "
        f"{'LogLoss_raw':<12} {'LogLoss_adj':<12} {'Threshold':<10}"
    )
    print("-" * 130)

    for name, model in models.items():
        best_thr = thresholds.get(name, 0.5)
        beta = betas.get(name, default_beta)

        # Raw and adjusted probabilities
        y_prob_raw = model.predict_proba(X_test)[:, 1].astype(np.float64)
        y_prob_raw = np.clip(y_prob_raw, 1e-12, 1 - 1e-12)

        y_prob_adj = get_calibrated_proba(model, X_test, name, beta)

        # Ranking/score metrics: use adjusted probs for Custom Tree
        roc_auc = roc_auc_score(y_test, y_prob_adj)
        pr_auc = average_precision_score(y_test, y_prob_adj)

        # Probability-quality metrics: report both
        ll_raw = log_loss(y_test, y_prob_raw)
        ll_adj = log_loss(y_test, y_prob_adj)

        # Threshold metrics: use raw probs (threshold was tuned on raw probs)
        y_pred = (y_prob_raw >= best_thr).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        stats.append([name, roc_auc, pr_auc, acc, prec, rec, f1, ll_raw, ll_adj, best_thr])

        print(
            f"{name:<15} {roc_auc:<10.3f} {pr_auc:<10.3f} {acc:<10.3f} "
            f"{prec:<10.3f} {rec:<10.3f} {f1:<10.3f} "
            f"{ll_raw:<12.3f} {ll_adj:<12.3f} {best_thr:<10.3f}"
        )

    # Add table to bottom
    plt.subplots_adjust(bottom=0.33)
    columns = [
        "Model", "AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1",
        "LogLoss_raw", "LogLoss_adj", "Best Thr"
    ]

    cell_text = []
    for row in stats:
        formatted = [row[0]]
        for val in row[1:]:
            formatted.append(f"{val:.3f}" if isinstance(val, (int, float, np.floating)) else str(val))
        cell_text.append(formatted)

    ax_table = plt.axes([0.03, 0.02, 0.94, 0.25])
    ax_table.axis("off")
    table = ax_table.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.4)

    print("Saving plot to 'figures/model_comparison_metrics.png'...")
    plt.savefig("figures/model_comparison_metrics.png", bbox_inches="tight")
    plt.close()

    # Figure 2: XGBoost Deep Dive
    if "XGBoost" in models:
        print("Generating XGBoost Deep Dive plot...")
        plt.figure(figsize=(15, 6))

        # Confusion matrix (thresholded on RAW probs)
        plt.subplot(1, 2, 1)
        xgb_model = models["XGBoost"]
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        best_thr = thresholds.get("XGBoost", 0.5)
        y_pred_best = (y_prob >= best_thr).astype(int)

        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix (XGBoost @ {best_thr:.2f} thr)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Feature importance (top 10)
        plt.subplot(1, 2, 2)
        if hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            plt.title("Top 10 Features (XGBoost)")
            plt.bar(range(10), importances[indices], align="center")
            plt.xticks(range(10), [f"Feature {i}" for i in indices], rotation=45)
            plt.ylabel("Relative Importance")
        else:
            plt.title("Top 10 Features (XGBoost)")
            plt.text(0.5, 0.5, "No feature_importances_ found on this model.", ha="center", va="center")
            plt.axis("off")

        print("Saving plot to 'figures/xgboost_deep_dive.png'...")
        plt.savefig("figures/xgboost_deep_dive.png", bbox_inches="tight")
        plt.close()


# Figure 3: Calibration Curve + Log Loss Bar Chart
def plot_calibration_and_loss(models, betas, X_test, y_test):
    print("Generating Calibration and Log Loss plot...")
    plt.figure(figsize=(15, 6))

    default_beta = 1.0

    # Calibration curves (use adjusted probs)
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        beta = betas.get(name, default_beta)
        prob_adj = get_calibrated_proba(model, X_test, name, beta)

        frac_pos, mean_pred = calibration_curve(y_test, prob_adj, n_bins=10)
        plt.plot(mean_pred, frac_pos, "s-", label=name)

    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Calibration Curve")
    plt.legend()

    # Log loss bar chart (use the SAME adjusted probs)
    plt.subplot(1, 2, 2)
    names = []
    values = []
    for name, model in models.items():
        beta = betas.get(name, default_beta)
        prob_adj = get_calibrated_proba(model, X_test, name, beta)
        loss = log_loss(y_test, prob_adj)
        names.append(name)
        values.append(loss)

    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=15, ha="right")
    plt.ylabel("Log Loss (same probs as calibration curve)")
    plt.title("Log Loss Bar Chart (Adjusted/Calibrated Probabilities)")

    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    print("Saving plot to 'figures/model_calibration.png'...")
    plt.savefig("figures/model_calibration.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    X_test, y_test = load_test_data()
    models, thresholds, betas = get_models()
    plot_results(models, thresholds, betas, X_test, y_test)
    plot_calibration_and_loss(models, betas, X_test, y_test)

