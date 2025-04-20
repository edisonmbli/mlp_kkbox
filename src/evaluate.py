# src/evaluate.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import numpy as np

def evaluate_model(y_true, y_pred, model, feature_names, model_type="lightgbm", save_path=None):
    print("📊 Evaluation Results:")
    logloss = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Log Loss: {logloss:.5f}")
    print(f"AUC Score: {auc:.5f}")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_roc.png")
    plt.show()

    # 特征重要性
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            importances = model.model.feature_importances_
        elif model_type == "catboost":
            importances = model.get_feature_importance()
        else:
            print("⚠️ Feature importance not supported for this model.")
            return

        sorted_idx = np.argsort(importances)[::-1][:20]
        top_features = [feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        plt.figure(figsize=(8, 6))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.title("Top 20 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_importance.png")
        plt.show()

    except Exception as e:
        print(f"⚠️ Failed to plot feature importance: {e}")
