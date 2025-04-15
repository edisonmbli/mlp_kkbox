# src/optimize_ensemble.py

import pandas as pd
import numpy as np
import optuna
import yaml
import os
from sklearn.metrics import log_loss

MODEL_LIST = ["lightgbm", "xgboost", "catboost"]
VAL_PRED_DIR = "outputs/val_preds"
LABEL_PATH = "outputs/val_labels.csv"
OUTPUT_YAML = "ensemble_weights_optimized.yaml"

def load_validation_data():
    label_df = pd.read_csv(LABEL_PATH)
    preds = []
    for model in MODEL_LIST:
        pred_df = pd.read_csv(f"{VAL_PRED_DIR}/{model}_val.csv")
        pred_df = pred_df.rename(columns={"is_churn_pred": model})
        preds.append(pred_df)
    merged = label_df
    for df in preds:
        merged = merged.merge(df, on="msno", how="left")
    return merged

def objective(trial, df):
    weights = [trial.suggest_float(f"w_{m}", 0, 1) for m in MODEL_LIST]
    weights = np.array(weights)
    weights /= weights.sum()  # normalize
    blended = sum(df[m] * w for m, w in zip(MODEL_LIST, weights))
    return log_loss(df["is_churn"], blended)

def main():
    print("🔍 Optimizing ensemble weights based on validation log loss...")
    df = load_validation_data()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df), n_trials=50)

    best_weights = study.best_params
    total = sum(best_weights.values())
    normalized_weights = {k.replace("w_", ""): v / total for k, v in best_weights.items()}

    print("✅ Best weights (normalized):")
    for model, weight in normalized_weights.items():
        print(f"  {model}: {weight:.4f}")

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(normalized_weights, f)
    print(f"📄 Saved to {OUTPUT_YAML}")

if __name__ == "__main__":
    main()
