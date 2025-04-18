import pandas as pd
import numpy as np
import optuna
import yaml
import os
from sklearn.metrics import log_loss

VAL_PRED_DIR = "outputs/val_preds"
LABEL_PATH = "outputs/val_labels.csv"
OUTPUT_YAML = "ensemble_weights_optimized.yaml"

def get_available_models():
    files = os.listdir(VAL_PRED_DIR)
    models = [f.replace("_val.csv", "") for f in files if f.endswith("_val.csv")]
    return sorted(models)

def load_validation_data(model_list):
    label_df = pd.read_csv(LABEL_PATH, dtype={"msno": str})
    preds = []
    for model in model_list:
        pred_path = os.path.join(VAL_PRED_DIR, f"{model}_val.csv")
        if not os.path.exists(pred_path):
            print(f"⚠️ Warning: {pred_path} not found, skipping.")
            continue
        pred_df = pd.read_csv(pred_path, dtype={"msno": str})
        pred_df = pred_df.rename(columns={"is_churn_pred": model})
        preds.append(pred_df)
    merged = label_df
    for df in preds:
        merged = merged.merge(df, on="msno", how="left")
    merged = merged.dropna()  # 保守策略：只保留完整样本
    return merged

def objective(trial, df, model_list):
    weights = [trial.suggest_float(f"w_{m}", 0, 1) for m in model_list]
    weights = np.array(weights)
    weights /= weights.sum()
    blended = sum(df[m] * w for m, w in zip(model_list, weights))
    return log_loss(df["is_churn"], blended)

def main():
    print("🔍 Optimizing ensemble weights based on validation log loss...")

    model_list = get_available_models()
    if not model_list:
        print("❌ No model prediction files found in val_preds/.")
        return

    print(f"📦 Detected models: {model_list}")
    df = load_validation_data(model_list)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df, model_list), n_trials=50)

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
