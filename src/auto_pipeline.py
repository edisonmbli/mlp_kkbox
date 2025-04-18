# src/auto_pipeline.py

import os
import subprocess
import json
import pandas as pd
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model_router import load_model_class, load_model_config
from src.utils.evaluation import save_validation_outputs
from src.utils.preprocessing import generate_test_features

PARAM_DIR = "params"
MODEL_DIR = "outputs/models"
PRED_DIR = "outputs/preds"
SUBMISSION_DIR = "outputs/submissions"
ENSEMBLE_WEIGHT_PATH = "ensemble_weights.yaml"
SAMPLE_SUB_PATH = "data/raw/sample_submission_v2.csv"

def ensure_dirs():
    os.makedirs(PARAM_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

def load_model_list_from_weights():
    if not os.path.exists(ENSEMBLE_WEIGHT_PATH):
        raise FileNotFoundError("ensemble_weights.yaml not found.")
    with open(ENSEMBLE_WEIGHT_PATH, "r") as f:
        weights_config = yaml.safe_load(f)
    model_list = list(weights_config.keys())
    weights = list(weights_config.values())
    return model_list, weights

def run_tuning_if_needed(model_type):
    param_path = f"{PARAM_DIR}/{model_type}_best_params.json"
    if not os.path.exists(param_path):
        print(f"🔧 Tuning {model_type}...")
        subprocess.run([
            "python", "-m", "src.tune_model",
            "--model_type", model_type,
            "--n_trials", "30"
            # "--sample", "False"
        ])
    else:
        print(f"✅ Found existing params for {model_type}, skipping tuning.")

def train_and_predict(model_type, train_df, transactions_df, logs_df, members_df):
    print(f"🧠 Training {model_type}...")
    config = load_model_config(model_type)
    model_class = load_model_class(model_type)

    features = merge_all_features(train_df, members_df, transactions_df, logs_df)
    msno = features["msno"]
    X = features.drop(columns=["msno", "is_churn"])
    y = features["is_churn"]

    X_train, X_val, y_train, y_val, msno_train, msno_val = train_test_split(
        X, y, msno, test_size=0.2, stratify=y, random_state=42
    )

    with open(f"{PARAM_DIR}/{model_type}_best_params.json", "r") as f:
        best_params = json.load(f)

    model = model_class(params=best_params, categorical_features=config.get("categorical_features"))
    model.fit(X_train, y_train, X_val, y_val)

    y_val_pred = model.predict_proba(X_val)
    save_validation_outputs(model_type, y_val, y_val_pred, msno_val)

    model_path = f"{MODEL_DIR}/{model_type}_model.pkl"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    print(f"📈 Predicting with {model_type}...")
    test_features = generate_test_features(SAMPLE_SUB_PATH, members_df, transactions_df, logs_df)
    X_test = test_features.drop(columns=["msno", "is_churn"])
    y_test_pred = model.predict_proba(X_test, X_ref=X)

    pred_df = pd.DataFrame({"msno": test_features["msno"], "is_churn": y_test_pred})
    pred_path = f"{PRED_DIR}/{model_type}_pred.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"📄 Prediction saved to {pred_path}")
    return pred_df

def blend_predictions(model_preds, weights):
    print("🔗 Blending predictions...")
    blended = model_preds[0].copy()
    blended["is_churn"] = 0
    for pred, w in zip(model_preds, weights):
        blended["is_churn"] += w * pred["is_churn"]
    return blended

def main():
    ensure_dirs()
    print("🚀 Starting full pipeline...")

    try:
        model_list, weights = load_model_list_from_weights()
    except FileNotFoundError:
        print("⚠ No ensemble_weights.yaml found. Please run optimize_ensemble.py first.")
        return

    print(f"📦 Models to run: {model_list}")
    print(f"📊 Weights: {weights}")

    train_df, transactions_df, logs_df, members_df = load_all_data(sample=False)

    model_preds = []
    for model_type in model_list:
        run_tuning_if_needed(model_type)
        pred_df = train_and_predict(model_type, train_df, transactions_df, logs_df, members_df)
        model_preds.append(pred_df)

    final_pred = blend_predictions(model_preds, weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"{SUBMISSION_DIR}/submission_blended_{timestamp}.csv"
    final_pred.to_csv(submission_path, index=False)
    print(f"✅ Final submission saved to {submission_path}")

if __name__ == "__main__":
    main()
