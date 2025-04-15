# src/auto_pipeline.py

import os
import subprocess
import json
import pandas as pd
import yaml
from datetime import datetime
from joblib import dump
from sklearn.model_selection import train_test_split

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model import train_model
from src.predict import generate_test_features, align_categorical_features

MODEL_LIST = ["lightgbm", "xgboost", "catboost"]
CATEGORICAL_COLS = ["city", "gender", "registered_via", "reg_year", "reg_month"]
PARAM_DIR = "params"
MODEL_DIR = "outputs/models"
PRED_DIR = "outputs/preds"
VAL_PRED_DIR = "outputs/val_preds"
SUBMISSION_DIR = "outputs/submissions"
LABEL_PATH = "outputs/val_labels.csv"
SAMPLE_SUB_PATH = "data/raw/sample_submission_v2.csv"
ENSEMBLE_WEIGHT_PATH = "ensemble_weights.yaml"

def ensure_dirs():
    os.makedirs(PARAM_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(VAL_PRED_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

def run_tuning_if_needed(model_type):
    param_path = f"{PARAM_DIR}/{model_type}_best_params.json"
    if not os.path.exists(param_path):
        print(f"🔧 Tuning {model_type}...")
        subprocess.run([
            "python", "-m", "src.tune_model",
            "--model_type", model_type,
            "--n_trials", "30",
            "--sample", "False"
        ])
    else:
        print(f"✅ Found existing params for {model_type}, skipping tuning.")

def train_and_predict(model_type, train_df, transactions_df, logs_df, members_df):
    print(f"🧠 Training {model_type}...")
    features = merge_all_features(train_df, members_df, transactions_df, logs_df)
    msno = features["msno"]
    X = features.drop(columns=["msno", "is_churn"])
    y = features["is_churn"]

    X_train, X_val, y_train, y_val, msno_train, msno_val = train_test_split(
        X, y, msno, test_size=0.2, stratify=y, random_state=42
    )

    model, y_pred, y_val_out, _ = train_model(
        X, y,
        model_type=model_type,
        categorical_features=CATEGORICAL_COLS,
        val_ids=msno_val
    )

    model_path = f"{MODEL_DIR}/{model_type}_model.joblib"
    dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    print(f"📈 Predicting with {model_type}...")
    test_features = generate_test_features(SAMPLE_SUB_PATH, members_df, transactions_df, logs_df)
    X_test = test_features.drop(columns=["msno", "is_churn"])

    if model_type == "lightgbm":
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype("category")
        X_test = align_categorical_features(X, X_test, CATEGORICAL_COLS)
    elif model_type == "xgboost":
        for col in CATEGORICAL_COLS:
            X_test[col] = X_test[col].astype("category").cat.codes
    elif model_type == "catboost":
        for col in CATEGORICAL_COLS:
            X_test[col] = X_test[col].astype(str).fillna("missing")

    y_test_pred = model.predict_proba(X_test)[:, 1]
    pred_df = pd.DataFrame({"msno": test_features["msno"], "is_churn": y_test_pred})
    pred_path = f"{PRED_DIR}/{model_type}_pred.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"📄 Prediction saved to {pred_path}")
    return pred_df

def blend_predictions(model_preds, weights=None):
    print("🔗 Blending predictions...")
    if weights is None:
        weights = [1 / len(model_preds)] * len(model_preds)
    blended = model_preds[0].copy()
    blended["is_churn"] = 0
    for pred, w in zip(model_preds, weights):
        blended["is_churn"] += w * pred["is_churn"]
    return blended

def main():
    ensure_dirs()
    print("🚀 Starting full pipeline...")

    train_df, transactions_df, logs_df, members_df = load_all_data(sample=False)

    model_preds = []
    for model_type in MODEL_LIST:
        run_tuning_if_needed(model_type)
        pred_df = train_and_predict(model_type, train_df, transactions_df, logs_df, members_df)
        model_preds.append(pred_df)

    # Step 2: Load weights from YAML
    if os.path.exists(ENSEMBLE_WEIGHT_PATH):
        with open(ENSEMBLE_WEIGHT_PATH, "r") as f:
            weights_config = yaml.safe_load(f)
        weights = [weights_config[model] for model in MODEL_LIST]
    else:
        print("⚠ No ensemble_weights.yaml found, using equal weights.")
        weights = [1 / len(MODEL_LIST)] * len(MODEL_LIST)

    final_pred = blend_predictions(model_preds, weights=weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"{SUBMISSION_DIR}/submission_blended_{timestamp}.csv"
    final_pred.to_csv(submission_path, index=False)
    print(f"✅ Final submission saved to {submission_path}")

if __name__ == "__main__":
    main()
