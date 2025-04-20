# src/auto_pipeline.py

import argparse
import os
import subprocess
import json
import pandas as pd
import yaml
from datetime import datetime

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model_router import load_model_class, load_model_config
from src.utils.evaluation import save_validation_outputs
from src.utils.evaluation import print_dataset_stats
from src.utils.preprocessing import get_or_cache_features
from src.utils.preprocessing import generate_test_features_cached
from src.utils.logger import get_logger

logger = get_logger("auto_pipeline")

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
        ])
    else:
        print(f"✅ Found existing params for {model_type}, skipping tuning.")

def train_and_predict(model_type, train_set, val_set, transactions, logs, members):
    logger.info(f"🧠 Training {model_type}...")
    config = load_model_config(model_type)
    model_class = load_model_class(model_type)

    # 合并训练集和验证集用于特征构造
    full_df = pd.concat([train_set, val_set], axis=0)
    expire_df = full_df[["msno", "last_expire_date"]].copy()

    features = get_or_cache_features(
        cache_path="data/processed/train_features.parquet",
        batch_mode=True,
        batch_args={
            "full_df": full_df,
            "members_df": members,
            "transactions_df": transactions,
            "logs_df": logs,
            "expire_df": expire_df,
            "batch_size": 50000
        },
        force_reload=False
    )

    # 拆分训练/验证特征
    train_feat = features[features["msno"].isin(train_set["msno"])]
    val_feat = features[features["msno"].isin(val_set["msno"])]

    # 过滤掉标签缺失的样本
    logger.warning(f"训练集中缺失标签的样本数: {train_feat['is_churn'].isna().sum()}")
    logger.warning(f"验证集中缺失标签的样本数: {val_feat['is_churn'].isna().sum()}")
    train_feat = train_feat[train_feat["is_churn"].notna()]
    val_feat = val_feat[val_feat["is_churn"].notna()]
    print_dataset_stats("train", train_feat)
    print_dataset_stats("val", val_feat)

    X_train = train_feat.drop(columns=["msno", "is_churn"])
    X_val = val_feat.drop(columns=["msno", "is_churn"])
    y_train = train_feat["is_churn"].astype(int)
    y_val = val_feat["is_churn"].astype(int)
    msno_val = val_feat["msno"]

    # 加载调参结果
    with open(f"{PARAM_DIR}/{model_type}_best_params.json", "r") as f:
        best_params = json.load(f)

    model = model_class(params=best_params, categorical_features=config.get("categorical_features"))
    model.fit(X_train, y_train, X_val, y_val)

    y_val_pred = model.predict_proba(X_val)
    save_validation_outputs(model_type, y_val, y_val_pred, msno_val)

    model_path = f"{MODEL_DIR}/{model_type}_model.pkl"
    model.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")

    logger.info(f"📈 Predicting with {model_type}...")
    test_features = generate_test_features_cached(
        members_df=members,
        transactions_df=transactions,
        logs_df=logs,
        sample_submission_path=SAMPLE_SUB_PATH,
        cache_path="data/processed/test_features.parquet",
        force_reload=False
    )

    X_test = test_features.drop(columns=["msno", "is_churn"])
    y_test_pred = model.predict_proba(X_test, X_ref=X_train)

    pred_df = pd.DataFrame({"msno": test_features["msno"], "is_churn": y_test_pred})
    pred_path = f"{PRED_DIR}/{model_type}_pred.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"📄 Prediction saved to {pred_path}")
    return pred_df


def blend_predictions(model_preds, weights):
    logger.info("🔗 Blending predictions...")
    blended = model_preds[0].copy()
    blended["is_churn"] = 0
    for pred, w in zip(model_preds, weights):
        blended["is_churn"] += w * pred["is_churn"]
    return blended

def main():
    ensure_dirs()
    logger.info("🚀 Starting full pipeline...")

    # 加载模型列表和权重
    model_list, weights = load_model_list_from_weights()
    logger.info(f"📦 Models to run: {model_list}")
    logger.info(f"📊 Weights: {weights}")

    # 加载数据 
    train_set, val_set, transactions, logs, members = load_all_data(sample=False)

    # 运行调参
    logger.info("🔧 Running tuning for each model...")
    model_preds = []
    for model_type in model_list:
        run_tuning_if_needed(model_type)
        pred_df = train_and_predict(model_type, train_set, val_set, transactions, logs, members)
        model_preds.append(pred_df)

    # 融合预测
    final_pred = blend_predictions(model_preds, weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"{SUBMISSION_DIR}/submission_blended_{timestamp}.csv"
    final_pred.to_csv(submission_path, index=False)
    logger.info(f"✅ Final submission saved to {submission_path}")

if __name__ == "__main__":
    main()
