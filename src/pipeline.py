# src/pipeline.py

import os
import json
import pandas as pd
import argparse
from src.data_loader import load_all_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.model_router import load_model_config
from src.config import SAMPLE, SAMPLE_SIZE
from src.model_registry import MODEL_REGISTRY
from src.utils.preprocessing import get_or_cache_features
from src.utils.evaluation import print_dataset_stats
from src.utils.logger import get_logger
from src.config import PARAM_DIR, MODEL_DIR

logger = get_logger("pipeline")

def run_pipeline(model_type="lightgbm", sample=SAMPLE, sample_size=SAMPLE_SIZE):
    print(f"🚀 Running pipeline with model: {model_type}, sample={sample}, sample_size={sample_size}")

    # Step 1: 加载数据（已按 expire_date 划分 train/val）
    train_set, val_set, transactions, logs, members = load_all_data(sample=sample, sample_size=sample_size)

    # Step 2: 特征工程（合并后按用户到期日过滤行为数据）
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

    # Step 3: 拆分训练/验证特征
    train_feat = features[features["msno"].isin(train_set["msno"])]
    val_feat = features[features["msno"].isin(val_set["msno"])]

    # 过滤掉标签缺失的样本
    logger.warning(f"训练集中缺失标签的样本数: {train_feat['is_churn'].isna().sum()}")
    logger.warning(f"验证集中缺失标签的样本数: {val_feat['is_churn'].isna().sum()}")
    train_feat = train_feat[train_feat["is_churn"].notna()]
    val_feat = val_feat[val_feat["is_churn"].notna()]

    X_train = train_feat.drop(columns=["msno", "is_churn"])
    X_val = val_feat.drop(columns=["msno", "is_churn"])
    y_train = train_feat["is_churn"].astype(int)
    y_val = val_feat["is_churn"].astype(int)
    val_ids = val_feat["msno"]

    print_dataset_stats("train", train_feat)
    print_dataset_stats("val", val_feat)

    # Step 4: 加载调参结果（如有）
    param_path = os.path.join(PARAM_DIR, f"{model_type}_best_params.json")
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)
        print(f"✅ Loaded tuned parameters from {param_path}")
    else:
        best_params = None
        print(f"⚠ No tuned parameters found for {model_type}, using default config.")

    # Step 5: 模型训练
    config = load_model_config(model_type)
    categorical_cols = config.get("categorical_features", [])
    model, y_pred, y_val, metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_type=model_type,
        val_ids=val_ids,
        params=best_params
    )
    model_path = f"{MODEL_DIR}/{model_type}_model.pkl"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Step 6: 模型评估
    evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        model=model,
        feature_names=X_train.columns.tolist(),
        model_type=model_type
    )

    return model, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="选择要训练的模型类型")
    parser.add_argument("--sample", action="store_true", default=False, help="是否使用采样数据")
    parser.add_argument("--sample_size", type=int, default=10000, help="采样数据大小")
    args = parser.parse_args()

    print(f"🚀 开始训练模型: {args.model_type}")
    
    model, metrics = run_pipeline(
        model_type=args.model_type,
        sample=args.sample,
        sample_size=args.sample_size
    )
    
    print(f"✅ 训练完成! 验证集指标: {metrics}")

if __name__ == "__main__":
    main()
