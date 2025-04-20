# src/tune_model.py

import os
import json
import argparse
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model_router import load_model_class, load_model_config
from src.model_registry import MODEL_REGISTRY
from src.utils.evaluation import print_dataset_stats
from src.utils.preprocessing import get_or_cache_features
from src.utils.logger import get_logger

logger = get_logger("tune_model")

def objective(trial, model_type, X_train, X_val, y_train, y_val):
    config = load_model_config(model_type)
    model_class = load_model_class(model_type)

    # 构建调参参数
    params = config.get("default_params", {}).copy()
    tune_space = config.get("tune_space", {})

    for param_name, spec_str in tune_space.items():
        param_type, low, high = [s.strip() for s in spec_str.split(",")]
        if param_type == "float":
            params[param_name] = trial.suggest_float(param_name, float(low), float(high))
        elif param_type == "int":
            params[param_name] = trial.suggest_int(param_name, int(low), int(high))
        else:
            raise ValueError(f"Unsupported param type: {param_type}")

    model = model_class(params=params, categorical_features=config.get("categorical_features"))
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict_proba(X_val)

    return log_loss(y_val, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Which model to tune")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--sample", action="store_true", help="Use sampled data")
    parser.add_argument("--sample_size", type=int, default=10000)
    args = parser.parse_args()

    print(f"🔍 Tuning {args.model_type} with {args.n_trials} trials...")

    # 加载数据（已按 expire_date 划分 train/val）
    train_set, val_set, transactions, logs, members = load_all_data(sample=args.sample, sample_size=args.sample_size)

    # 合并训练集和验证集用于特征构造
    full_df = pd.concat([train_set, val_set], axis=0)
    expire_df = full_df[["msno", "last_expire_date"]].copy()

    # 分批构造&缓存
    features = get_or_cache_features(
        cache_path="data/processed/train_features.parquet",
        batch_mode=True,  # 分批构造
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
    print_dataset_stats("train", train_feat)
    print_dataset_stats("val", val_feat)

    # 过滤掉标签缺失的样本
    logger.warning(f"训练集中缺失标签的样本数: {train_feat['is_churn'].isna().sum()}")
    logger.warning(f"验证集中缺失标签的样本数: {val_feat['is_churn'].isna().sum()}")
    train_feat = train_feat[train_feat["is_churn"].notna()]
    val_feat = val_feat[val_feat["is_churn"].notna()]

    X_train = train_feat.drop(columns=["msno", "is_churn"])
    X_val = val_feat.drop(columns=["msno", "is_churn"])
    y_train = train_feat["is_churn"].astype(int)
    y_val = val_feat["is_churn"].astype(int)

    # 调参
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.model_type, X_train.copy(), X_val.copy(), y_train, y_val), n_trials=args.n_trials)

    print(f"✅ Best log loss: {study.best_value:.5f}")
    print("📌 Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存最佳参数
    os.makedirs("params", exist_ok=True)
    param_path = f"params/{args.model_type}_best_params.json"
    with open(param_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"📝 Best parameters saved to {param_path}")

if __name__ == "__main__":
    main()
