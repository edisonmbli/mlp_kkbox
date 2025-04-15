# src/tune_model.py

import json
import os
import argparse
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from lightgbm.callback import early_stopping

def objective(trial, model_type, X_train, X_val, y_train, y_val, categorical_cols):
    if model_type == "lightgbm":
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
            "n_estimators": 1000
        }
        model = LGBMClassifier(**params)
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50)])
        y_pred = model.predict_proba(X_val)[:, 1]

    elif model_type == "xgboost":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "n_estimators": 1000,
            "early_stopping_rounds": 50
        }
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_val[col] = X_val[col].astype("category").cat.codes
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_pred = model.predict_proba(X_val)[:, 1]

    elif model_type == "catboost":
        params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "iterations": 1000,
            "verbose": False
        }
        for col in categorical_cols:
            X_train[col] = X_train[col].astype(str).fillna("missing")
            X_val[col] = X_val[col].astype(str).fillna("missing")
        model = CatBoostClassifier(cat_features=categorical_cols, **params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        y_pred = model.predict_proba(X_val)[:, 1]

    else:
        raise ValueError("Unsupported model type")

    return log_loss(y_val, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--sample", type=bool, default=True)
    parser.add_argument("--sample_size", type=int, default=10000)
    args = parser.parse_args()

    print(f"🔍 Tuning {args.model_type} with {args.n_trials} trials...")

    train_df, transactions_df, logs_df, members_df = load_all_data(sample=args.sample, sample_size=args.sample_size)
    features = merge_all_features(train_df, members_df, transactions_df, logs_df)
    X = features.drop(columns=["msno", "is_churn"])
    y = features["is_churn"]
    categorical_cols = ["city", "gender", "registered_via", "reg_year", "reg_month"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.model_type, X_train.copy(), X_val.copy(), y_train, y_val, categorical_cols), n_trials=args.n_trials)

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
