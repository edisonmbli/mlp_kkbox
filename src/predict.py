# src/predict.py

import argparse
import pandas as pd
import os
from joblib import dump, load
from datetime import datetime

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model import train_model

SUBMISSION_PATH = "outputs/submissions"
MODEL_PATH = "outputs/models"

def generate_test_features(sample_submission_path, members_df, transactions_df, logs_df):
    test_users = pd.read_csv(sample_submission_path)
    test_df = test_users.copy()
    test_df["is_churn"] = -1  # dummy label for compatibility
    features = merge_all_features(test_df, members_df, transactions_df, logs_df)
    return features

def align_categorical_features(X_train, X_test, categorical_cols):
    for col in categorical_cols:
        if col in X_test.columns and col in X_train.columns:
            if pd.api.types.is_categorical_dtype(X_train[col]):
                X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)
    return X_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--submission_name", type=str, default=None)
    args = parser.parse_args()

    print(f"🚀 Running prediction with model: {args.model_type}")

    # Step 1: 加载训练数据
    train_df, transactions_df, logs_df, members_df = load_all_data(sample=args.sample, sample_size=args.sample_size)
    train_features = merge_all_features(train_df, members_df, transactions_df, logs_df)
    X_train = train_features.drop(columns=["msno", "is_churn"])
    y_train = train_features["is_churn"]
    categorical_cols = ["city", "gender", "registered_via", "reg_year", "reg_month"]

    # Step 2: 加载或训练模型
    model_file = os.path.join(MODEL_PATH, f"{args.model_type}_model.joblib")
    if args.load_model and os.path.exists(model_file):
        print(f"📦 Loading model from {model_file}")
        model = load(model_file)
    else:
        print("🧠 Training model...")
        model, _, _, _ = train_model(X_train, y_train, model_type=args.model_type, categorical_features=categorical_cols)
        if args.save_model:
            os.makedirs(MODEL_PATH, exist_ok=True)
            dump(model, model_file)
            print(f"✅ Model saved to {model_file}")

    # Step 3: 构建测试集特征
    test_features = generate_test_features(
        sample_submission_path="data/raw/sample_submission_v2.csv",
        members_df=members_df,
        transactions_df=transactions_df,
        logs_df=logs_df
    )
    X_test = test_features.drop(columns=["msno", "is_churn"])

    # Step 3.5: 对齐分类特征（仅对 LightGBM 有效）
    if args.model_type == "lightgbm":
        for col in categorical_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
        X_test = align_categorical_features(X_train, X_test, categorical_cols)

    # Step 4: 预测
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Step 5: 生成提交文件
    submission = pd.DataFrame({
        "msno": test_features["msno"],
        "is_churn": y_test_pred
    })

    os.makedirs(SUBMISSION_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args.submission_name or f"submission_{args.model_type}_{timestamp}.csv"
    submission_path = os.path.join(SUBMISSION_PATH, filename)
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")

if __name__ == "__main__":
    main()
