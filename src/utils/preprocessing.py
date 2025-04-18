# src/utils/preprocessing.py

import pandas as pd
from src.feature_engineering import merge_all_features

def align_categorical_features(X_train, X_test, categorical_cols):
    """
    将测试集中的分类特征与训练集对齐，确保类别一致。
    适用于 LightGBM 等对 category 类型敏感的模型。
    """
    for col in categorical_cols:
        if col in X_test.columns and col in X_train.columns:
            if pd.api.types.is_categorical_dtype(X_train[col]):
                X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)
    return X_test


def generate_test_features(sample_submission_path, members_df, transactions_df, logs_df):
    test_users = pd.read_csv(sample_submission_path)
    test_df = test_users.copy()
    test_df["is_churn"] = -1  # dummy label for兼容 merge_all_features
    features = merge_all_features(test_df, members_df, transactions_df, logs_df)
    return features