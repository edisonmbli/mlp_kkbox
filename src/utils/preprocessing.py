# src/utils/preprocessing.py

import os
import pandas as pd
from tqdm import tqdm
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


# def generate_test_features(sample_submission_path, members_df, transactions_df, logs_df):
#     """
#     为每个测试用户动态生成真实的 last_expire_date，用于行为数据过滤。
#     """
#     test_users = pd.read_csv(sample_submission_path)
#     test_df = test_users.copy()
#     test_df["is_churn"] = -1  # dummy label

#     # 提取每个测试用户的最新 membership_expire_date
#     transactions_df["membership_expire_date"] = pd.to_datetime(transactions_df["membership_expire_date"], format="%Y%m%d")
#     last_expire = transactions_df.groupby("msno")["membership_expire_date"].max().reset_index()
#     last_expire.columns = ["msno", "last_expire_date"]

#     # 合并到测试集
#     test_df = test_df.merge(last_expire, on="msno", how="left")

#     # 如果某些用户没有交易记录，默认设为 2017-03-31（保守处理）
#     test_df["last_expire_date"] = test_df["last_expire_date"].fillna(pd.to_datetime("2017-03-31"))

#     expire_df = test_df[["msno", "last_expire_date"]].copy()
#     features = merge_all_features(test_df, members_df, transactions_df, logs_df, expire_df)

#     return features


def generate_test_features_cached(members_df, transactions_df, logs_df, sample_submission_path, cache_path, force_reload=False):
    test_users = pd.read_csv(sample_submission_path)
    test_df = test_users.copy()
    test_df["is_churn"] = -1  # dummy label

    transactions_df["membership_expire_date"] = pd.to_datetime(transactions_df["membership_expire_date"], format="%Y%m%d", errors="coerce")
    last_expire = transactions_df.groupby("msno")["membership_expire_date"].max().reset_index()
    last_expire.columns = ["msno", "last_expire_date"]

    test_df = test_df.merge(last_expire, on="msno", how="left")
    test_df["last_expire_date"] = test_df["last_expire_date"].fillna(pd.to_datetime("2017-03-31"))
    expire_df = test_df[["msno", "last_expire_date"]].copy()

    return get_or_cache_features(
        cache_path=cache_path,
        batch_mode=True,
        batch_args={
            "full_df": test_df,
            "members_df": members_df,
            "transactions_df": transactions_df,
            "logs_df": logs_df,
            "expire_df": expire_df,
            "batch_size": 50000
        },
        force_reload=force_reload
    )


def merge_all_features_in_batches(
    full_df,
    members_df,
    transactions_df,
    logs_df,
    expire_df,
    batch_size=50000
):
    """
    分批运行 merge_all_features，降低内存占用。
    参数：
        - full_df: 包含 msno、is_churn 的 DataFrame（train + val）
        - members_df, transactions_df, logs_df: 全量数据
        - expire_df: 包含 msno 和 last_expire_date 的 DataFrame
        - batch_size: 每批处理的用户数
    返回：
        - 合并后的特征 DataFrame
    """
    all_msno = full_df["msno"].unique()
    batches = [all_msno[i:i+batch_size] for i in range(0, len(all_msno), batch_size)]

    features_list = []
    print(f"🔄 Generating features in {len(batches)} batches (batch size = {batch_size})")

    for i, batch_msno in enumerate(tqdm(batches, desc="🧩 Feature batches")):
        print(f"📦 Processing batch {i+1}/{len(batches)}: {len(batch_msno)} users")

        batch_df = full_df[full_df["msno"].isin(batch_msno)].copy()
        batch_expire = expire_df[expire_df["msno"].isin(batch_msno)].copy()

        # 防止 SettingWithCopyWarning
        if "last_expire_date" in batch_expire.columns:
            batch_expire["last_expire_date"] = pd.to_datetime(batch_expire["last_expire_date"], errors="coerce")

        batch_features = merge_all_features(batch_df, members_df, transactions_df, logs_df, batch_expire)
        features_list.append(batch_features)

    features = pd.concat(features_list, ignore_index=True)
    print(f"✅ Feature generation complete. Total rows: {len(features)}")
    return features


def get_or_cache_features(
    cache_path,
    compute_fn=None,
    batch_mode=False,
    batch_args=None,
    force_reload=False
):
    """
    通用特征缓存函数：支持标准模式和分批模式。
    参数：
        - cache_path: Parquet 缓存路径
        - compute_fn: 标准特征生成函数（如 merge_all_features）
        - batch_mode: 是否启用分批处理
        - batch_args: 分批处理所需参数字典（传给 merge_all_features_in_batches）
        - force_reload: 是否强制重新生成
    返回：
        - pd.DataFrame
    """
    if os.path.exists(cache_path) and not force_reload:
        print(f"📦 Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    print("🔄 Computing features...")
    if batch_mode:
        assert batch_args is not None, "batch_args must be provided when batch_mode=True"
        df = merge_all_features_in_batches(**batch_args)
    else:
        assert compute_fn is not None, "compute_fn must be provided when batch_mode=False"
        df = compute_fn()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"✅ Saved features to {cache_path}")
    return df
