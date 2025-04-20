# src/feature_engineering.py

import pandas as pd
import numpy as np

def extract_user_profile_features(members_df):
    df = members_df.copy()
    df["bd"] = df["bd"].apply(lambda x: x if 10 <= x <= 100 else np.nan)
    df["reg_date"] = pd.to_datetime(df["registration_init_time"], format="%Y%m%d")
    df["reg_year"] = df["reg_date"].dt.year
    df["reg_month"] = df["reg_date"].dt.month
    df["is_gender_missing"] = df["gender"].isna().astype(int)
    df["is_age_missing"] = df["bd"].isna().astype(int)
    return df[["msno", "city", "bd", "gender", "registered_via", "reg_year", "reg_month", "is_gender_missing", "is_age_missing"]]

def filter_behavior_by_expire(df, expire_df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d")
    expire_df["last_expire_date"] = pd.to_datetime(expire_df["last_expire_date"])
    df = df.merge(expire_df, on="msno", how="left")
    return df[df[date_col] < df["last_expire_date"]]

def extract_transaction_features(transactions_df):
    df = transactions_df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y%m%d")
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format="%Y%m%d")
    df["discount"] = df["plan_list_price"] - df["actual_amount_paid"]
    df = df[df["discount"] < 1000]

    last_tx = df.sort_values("transaction_date").groupby("msno").tail(1)
    last_tx = last_tx[["msno", "is_auto_renew", "is_cancel", "actual_amount_paid", "payment_plan_days", "discount"]]
    last_tx.columns = ["msno", "last_auto_renew", "last_cancel", "last_paid", "last_plan_days", "last_discount"]

    agg = df.groupby("msno").agg({
        "payment_plan_days": ["mean", "std", "nunique"],
        "actual_amount_paid": ["mean", "std"],
        "discount": ["mean", "std"],
        "payment_method_id": "nunique",
        "transaction_date": "count"
    })
    agg.columns = ["_".join(col) for col in agg.columns]
    agg.reset_index(inplace=True)

    return last_tx.merge(agg, on="msno", how="left")

def extract_time_gap_features(transactions_df, members_df):
    df = transactions_df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y%m%d")
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format="%Y%m%d")
    members_df["registration_init_time"] = pd.to_datetime(members_df["registration_init_time"], format="%Y%m%d")

    first_tx = df.sort_values("transaction_date").groupby("msno").head(1)
    reg_map = members_df.set_index("msno")["registration_init_time"]
    first_tx["reg_to_first_tx_days"] = (first_tx["transaction_date"] - first_tx["msno"].map(reg_map)).dt.days

    df = df.sort_values(["msno", "transaction_date"])
    df["prev_expire"] = df.groupby("msno")["membership_expire_date"].shift(1)
    df["gap_days"] = (df["transaction_date"] - df["prev_expire"]).dt.days
    gap_agg = df.groupby("msno")["gap_days"].agg(["mean", "min", "max"]).reset_index()

    return first_tx[["msno", "reg_to_first_tx_days"]].merge(gap_agg, on="msno", how="left")

def extract_user_log_features(logs_df):
    df = logs_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    agg = df.groupby("msno").agg({
        "num_25": "sum",
        "num_50": "sum",
        "num_75": "sum",
        "num_985": "sum",
        "num_100": "sum",
        "num_unq": "mean",
        "total_secs": "sum",
        "date": "nunique"
    }).reset_index()
    agg.rename(columns={"date": "active_days"}, inplace=True)
    agg["avg_secs_per_day"] = agg["total_secs"] / (agg["active_days"] + 1e-5)
    agg["full_play_ratio"] = agg["num_100"] / (
        agg["num_25"] + agg["num_50"] + agg["num_75"] + agg["num_985"] + agg["num_100"] + 1e-5)
    return agg[["msno", "active_days", "avg_secs_per_day", "full_play_ratio", "num_unq"]]

def extract_recent_user_log_features(logs_df, days=30):
    df = logs_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    max_date = df["date"].max()
    recent_df = df[df["date"] >= max_date - pd.Timedelta(days=days)]

    agg = recent_df.groupby("msno").agg({
        "total_secs": "sum",
        "date": "nunique",
        "num_100": "sum",
        "num_25": "sum",
        "num_50": "sum",
        "num_75": "sum",
        "num_985": "sum"
    }).reset_index()
    agg.rename(columns={"date": f"recent_{days}_active_days", "total_secs": f"recent_{days}_secs"}, inplace=True)
    agg[f"recent_{days}_full_play_ratio"] = agg["num_100"] / (
        agg["num_25"] + agg["num_50"] + agg["num_75"] + agg["num_985"] + agg["num_100"] + 1e-5)
    return agg[["msno", f"recent_{days}_active_days", f"recent_{days}_secs", f"recent_{days}_full_play_ratio"]]

def extract_behavior_trend_features(full_log_df, recent_log_df):
    full = extract_user_log_features(full_log_df)
    recent = extract_recent_user_log_features(recent_log_df, days=7)
    merged = full.merge(recent, on="msno", how="left")
    merged["secs_change_rate"] = merged["recent_7_secs"] / (merged["avg_secs_per_day"] * 7 + 1e-5)
    merged["activity_drop"] = (merged["recent_7_active_days"] < (merged["active_days"] / 4)).astype(int)
    return merged[["msno", "secs_change_rate", "activity_drop"]]

def merge_all_features(train_df, members_df, transactions_df, logs_df, expire_df):
    # 过滤行为数据
    tx_filtered = filter_behavior_by_expire(transactions_df, expire_df, "transaction_date")
    logs_filtered = filter_behavior_by_expire(logs_df, expire_df, "date")

    f1 = extract_user_profile_features(members_df)
    f2 = extract_transaction_features(tx_filtered)
    f3 = extract_time_gap_features(tx_filtered, members_df)
    f4 = extract_user_log_features(logs_filtered)
    f5 = extract_recent_user_log_features(logs_filtered, days=7)
    f6 = extract_recent_user_log_features(logs_filtered, days=30)
    f7 = extract_behavior_trend_features(logs_filtered, logs_filtered)

    features = train_df[["msno", "is_churn"]].merge(f1, on="msno", how="left")
    features = features.merge(f2, on="msno", how="left")
    features = features.merge(f3, on="msno", how="left")
    features = features.merge(f4, on="msno", how="left")
    features = features.merge(f5, on="msno", how="left")
    features = features.merge(f6, on="msno", how="left")
    features = features.merge(f7, on="msno", how="left")

    return features
