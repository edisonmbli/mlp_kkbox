# src/data_loader.py

import pandas as pd
import os
from src.config import DATA_DIR, SAMPLE, SAMPLE_SIZE, RANDOM_SEED, MAX_LOG_ROWS, FILES, PARQUET_DIR
from src.utils.logger import get_logger
from src.utils.validation import split_by_expire_month

logger = get_logger("data_loader")

def load_csv(filename, usecols=None):
    path = os.path.join(DATA_DIR, filename)
    logger.info(f"Loading {filename} ...")
    return pd.read_csv(path, usecols=usecols)

def load_full_transactions(force_reload=False):
    """
    合并 transactions.csv 和 transactions_v2.csv，保存为 Parquet 格式。
    """
    parquet_path = os.path.join(PARQUET_DIR, "transactions_full.parquet")
    if os.path.exists(parquet_path) and not force_reload:
        logger.info("📦 Loading cached full transactions from Parquet...")
        return pd.read_parquet(parquet_path)

    logger.info("🔄 Merging transactions.csv + transactions_v2.csv ...")
    tx1 = pd.read_csv(os.path.join(DATA_DIR, FILES["transactions"]))
    tx2 = pd.read_csv(os.path.join(DATA_DIR, FILES["transactions_v2"]))
    df = pd.concat([tx1, tx2], ignore_index=True)
    df = df.drop_duplicates(subset=["msno", "transaction_date", "membership_expire_date"])
    df.to_parquet(parquet_path, index=False)
    logger.info(f"✅ Saved full transactions to {parquet_path}")
    return df

def load_full_user_logs(users=None, force_reload=False):
    """
    合并 user_logs.csv 和 user_logs_v2.csv，支持按用户过滤，保存为 Parquet 格式。
    """
    parquet_path = os.path.join(PARQUET_DIR, "user_logs_full.parquet")
    if os.path.exists(parquet_path) and not force_reload:
        logger.info("📦 Loading cached full user logs from Parquet...")
        df = pd.read_parquet(parquet_path)
        if users is not None:
            df = df[df["msno"].isin(users)]
        return df

    logger.info("🔄 Merging user_logs.csv + user_logs_v2.csv in chunks...")
    chunks = []
    for fname in [FILES["user_logs"], FILES["user_logs_v2"]]:
        path = os.path.join(DATA_DIR, fname)
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            if users is not None:
                chunk = chunk[chunk["msno"].isin(users)]
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"✅ Saved full user logs to {parquet_path}")
    return df

def load_members(users=None):
    df = load_csv(FILES["members"])
    if users is not None:
        df = df[df["msno"].isin(users)]
        logger.info(f"Filtered members to {len(df)} rows for selected users.")
    return df

def load_all_data(sample=False, sample_size=10000):
    logger.info("sample: %s", sample)
    
    """
    加载所有数据，并按月划分训练集和验证集
    """
    train_path = os.path.join(DATA_DIR, FILES["train"])
    train_df = pd.read_csv(train_path)
    transactions = load_full_transactions()

    train_set, val_set = split_by_expire_month(
        train_df=train_df,
        transactions=transactions,
        train_month="2017-02",
        val_month="2017-03",
        sample=sample,
        sample_size=sample_size
    )

    all_users = pd.concat([train_set["msno"], val_set["msno"]]).unique()
    logs = load_full_user_logs(users=all_users)
    members = load_members(all_users)

    return train_set, val_set, transactions, logs, members


