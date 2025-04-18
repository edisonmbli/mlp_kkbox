# src/data_loader.py

import pandas as pd
import os
from src.config import DATA_DIR, SAMPLE, SAMPLE_SIZE, RANDOM_SEED, MAX_LOG_ROWS, FILES
from src.utils.logger import get_logger

logger = get_logger("data_loader")

def load_csv(filename, usecols=None):
    path = os.path.join(DATA_DIR, filename)
    logger.info(f"Loading {filename} ...")
    return pd.read_csv(path, usecols=usecols)

def load_train_data(sample=SAMPLE, sample_size=SAMPLE_SIZE):
    df = load_csv(FILES["train"])

    logger.info(f"sample = {sample}")

    if sample:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        logger.info(f"Sampled {sample_size} users from train data.")
    return df

def load_transactions(users=None):
    df = load_csv(FILES["transactions"])
    if users is not None:
        df = df[df["msno"].isin(users)]
        logger.info(f"Filtered transactions to {len(df)} rows for sampled users.")
    return df

def load_user_logs(users=None, max_rows=MAX_LOG_ROWS):
    path = os.path.join(DATA_DIR, FILES["user_logs"])
    chunks = []
    total_rows = 0
    logger.info("Loading user logs in chunks...")
    for chunk in pd.read_csv(path, chunksize=1_000_000):
        if users is not None:
            chunk = chunk[chunk["msno"].isin(users)]
        chunks.append(chunk)
        total_rows += len(chunk)
        if max_rows and total_rows >= max_rows:
            break
    logger.info(f"Loaded {total_rows} rows of user logs.")
    return pd.concat(chunks, ignore_index=True)

def load_members(users=None):
    df = load_csv(FILES["members"])
    if users is not None:
        df = df[df["msno"].isin(users)]
        logger.info(f"Filtered members to {len(df)} rows for sampled users.")
    return df

def load_all_data(sample=SAMPLE, sample_size=SAMPLE_SIZE):
    train = load_train_data(sample=sample, sample_size=sample_size)
    users = train["msno"].unique()
    transactions = load_transactions(users)
    logs = load_user_logs(users)
    members = load_members(users)
    return train, transactions, logs, members
