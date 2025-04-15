# src/config.py

import os

# 根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# 日志路径
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")

# 采样设置
SAMPLE = True
SAMPLE_SIZE = 10000
RANDOM_SEED = 42
MAX_LOG_ROWS = 5_000_000  # 限制日志加载量（可选）

# 文件名
FILES = {
    "train": "train_v2.csv",
    "transactions": "transactions_v2.csv",
    "user_logs": "user_logs_v2.csv",
    "members": "members_v3.csv"
}