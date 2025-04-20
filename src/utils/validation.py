import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("validation")

def split_by_expire_month(
    train_df,
    transactions,
    train_month="2017-02",
    val_month="2017-03",
    sample=False,
    sample_size=10000,
    random_state=42
):
    """
    按照 membership_expire_date 所在月份划分训练集和验证集。
    每个用户只保留该月内的最后一次到期记录，且训练集和验证集用户不重叠。
    参数：
        - train_df: 包含 msno 和 is_churn 的 DataFrame（通常为 train_v2.csv）
        - transactions: 合并后的完整交易数据 DataFrame
    """
    # 确保日期格式正确
    transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], errors="coerce", format="%Y%m%d")

    # 构建训练集
    train_start = pd.to_datetime(f"{train_month}-01")
    train_end = train_start + pd.offsets.MonthEnd(0)
    train_tx = transactions[
        (transactions["membership_expire_date"] >= train_start) &
        (transactions["membership_expire_date"] <= train_end)
    ]
    train_last = train_tx.groupby("msno")["membership_expire_date"].max().reset_index()
    train_last.columns = ["msno", "last_expire_date"]
    train_users = set(train_last["msno"])
    train_set = train_last.merge(train_df, on="msno", how="left")

    # 构建验证集（排除训练集用户）
    val_start = pd.to_datetime(f"{val_month}-01")
    val_end = val_start + pd.offsets.MonthEnd(0)
    val_tx = transactions[
        (transactions["membership_expire_date"] >= val_start) &
        (transactions["membership_expire_date"] <= val_end)
    ]
    val_last = val_tx.groupby("msno")["membership_expire_date"].max().reset_index()
    val_last.columns = ["msno", "last_expire_date"]
    val_last = val_last[~val_last["msno"].isin(train_users)]
    val_set = val_last.merge(train_df, on="msno", how="left")

    # 打印划分信息
    logger.info(f"📆 训练集到期范围: {train_start.date()} ~ {train_end.date()}，样本数: {len(train_set)}")
    logger.info(f"📆 验证集到期范围: {val_start.date()} ~ {val_end.date()}，样本数: {len(val_set)}")

    # 可选采样
    if sample:
        train_set = train_set.sample(n=min(sample_size, len(train_set)), random_state=random_state)
        val_set = val_set.sample(n=min(int(sample_size * 0.25), len(val_set)), random_state=random_state)
        logger.info(f"🎯 采样后: train={len(train_set)}, val={len(val_set)}")

    # 防御性断言
    assert len(train_set) > 0, "❌ 训练集为空，请检查月份或数据分布"
    assert len(val_set) > 0, "❌ 验证集为空，请检查月份或数据分布"

    return train_set, val_set
