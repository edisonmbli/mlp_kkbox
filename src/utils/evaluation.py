# src/utils/evaluation.py

import os
import pandas as pd

def save_validation_outputs(model_type, y_val, y_pred, val_ids):
    os.makedirs("outputs/val_preds", exist_ok=True)
    pred_df = pd.DataFrame({
        "msno": val_ids,
        "is_churn_pred": y_pred
    })
    pred_df.to_csv(f"outputs/val_preds/{model_type}_val.csv", index=False)

    label_path = "outputs/val_labels.csv"
    if not os.path.exists(label_path):
        label_df = pd.DataFrame({
            "msno": val_ids,
            "is_churn": y_val
        })
        label_df.to_csv(label_path, index=False)

def print_dataset_stats(name, df, label_col="is_churn", date_col="last_expire_date"):
    """
    打印数据集的基本信息，包括样本数、标签分布、到期时间范围等。
    参数：
        - name: 数据集名称（如 "train", "val"）
        - df: 包含标签和到期时间的 DataFrame
        - label_col: 标签列名，默认 "is_churn"
        - date_col: 到期时间列名，默认 "last_expire_date"
    """
    print(f"\n📊 [{name}] 样本数: {len(df)}")
    if label_col in df.columns:
        label_counts = df[label_col].value_counts().to_dict()
        print(f"🟢 [{name}] 标签分布: {label_counts}")
    if date_col in df.columns:
        print(f"📆 [{name}] 到期时间范围: {df[date_col].min()} ~ {df[date_col].max()}")
