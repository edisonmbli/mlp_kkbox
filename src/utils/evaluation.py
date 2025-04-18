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
