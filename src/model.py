# src/model.py

import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm.callback import early_stopping

def save_validation_outputs(model_type, y_val, y_pred, val_ids):
    os.makedirs("outputs/val_preds", exist_ok=True)
    pred_df = pd.DataFrame({
        "msno": val_ids,
        "is_churn_pred": y_pred
    })
    pred_df.to_csv(f"outputs/val_preds/{model_type}_val.csv", index=False)

    # 保存标签（只保存一次）
    label_path = "outputs/val_labels.csv"
    if not os.path.exists(label_path):
        label_df = pd.DataFrame({
            "msno": val_ids,
            "is_churn": y_val
        })
        label_df.to_csv(label_path, index=False)

def train_model(X, y, model_type="lightgbm", categorical_features=None, params=None, test_size=0.2, random_state=42, val_ids=None):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = None
    y_pred = None

    # 加载优化过的最佳模型(如有)
    if params is None:
        param_path = f"params/{model_type}_best_params.json"
        if os.path.exists(param_path):
            with open(param_path, "r") as f:
                params = json.load(f)
            print(f"📥 Loaded best params from {param_path}")
        else:
            print(f"⚠️ No saved params found for {model_type}, using default.")
            params = {}

    if model_type == "lightgbm":
        # LightGBM 要求分类特征为 category 类型
        for col in categorical_features or []:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        model = LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            categorical_feature=categorical_features,
            verbose=100,
            **(params or {})
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50)]
        )
        y_pred = model.predict_proba(X_val)[:, 1]

    elif model_type == "xgboost":
        # XGBoost 要求所有特征为数值类型，分类特征需提前编码
        # 将分类特征 label encode 为整数
        for col in categorical_features or []:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_val[col] = X_val[col].astype("category").cat.codes

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=1000,
            verbose=100,
            early_stopping=50,
            **(params or {})
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            # callbacks=[early_stopping(50)]
        )
        y_pred = model.predict_proba(X_val)[:, 1]

    elif model_type == "catboost":
        # CatBoost 要求分类特征为 string 类型
        for col in categorical_features or []:
            X_train[col] = X_train[col].astype(str).fillna("missing")
            X_val[col] = X_val[col].astype(str).fillna("missing")

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            cat_features=categorical_features or [],
            verbose=100,
            iterations=1000,
            **(params or {})
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        y_pred = model.predict_proba(X_val)[:, 1]

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logloss = log_loss(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print(f"[{model_type}] Log Loss: {logloss:.5f}, AUC: {auc:.5f}")

    # 假设你在 train_test_split 时保留了 val_ids
    val_ids = X_val.index if "msno" not in X_val.columns else X_val["msno"]
    if val_ids is not None:
        save_validation_outputs(model_type, y_val, y_pred, val_ids)

    return model, y_pred, y_val, {"logloss": logloss, "auc": auc}
