# src/pipeline.py

from src.data_loader import load_all_data
from src.feature_engineering import merge_all_features
from src.model import train_model
from src.evaluate import evaluate_model
from src.config import SAMPLE, SAMPLE_SIZE

def run_pipeline(model_type="lightgbm", sample=SAMPLE, sample_size=SAMPLE_SIZE):
    print(f"🚀 Running pipeline with model: {model_type}, sample={sample}, sample_size={sample_size}")

    # Step 1: 加载数据
    train_df, transactions_df, logs_df, members_df = load_all_data(sample=sample, sample_size=sample_size)

    # Step 2: 特征工程
    features = merge_all_features(train_df, members_df, transactions_df, logs_df)
    X = features.drop(columns=["msno", "is_churn"])
    y = features["is_churn"]

    # Step 3: 模型训练
    categorical_cols = ["city", "gender", "registered_via", "reg_year", "reg_month"]
    model, y_pred, y_val, metrics = train_model(X, y, model_type=model_type, categorical_features=categorical_cols)

    # Step 4: 模型评估（使用验证集标签）
    evaluate_model(y_true=y_val, y_pred=y_pred, model=model, feature_names=X.columns.tolist(), model_type=model_type)

    return model, metrics
