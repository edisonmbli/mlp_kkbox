import argparse
import os
import json
import pandas as pd
from datetime import datetime

from src.data_loader import load_all_data
from src.model_router import load_model_class, load_model_config
from src.utils.preprocessing import generate_test_features
from src.model_registry import MODEL_REGISTRY

MODEL_DIR = "outputs/models"
SUBMISSION_DIR = "outputs/submissions"
SAMPLE_SUB_PATH = "data/raw/sample_submission_v2.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Which model to predict")
    parser.add_argument("--sample", type=bool, default=False)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--submission_name", type=str, default=None)
    args = parser.parse_args()

    print(f"🚀 Running prediction with model: {args.model_type}")

    model_class = load_model_class(args.model_type)
    config = load_model_config(args.model_type)

    train_df, transactions_df, logs_df, members_df = load_all_data(sample=args.sample, sample_size=args.sample_size)
    test_features = generate_test_features(SAMPLE_SUB_PATH, members_df, transactions_df, logs_df)
    X_test = test_features.drop(columns=["msno", "is_churn"])

    train_features = generate_test_features(SAMPLE_SUB_PATH, members_df, transactions_df, logs_df)
    X_ref = train_features.drop(columns=["msno", "is_churn"])

    with open(f"params/{args.model_type}_best_params.json", "r") as f:
        best_params = json.load(f)

    model = model_class(params=best_params, categorical_features=config.get("categorical_features"))
    model_path = os.path.join(MODEL_DIR, f"{args.model_type}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
    model.load(model_path)

    y_test_pred = model.predict_proba(X_test, X_ref=X_ref)
    submission = pd.DataFrame({
        "msno": test_features["msno"],
        "is_churn": y_test_pred
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = args.submission_name or f"submission_{args.model_type}_{timestamp}.csv"
    submission_path = os.path.join(SUBMISSION_DIR, filename)
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")

if __name__ == "__main__":
    main()
