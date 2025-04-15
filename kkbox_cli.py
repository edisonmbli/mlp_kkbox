# kkbox_cli.py

import argparse
import subprocess
import sys
import os

def run_tune(model_type, n_trials=30, sample=False):
    cmd = [
        "python", "-m", "src.tune_model",
        "--model_type", model_type,
        "--n_trials", str(n_trials),
        "--sample", str(sample)
    ]
    subprocess.run(cmd)

def run_train():
    subprocess.run(["python", "-m", "src.auto_pipeline"])

def run_ensemble():
    subprocess.run(["python", "-m", "src.optimize_ensemble"])
    if os.path.exists("ensemble_weights_optimized.yaml"):
        os.replace("ensemble_weights_optimized.yaml", "ensemble_weights.yaml")
        print("✅ Updated ensemble_weights.yaml with optimized weights.")

def run_submit():
    # 需要提前配置 kaggle API
    from datetime import datetime
    latest_file = sorted(os.listdir("outputs/submissions"))[-1]
    submission_path = os.path.join("outputs/submissions", latest_file)
    message = f"Auto submission at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", "kkbox-churn-prediction-challenge",
        "-f", submission_path,
        "-m", message
    ])

def main():
    parser = argparse.ArgumentParser(description="KKBox Churn Prediction CLI")
    subparsers = parser.add_subparsers(dest="command")

    # tune
    tune_parser = subparsers.add_parser("tune", help="调参指定模型")
    tune_parser.add_argument("--model", required=True, choices=["lightgbm", "xgboost", "catboost"])
    tune_parser.add_argument("--n_trials", type=int, default=30)
    tune_parser.add_argument("--sample", type=bool, default=False)

    # train
    subparsers.add_parser("train", help="运行完整 pipeline")

    # ensemble
    subparsers.add_parser("ensemble", help="优化融合权重并更新配置")

    # submit
    subparsers.add_parser("submit", help="提交最新的 submission 文件到 Kaggle")

    # version
    subparsers.add_parser("version", help="查看版本信息")

    args = parser.parse_args()

    if args.command == "tune":
        run_tune(args.model, args.n_trials, args.sample)
    elif args.command == "train":
        run_train()
    elif args.command == "ensemble":
        run_ensemble()
    elif args.command == "submit":
        run_submit()
    elif args.command == "version":
        print("KKBox CLI v1.0.0")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
