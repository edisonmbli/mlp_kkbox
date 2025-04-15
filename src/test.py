import argparse
from src.pipeline import run_pipeline

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行KKBox用户流失预测模型')
    
    # 添加命令行参数
    parser.add_argument('--model_type', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'catboost'],
                        help='模型类型 (默认: lightgbm)')
    
    parser.add_argument('--sample', type=bool, default=True,
                        help='是否采样数据 (默认: True)')
    
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='采样大小 (默认: 10000)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行模型
    print(f"使用参数: model_type={args.model_type}, sample={args.sample}, sample_size={args.sample_size}")
    model, metrics = run_pipeline(
        model_type=args.model_type,
        sample=args.sample,
        sample_size=args.sample_size
    )
    
    print(f"模型评估指标: {metrics}")
    return model, metrics

if __name__ == "__main__":
    main()
