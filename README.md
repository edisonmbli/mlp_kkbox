# KKBox Churn Prediction Pipeline

本项目基于 Kaggle 的 [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) 构建，目标是预测用户在会员到期后 30 天内是否会续订服务。项目实现了从数据加载、特征工程、模型训练、调参、融合到提交文件生成的全流程自动化建模系统。

---

## 📁 项目结构
kkbox-churn-prediction/
├── data/
│   └── raw/                      # 原始数据（如 train_v2.csv, user_logs_v2.csv 等）
├── outputs/
│   ├── models/                  # 保存训练好的模型
│   ├── preds/                   # 保存测试集预测结果
│   ├── val_preds/               # 保存验证集预测结果
│   ├── submissions/             # 最终提交文件
│   └── val_labels.csv           # 验证集真实标签
├── params/                      # 每个模型的最佳参数 JSON 文件
├── ensemble_weights.yaml        # 融合权重配置（可手动或自动生成）
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── tune_model.py
│   ├── auto_pipeline.py
│   └── optimize_ensemble.py
---

## 🚀 快速开始

### 1. 准备数据

将以下文件解压并放入 `data/raw/` 目录：

- `train_v2.csv`
- `transactions_v2.csv`
- `user_logs_v2.csv`
- `members_v3.csv`
- `sample_submission_v2.csv`

### 2. 可选：调参（首次或参数更新时）

```bash
python -m src.tune_model --model_type lightgbm --n_trials 30 --sample False
python -m src.tune_model --model_type xgboost --n_trials 30 --sample False
python -m src.tune_model --model_type catboost --n_trials 30 --sample False

调参结果将保存至 ‎⁠params/{model_type}_best_params.json⁠。

3. 主流程：自动训练 + 预测 + 融合 + 提交文件生成python -m src.auto_pipeline

该命令将：
	•	自动调参（如未调过）；
	•	使用最佳参数训练模型；
	•	保存验证集预测结果和标签；
	•	对测试集进行预测；
	•	从 ‎⁠ensemble_weights.yaml⁠ 读取融合权重（如无则使用等权重）；
	•	生成提交文件至 ‎⁠outputs/submissions/⁠。

4. 可选：优化融合权重（基于验证集）

如果你希望根据验证集表现自动优化 LightGBM、XGBoost 和 CatBoost 的融合权重，可以执行：python -m src.optimize_ensemble

该命令会：
	•	加载每个模型在验证集上的预测结果（位于 ‎⁠outputs/val_preds/⁠）；
	•	加载验证集真实标签（位于 ‎⁠outputs/val_labels.csv⁠）；
	•	使用 Optuna 搜索最优融合权重（以 log loss 最小为目标）；
	•	输出最优权重组合，并保存为 ‎⁠ensemble_weights_optimized.yaml⁠。

注意：该文件不会自动替换主流程使用的 ‎⁠ensemble_weights.yaml⁠。

如果你希望在下一次运行 ‎⁠auto_pipeline.py⁠ 时使用优化后的权重，请手动执行：mv ensemble_weights_optimized.yaml ensemble_weights.yaml

然后重新运行融合与提交流程（不需要重新训练模型）：python -m src.auto_pipeline

📌 模型支持
	•	LightGBM
	•	XGBoost
	•	CatBoost

每个模型均支持：
	•	自动调参（Optuna）
	•	分类特征处理
	•	验证集评估与 early stopping
	•	模型保存与加载

📈 特征工程亮点
	•	用户画像特征（城市、性别、注册方式、年龄等）
	•	交易行为特征（是否自动续订、付款金额、订阅周期等）
	•	时间差特征（注册到首次订阅、上次到期到本次交易间隔等）
	•	用户活跃度特征（听歌时长、完整播放比例、活跃天数等）
	•	滚动窗口特征（最近 7 天、30 天行为）
	•	行为趋势特征（活跃度变化率、是否变懒）
	•	交叉特征与缺失值标记

📄 提交文件格式

最终提交文件位于 ‎⁠outputs/submissions/⁠，格式如下：msno,is_churn
user_0001,0.0234
user_0002,0.9812
...

📚 依赖环境
	•	Python 3.8+
	•	pandas, numpy, scikit-learn
	•	lightgbm, xgboost, catboost
	•	optuna
	•	pyyaml
	•	joblib

建议使用 conda 或 virtualenv 创建独立环境。

🧠 后续可扩展方向
	•	多模型 stacking 融合
	•	时间序列建模（如 LSTM）
	•	用户分群建模（segmentation + per-group model）
	•	自动提交到 Kaggle（使用 kaggle API）
	•	封装为 CLI 工具或一键 ‎⁠.sh⁠ 脚本

🙌 致谢

本项目基于 KKBox 提供的真实用户行为数据，感谢 WSDM Cup 和 Kaggle 社区的支持与启发。