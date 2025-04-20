#!/bin/bash

echo "🚀 Step 1: Tuning models (LightGBM, XGBoost, CatBoost)..."
python -m src.tune_model --model_type lightgbm --n_trials 30 --sample False
python -m src.tune_model --model_type xgboost --n_trials 30 --sample False
python -m src.tune_model --model_type catboost --n_trials 30 --sample False

echo "🧠 Step 2: Running full pipeline (train + predict + blend)..."
python -m src.auto_pipeline

echo "🔍 Step 3: Optimizing ensemble weights based on validation set..."
python -m src.optimize_ensemble

echo "📦 Step 4: Applying optimized weights..."
mv ensemble_weights_optimized.yaml ensemble_weights.yaml

echo "📈 Step 5: Re-running pipeline with updated weights (no retraining)..."
python -m src.auto_pipeline

echo "📈 Step 6: Running prediction for each model..."
python -m src.predict --model_type lightgbm
python -m src.predict --model_type xgboost
python -m src.predict --model_type catboost

echo "✅ All steps completed. Check outputs/submissions/ for your final submission file."
