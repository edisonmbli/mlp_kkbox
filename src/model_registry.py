from src.models.lightgbm_model import LightGBMModel
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "xgboost": XGBoostModel
}
