import yaml
import os
from src.model_registry import MODEL_REGISTRY

def load_model_class(model_type):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_REGISTRY[model_type]

def load_model_config(model_type):
    config_path = os.path.join("model_configs", f"{model_type}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
