from src.model_router import load_model_class, load_model_config

def train_model(X, y, model_type="lightgbm", test_size=0.2, random_state=42, val_ids=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, roc_auc_score

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    config = load_model_config(model_type)
    model_class = load_model_class(model_type)
    model = model_class(params=config.get("default_params"), categorical_features=config.get("categorical_features"))
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict_proba(X_val)
    logloss = log_loss(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print(f"[{model_type}] Log Loss: {logloss:.5f}, AUC: {auc:.5f}")

    if val_ids is not None:
        from src.utils.evaluation import save_validation_outputs
        save_validation_outputs(model_type, y_val, y_pred, val_ids)

    return model, y_pred, y_val, {"logloss": logloss, "auc": auc}
