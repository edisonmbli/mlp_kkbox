import pandas as pd
import joblib
from xgboost import XGBClassifier
from xgboost import callback
from xgboost.callback import EarlyStopping
from src.utils.preprocessing import align_categorical_features

class XGBoostModel:
    # Note: early_stopping_rounds is passed via params due to environment-specific compatibility issues.
    # In some environments, passing it directly to .fit() may raise TypeError.
    def __init__(self, params=None, categorical_features=None):
        self.params = params or {}
        self.categorical_features = categorical_features or []
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        for col in self.categorical_features:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_val[col] = X_val[col].astype("category").cat.codes

        self.model = XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def prepare_input(self, X):
        for col in self.categorical_features:
            X[col] = X[col].astype("category").cat.codes
        return X

    def predict_proba(self, X, X_ref=None):
        X = self.prepare_input(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
