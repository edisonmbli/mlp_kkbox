import pandas as pd
import joblib
from catboost import CatBoostClassifier
from src.utils.preprocessing import align_categorical_features

class CatBoostModel:
    def __init__(self, params=None, categorical_features=None):
        self.params = params or {}
        self.categorical_features = categorical_features or []
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        for col in self.categorical_features:
            X_train[col] = X_train[col].astype(str).fillna("missing")
            X_val[col] = X_val[col].astype(str).fillna("missing")

        self.model = CatBoostClassifier(
            cat_features=self.categorical_features,
            **self.params
        )
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

    def prepare_input(self, X):
        for col in self.categorical_features:
            X[col] = X[col].astype(str).fillna("missing")
        return X

    def predict_proba(self, X, X_ref=None):
        X = self.prepare_input(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)
