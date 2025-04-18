import lightgbm as lgb
import pandas as pd
import joblib
from lightgbm.callback import early_stopping
from src.utils.preprocessing import align_categorical_features

class LightGBMModel:
    def __init__(self, params=None, categorical_features=None):
        self.params = params or {}
        self.categorical_features = categorical_features or []
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        for col in self.categorical_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(50)]
        )

    def prepare_input(self, X_ref, X_new):
        for col in self.categorical_features:
            if col in X_ref.columns:
                X_ref[col] = X_ref[col].astype("category")
        return align_categorical_features(X_ref, X_new, self.categorical_features)

    def predict_proba(self, X, X_ref=None):
        if X_ref is not None:
            X = self.prepare_input(X_ref, X)
        else:
            for col in self.categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype("category")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
