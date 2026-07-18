import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


class MLEngine:
    def __init__(self, config: dict):
        self.config = config["detection"]["supervised"]
        self.rf = None
        self.svm = None
        self.ensemble = None

    def build_models(self):
        rf_cfg = self.config["random_forest"]
        self.rf = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
        self.ensemble = self.rf

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def partial_retrain(self, X_new, y_new):
        self.rf.fit(X_new, y_new)
