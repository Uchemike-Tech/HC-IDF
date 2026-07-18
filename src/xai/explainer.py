import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd


class XAIExplainer:
    def __init__(self, config: dict):
        self.config = config["xai"]
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None

    def fit_shap(self, model, X_train: pd.DataFrame):
        if self.config["shap"]["enabled"]:
            self.shap_explainer = shap.TreeExplainer(model)
            self.feature_names = X_train.columns.tolist()

    def fit_lime(self, model, X_train: pd.DataFrame):
        if self.config["lime"]["enabled"]:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["Normal", "Malicious"],
                mode="classification",
            )

    def explain_shap(self, X: pd.DataFrame) -> dict:
        if self.shap_explainer is None:
            return {}
        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_importance = np.abs(shap_values).mean(axis=0)
        top_k = self.config["shap"]["max_display"]
        top_indices = np.argsort(feature_importance)[-top_k:]

        return {
            "shap_values": shap_values,
            "top_features": {
                self.feature_names[i]: float(feature_importance[i])
                for i in reversed(top_indices)
            },
        }

    def explain_lime(self, X_instance: np.ndarray) -> dict:
        if self.lime_explainer is None:
            return {}
        num_features = self.config["lime"]["num_features"]
        exp = self.lime_explainer.explain_instance(
            X_instance,
            lambda x: np.column_stack([1 - x, x]),
            num_features=num_features,
        )
        return {
            "lime_weights": dict(exp.as_list()),
            "lime_html": exp.as_html(),
        }

    def generate_narrative(self, top_features: dict, alert_type: str = None) -> str:
        parts = []
        for feature, weight in sorted(top_features.items(), key=lambda x: -x[1])[:3]:
            parts.append(f"{feature} ({weight:.1%} contribution)")
        narrative = f"Alert triggered primarily by: {', '.join(parts)}."
        if alert_type:
            narrative += f" Classification: {alert_type}."
        return narrative
