import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)


class MetricsCalculator:
    def __init__(self):
        self.results = {}

    def compute_all(self, y_true, y_pred, y_proba=None):
        self.results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": self._compute_fpr(y_true, y_pred),
            "false_negative_rate": self._compute_fnr(y_true, y_pred),
        }
        if y_proba is not None:
            self.results["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        return self.results

    def _compute_fpr(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    def _compute_fnr(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0

    def compute_mitm_specific(self, y_true, y_pred, mitm_mask):
        mask = mitm_mask == 1
        if mask.sum() == 0:
            return {"mitm_precision": 0, "mitm_recall": 0, "mitm_f1": 0}
        return {
            "mitm_precision": precision_score(y_true[mask], y_pred[mask], zero_division=0),
            "mitm_recall": recall_score(y_true[mask], y_pred[mask], zero_division=0),
            "mitm_f1": f1_score(y_true[mask], y_pred[mask], zero_division=0),
        }

    def compute_detection_latency(self, timestamps: list) -> dict:
        latencies = np.diff(timestamps) * 1000
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "min_latency_ms": float(np.min(latencies)),
        }

    def summary(self):
        return self.results
