import numpy as np
import pandas as pd
from collections import deque


class FeedbackLoop:
    def __init__(self, config: dict):
        self.config = config["feedback"]
        self.true_positive_buffer = deque(maxlen=config["feedback"]["max_false_positive_buffer"])
        self.false_positive_buffer = deque(maxlen=config["feedback"]["max_false_positive_buffer"])
        self.retraining_count = 0

    def record_decision(self, prediction: int, actual: int,
                        features: pd.DataFrame, analyst_verdict: str):
        if analyst_verdict == "confirmed_tp":
            self.true_positive_buffer.append((features, prediction))
        elif analyst_verdict == "dismissed_fp":
            self.false_positive_buffer.append((features, prediction))

    def should_retrain(self) -> bool:
        total = len(self.true_positive_buffer) + len(self.false_positive_buffer)
        return total >= self.config["retraining_interval"]

    def get_retraining_data(self):
        X_retrain = []
        y_retrain = []

        for features, _ in self.true_positive_buffer:
            X_retrain.append(features)
            y_retrain.append(1)

        for features, _ in self.false_positive_buffer:
            X_retrain.append(features)
            y_retrain.append(0)

        if not X_retrain:
            return None, None

        X_retrain = np.vstack(X_retrain) if isinstance(X_retrain[0], np.ndarray) else np.array(X_retrain)
        y_retrain = np.array(y_retrain)

        self.true_positive_buffer.clear()
        self.false_positive_buffer.clear()
        self.retraining_count += 1

        return X_retrain, y_retrain

    def get_feedback_stats(self) -> dict:
        return {
            "tp_buffer_size": len(self.true_positive_buffer),
            "fp_buffer_size": len(self.false_positive_buffer),
            "retraining_cycles": self.retraining_count,
        }
