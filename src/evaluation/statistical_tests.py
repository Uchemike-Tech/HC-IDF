import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


class StatisticalTests:
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def mcnemar_test(self, y_true, pred_model_a, pred_model_b) -> dict:
        cm = confusion_matrix(
            (pred_model_a == y_true).astype(int),
            (pred_model_b == y_true).astype(int),
        )
        b = cm[0, 1]
        c = cm[1, 0]
        n = b + c

        if n == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        stat = (abs(b - c) - 1) ** 2 / n
        p_value = 1 - stats.chi2.cdf(stat, df=1)

        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def paired_ttest(self, scores_a: list, scores_b: list) -> dict:
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def friedman_test(self, *score_vectors) -> dict:
        stat, p_value = stats.friedmanchisquare(*score_vectors)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def cohens_kappa(self, y_true, y_pred) -> float:
        cm = confusion_matrix(y_true, y_pred)
        n = np.sum(cm)
        p_o = np.trace(cm) / n
        p_e = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n ** 2)
        return float((p_o - p_e) / (1 - p_e))
