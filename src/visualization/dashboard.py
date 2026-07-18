import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Dashboard:
    def __init__(self):
        sns.set_theme(style="whitegrid")

    def plot_shap_feature_importance(self, shap_values: dict, save_path: str = None):
        features = list(shap_values["top_features"].keys())
        values = list(shap_values["top_features"].values())

        plt.figure(figsize=(10, 6))
        plt.barh(features, values, color="steelblue")
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_roc_curves(self, fpr_list: list, tpr_list: list,
                        labels: list, save_path: str = None):
        plt.figure(figsize=(8, 6))
        for fpr, tpr, label in zip(fpr_list, tpr_list, labels):
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_fpr_over_time(self, cycles: list, fpr_values: list,
                           save_path: str = None):
        plt.figure(figsize=(8, 5))
        plt.plot(cycles, fpr_values, marker="o", linestyle="-", color="crimson")
        plt.xlabel("Retraining Cycle")
        plt.ylabel("False Positive Rate")
        plt.title("FPR Reduction via Feedback Loop")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_comparison_table(self, metrics_dict: dict, save_path: str = None):
        df = pd.DataFrame(metrics_dict).T
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        table = ax.table(
            cellText=df.round(4).values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        plt.title("Performance Comparison: Baseline vs HC-IDF")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
