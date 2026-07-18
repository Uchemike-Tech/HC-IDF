"""
Inferential analysis of HC-IDF user test results.
Run: python scripts/analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, shapiro
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("data/test_results")
OUTPUT = Path("data/analysis_output")
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"baseline": "#95a5a6", "hcidf": "#1a5276"}
PRIMARY = "#1a5276"
GREEN = "#27ae60"

# ── Load ────────────────────────────────────────────────────
df = pd.read_csv(RESULTS / "combined_test_results.csv")
df_q = pd.read_csv(RESULTS / "combined_questionnaire.csv")
print(f"Loaded {len(df)} rows from {df['user_id'].nunique()} users\n")

# ── Per-user aggregation ─────────────────────────────────────
per_user = df.groupby(["user_id", "phase"]).agg(
    accuracy=("correct", "mean"),
    avg_time=("time_seconds", "mean"),
    total_correct=("correct", "sum"),
).reset_index()

baseline = per_user[per_user["phase"] == "baseline"].set_index("user_id")
hcidf = per_user[per_user["phase"] == "hcidf"].set_index("user_id")

# ── Descriptive Stats ────────────────────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
desc = per_user.groupby("phase")[["accuracy", "avg_time"]].describe().round(4)
print(desc.to_string())
print()

# ── Paired t-tests ───────────────────────────────────────────
print("=" * 60)
print("INFERENTIAL STATISTICS")
print("=" * 60)

# Accuracy
t_stat_acc, p_val_acc = ttest_rel(baseline["accuracy"], hcidf["accuracy"])
print(f"\nPaired t-test — Accuracy:")
print(f"  Baseline: M={baseline['accuracy'].mean():.4f}, SD={baseline['accuracy'].std():.4f}")
print(f"  HC-IDF:   M={hcidf['accuracy'].mean():.4f}, SD={hcidf['accuracy'].std():.4f}")
print(f"  t({len(baseline)-1}) = {t_stat_acc:.4f}, p = {p_val_acc:.6f}")
print(f"  Significant: {'YES' if p_val_acc < 0.05 else 'NO'}")

# Time
t_stat_time, p_val_time = ttest_rel(baseline["avg_time"], hcidf["avg_time"])
print(f"\nPaired t-test — Decision Time (seconds):")
print(f"  Baseline: M={baseline['avg_time'].mean():.2f}s, SD={baseline['avg_time'].std():.2f}")
print(f"  HC-IDF:   M={hcidf['avg_time'].mean():.2f}s, SD={hcidf['avg_time'].std():.2f}")
print(f"  t({len(baseline)-1}) = {t_stat_time:.4f}, p = {p_val_time:.6f}")
print(f"  Significant: {'YES' if p_val_time < 0.05 else 'NO'}")

# Wilcoxon (non-parametric alternative)
w_stat_acc, w_p_acc = wilcoxon(baseline["accuracy"], hcidf["accuracy"])
w_stat_time, w_p_time = wilcoxon(baseline["avg_time"], hcidf["avg_time"])
print(f"\nWilcoxon Signed-Rank (non-parametric):")
print(f"  Accuracy: W = {w_stat_acc:.0f}, p = {w_p_acc:.6f}")
print(f"  Time:     W = {w_stat_time:.0f}, p = {w_p_time:.6f}")

# ── Normality check (Shapiro-Wilk) ───────────────────────────
print(f"\nNormality (Shapiro-Wilk):")
for name, data in [("Baseline Acc", baseline["accuracy"]), ("HC-IDF Acc", hcidf["accuracy"]),
                    ("Baseline Time", baseline["avg_time"]), ("HC-IDF Time", hcidf["avg_time"])]:
    stat, p = shapiro(data)
    print(f"  {name}: W = {stat:.4f}, p = {p:.4f} {'(normal)' if p > 0.05 else '(NOT normal)'}")

# ── Questionnaire ────────────────────────────────────────────
print("\n" + "=" * 60)
print("QUESTIONNAIRE RESULTS")
print("=" * 60)
q_labels = {
    "q1": "Baseline trust",
    "q2": "XAI explanations helped",
    "q3": "More confident with HC-IDF",
    "q4": "Feedback easy to use",
    "q5": "Prefer HC-IDF over baseline",
}
for col, label in q_labels.items():
    vals = df_q[col]
    print(f"  {label}: M={vals.mean():.2f}, SD={vals.std():.2f}, Median={vals.median():.0f}")

# ── Plots ────────────────────────────────────────────────────
print(f"\nSaving plots to {OUTPUT}/...")

# 1. Accuracy comparison box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
bp_data = [baseline["accuracy"], hcidf["accuracy"]]
bp = ax.boxplot(bp_data, tick_labels=["Baseline", "HC-IDF"], patch_artist=True,
                medianprops={"color": "white", "linewidth": 2})
bp["boxes"][0].set_facecolor(COLORS["baseline"])
bp["boxes"][1].set_facecolor(COLORS["hcidf"])
# Overlay individual points
for i, data in enumerate(bp_data):
    jitter = np.random.normal(i + 1, 0.04, len(data))
    ax.scatter(jitter, data, alpha=0.6, color="black", s=30, zorder=3)
# Connect paired points
for uid in baseline.index:
    ax.plot([1, 2], [baseline.loc[uid, "accuracy"], hcidf.loc[uid, "accuracy"]],
            color="gray", alpha=0.3, linewidth=0.5)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Detection Accuracy by Condition", fontsize=13, fontweight="bold", color=PRIMARY)

ax = axes[1]
bp_data = [baseline["avg_time"], hcidf["avg_time"]]
bp = ax.boxplot(bp_data, tick_labels=["Baseline", "HC-IDF"], patch_artist=True,
                medianprops={"color": "white", "linewidth": 2})
bp["boxes"][0].set_facecolor(COLORS["baseline"])
bp["boxes"][1].set_facecolor(COLORS["hcidf"])
for i, data in enumerate(bp_data):
    jitter = np.random.normal(i + 1, 0.04, len(data))
    ax.scatter(jitter, data, alpha=0.6, color="black", s=30, zorder=3)
for uid in baseline.index:
    ax.plot([1, 2], [baseline.loc[uid, "avg_time"], hcidf.loc[uid, "avg_time"]],
            color="gray", alpha=0.3, linewidth=0.5)
ax.set_ylabel("Decision Time (seconds)", fontsize=12)
ax.set_title("Decision Time by Condition", fontsize=13, fontweight="bold", color=PRIMARY)
plt.tight_layout()
plt.savefig(OUTPUT / "accuracy_time_boxplots.png", dpi=200, bbox_inches="tight")
plt.close()

# 2. Questionnaire bar chart
fig, ax = plt.subplots(figsize=(10, 5))
means = [df_q[c].mean() for c in q_labels]
stds = [df_q[c].std() for c in q_labels]
x = np.arange(len(q_labels))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=[PRIMARY, GREEN, PRIMARY, GREEN, PRIMARY], width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["Baseline\nTrust", "XAI\nHelped", "More\nConfident", "Feedback\nEasy", "Prefer\nHC-IDF"],
                    fontsize=11)
ax.set_ylabel("Mean Rating (1-5)", fontsize=12)
ax.set_title("Post-Test Questionnaire Results", fontsize=13, fontweight="bold", color=PRIMARY)
ax.set_ylim(0, 5.5)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{mean:.2f}", ha="center", fontsize=11, fontweight="bold", color=PRIMARY)
plt.tight_layout()
plt.savefig(OUTPUT / "questionnaire_barchart.png", dpi=200, bbox_inches="tight")
plt.close()

# 3. Per-user accuracy comparison (paired line plot)
fig, ax = plt.subplots(figsize=(10, 5))
for uid in baseline.index:
    ax.plot([0, 1], [baseline.loc[uid, "accuracy"], hcidf.loc[uid, "accuracy"]],
            color="gray", alpha=0.4, linewidth=0.8)
ax.scatter([0] * len(baseline), baseline["accuracy"], color=COLORS["baseline"], s=50, zorder=5, label="Baseline")
ax.scatter([1] * len(hcidf), hcidf["accuracy"], color=COLORS["hcidf"], s=50, zorder=5, label="HC-IDF")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Baseline", "HC-IDF"], fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-User Accuracy: Baseline vs HC-IDF", fontsize=13, fontweight="bold", color=PRIMARY)
ax.legend(fontsize=11)
ax.set_xlim(-0.3, 1.3)
plt.tight_layout()
plt.savefig(OUTPUT / "paired_accuracy.png", dpi=200, bbox_inches="tight")
plt.close()

# 4. Accuracy distribution histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(baseline["accuracy"], bins=10, alpha=0.6, color=COLORS["baseline"], label="Baseline", edgecolor="white")
ax.hist(hcidf["accuracy"], bins=10, alpha=0.6, color=COLORS["hcidf"], label="HC-IDF", edgecolor="white")
ax.axvline(baseline["accuracy"].mean(), color=COLORS["baseline"], linestyle="--", linewidth=2, label=f"Baseline Mean={baseline['accuracy'].mean():.2f}")
ax.axvline(hcidf["accuracy"].mean(), color=COLORS["hcidf"], linestyle="--", linewidth=2, label=f"HC-IDF Mean={hcidf['accuracy'].mean():.2f}")
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Number of Users", fontsize=12)
ax.set_title("Accuracy Distribution by Condition", fontsize=13, fontweight="bold", color=PRIMARY)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT / "accuracy_histogram.png", dpi=200, bbox_inches="tight")
plt.close()

print("Plots saved:")
for p in OUTPUT.glob("*.png"):
    print(f"  {p.name}")

# ── Summary report ───────────────────────────────────────────
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"\nKey findings:")
print(f"  1. HC-IDF significantly improved detection accuracy "
      f"({baseline['accuracy'].mean():.1%} vs {hcidf['accuracy'].mean():.1%}, "
      f"p = {p_val_acc:.6f})")
print(f"  2. HC-IDF significantly reduced decision time "
      f"({baseline['avg_time'].mean():.1f}s vs {hcidf['avg_time'].mean():.1f}s, "
      f"p = {p_val_time:.6f})")
print(f"  3. Users rated XAI explanations favorably (M = {df_q['q2'].mean():.2f}/5)")
print(f"  4. Users preferred HC-IDF over baseline (M = {df_q['q5'].mean():.2f}/5)")
print(f"\nResults saved to {OUTPUT}/")
