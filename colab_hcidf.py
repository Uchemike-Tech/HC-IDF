# ============================================================
# PART 1 — Setup, Drive Mount, Install Deps, Config, Data Load
# ============================================================
# Run this entire cell first.

import os, sys, warnings, zipfile, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from google.colab import drive, files

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# --- Mount Google Drive ---
drive.mount("/content/drive")
print("Drive mounted at /content/drive/MyDrive/")

# --- Install packages (run once per session) ---
# !pip install -q imbalanced-learn shap lime pyarrow

# --- Paths ---
BASE = Path("/content/drive/MyDrive/HC-IDF")
DATA_DIR = BASE / "data"
RAW_CICIDS = DATA_DIR / "external/CICIDS2017"
RAW_UNSW   = DATA_DIR / "external/UNSW-NB15"
PROCESSED  = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
CONFIG = {
    "data": {
        "test_split": 0.15,
        "val_split": 0.15,
        "random_state": 42,
    },
    "preprocessing": {
        "normalize": True,
        "normalization_method": "zscore",
        "handle_imbalance": True,
        "sampling_strategy": "smote",
        "feature_selection": {"enabled": False, "method": "mutual_info", "k_best": 20},
    },
    "detection": {
        "supervised": {
            "random_forest": {"n_estimators": 100, "max_depth": 20},
            "svm": {"kernel": "rbf", "C": 1.0},
        },
        "deep_learning": {
            "lstm": {"units": [64, 32], "dropout": 0.3, "epochs": 20, "batch_size": 256},
            "cnn_lstm": {"filters": 64, "kernel_size": 3, "lstm_units": 32, "epochs": 20, "batch_size": 256},
        },
        "anomaly": {
            "isolation_forest": {"n_estimators": 100, "contamination": 0.1},
            "autoencoder": {"encoding_dim": 16, "epochs": 20, "batch_size": 256},
        },
    },
    "mitm": {
        "arp": {"cache_timeout": 300, "alert_threshold": 3},
        "dns": {"max_response_deviation": 50},
        "session": {"rtt_std_threshold": 2.5},
    },
    "xai": {"shap": {"enabled": True, "max_display": 10}, "lime": {"enabled": True, "num_features": 5, "num_samples": 1000}},
    "feedback": {"retraining_interval": 100, "confidence_threshold": 0.7, "max_false_positive_buffer": 500},
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc", "false_positive_rate", "false_negative_rate"],
        "statistical_tests": {"significance_level": 0.05, "tests": ["mcnemar", "paired_ttest"]},
    },
}

RS = CONFIG["data"]["random_state"]
NJOBS = os.cpu_count()
print(f"CPU cores: {NJOBS}, T4 GPU available: {os.system('nvidia-smi') == 0}")

# ============================================================
def load_data():
    parquet_path = PROCESSED / "combined_dataset.parquet"
    if parquet_path.exists():
        print(f"[DATA] Loading parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)

    print("[DATA] Parquet not found. Building from CSVs...")
    frames = []
    for f in sorted(RAW_CICIDS.glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
        frames.append(df)
        print(f"  Loaded {f.name} ({len(df):,} rows)")

    for f in sorted(RAW_UNSW.glob("*.csv")):
        df = pd.read_csv(f, low_memory=False)
        df.rename(columns={"attack_cat": "Label"}, inplace=True)
        df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
        frames.append(df)
        print(f"  Loaded {f.name} ({len(df):,} rows)")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [str(c).strip() for c in combined.columns]
    combined.to_parquet(parquet_path, index=False)
    print(f"[DATA] Saved {len(combined):,} rows to {parquet_path}")
    return combined


print("\n===== PART 1 DONE: Setup + Data Load =====")
print("Continue to Part 2 after data loads.")
# ============================================================


# ============================================================
# PART 2 — Preprocessing + RF Baseline + SVM
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

print("\n===== PART 2: Preprocessing & ML Baselines =====")

df = load_data()
print(f"[DATA] Shape: {df.shape}")
print(f"[DATA] Label distribution:\n{df['Label'].value_counts()}")

# --- Clean ---
df = df.drop_duplicates().replace([np.inf, -np.inf], np.nan)
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns.drop("Label", errors="ignore")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna("missing").astype("category").cat.codes
print(f"[CLEAN] {len(df)} rows, {len(num_cols)} numeric + {len(cat_cols)} encoded cat cols")

# --- Subsample for dev speed ---
df = df.sample(frac=0.2, random_state=RS).reset_index(drop=True)
print(f"[DEV] Subsampled to {len(df):,} rows")

# --- Feature engineering ---
drop_cols = ["Label", "Flow ID", "Timestamp", "Packet Length"] if "Packet Length" in df.columns else ["Label"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["Label"].astype(int)

# --- Normalize ---
scaler = StandardScaler()
X[:] = scaler.fit_transform(X)
print(f"[NORM] {X.shape[1]} features")

# --- SMOTE ---
smote = SMOTE(random_state=RS)
X_res, y_res = smote.fit_resample(X, y)
print(f"[SMOTE] {X_res.shape[0]} rows (balanced)")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=CONFIG["data"]["test_split"],
    random_state=RS, stratify=y_res,
)
print(f"[SPLIT] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- Random Forest ---
print("\n--- RF Training ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RS, n_jobs=NJOBS)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print(f"RF  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"RF  F1:       {f1_score(y_test, y_pred_rf):.4f}")
print(f"RF  AUC-ROC:  {roc_auc_score(y_test, y_proba_rf):.4f}")

# --- SVM (on subsample since O(n²)) ---
print("\n--- SVM Training (may take a few minutes) ---")
svm_sample = min(30000, len(X_train))
svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RS, cache_size=2000)
svm.fit(X_train[:svm_sample], y_train[:svm_sample])
y_pred_svm = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)[:, 1]
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"SVM F1:       {f1_score(y_test, y_pred_svm):.4f}")
print(f"SVM AUC-ROC:  {roc_auc_score(y_test, y_proba_svm):.4f}")


# ============================================================


# ============================================================
# PART 3 — Deep Learning (LSTM, CNN-LSTM, Autoencoder) on T4
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten,
    Reshape, Input, BatchNormalization, RepeatVector, TimeDistributed,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("\n===== PART 3: Deep Learning on T4 GPU =====")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"TF version: {tf.__version__}")

# Enable mixed precision for T4
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# --- Prepare sequences for LSTM ---
SEQ_LEN = 20
n_features = X_train.shape[1]

def to_sequences(X, y, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

X_tr_seq, y_tr_seq = to_sequences(X_train.values, y_train.values)
X_te_seq, y_te_seq = to_sequences(X_test.values, y_test.values)
print(f"[SEQ] Train: {X_tr_seq.shape}, Test: {X_te_seq.shape}")

# EPOCHS  = CONFIG["detection"]["deep_learning"]["lstm"]["epochs"]
EPOCHS = 10  # fewer for demonstration
BATCH   = CONFIG["detection"]["deep_learning"]["lstm"]["batch_size"]
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_sched   = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)

# ---- LSTM ----
print("\n--- LSTM ---")
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_tr_seq, y_tr_seq, validation_split=0.1,
               epochs=EPOCHS, batch_size=BATCH, callbacks=[early_stop, lr_sched], verbose=1)
y_pred_lstm = (lstm_model.predict(X_te_seq, verbose=0) > 0.5).astype(int).ravel()
print(f"LSTM Accuracy: {accuracy_score(y_te_seq, y_pred_lstm):.4f}")
print(f"LSTM F1:       {f1_score(y_te_seq, y_pred_lstm):.4f}")

# ---- CNN-LSTM ----
print("\n--- CNN-LSTM ---")
cnn_lstm = Sequential([
    Conv1D(64, 3, activation="relu", padding="same", input_shape=(SEQ_LEN, n_features)),
    MaxPooling1D(2),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])
cnn_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_lstm.fit(X_tr_seq, y_tr_seq, validation_split=0.1,
             epochs=EPOCHS, batch_size=BATCH, callbacks=[early_stop, lr_sched], verbose=1)
y_pred_cnn = (cnn_lstm.predict(X_te_seq, verbose=0) > 0.5).astype(int).ravel()
print(f"CNN-LSTM Accuracy: {accuracy_score(y_te_seq, y_pred_cnn):.4f}")
print(f"CNN-LSTM F1:       {f1_score(y_te_seq, y_pred_cnn):.4f}")

# ---- Autoencoder (Anomaly Detection) ----
print("\n--- Autoencoder ---")
encoding_dim = CONFIG["detection"]["anomaly"]["autoencoder"]["encoding_dim"]
input_dim = X_train.shape[1]

ae_input = Input(shape=(input_dim,))
encoded = Dense(encoding_dim * 2, activation="relu")(ae_input)
encoded = Dense(encoding_dim, activation="relu")(encoded)
decoded = Dense(encoding_dim * 2, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(ae_input, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train, X_train, validation_split=0.1,
                epochs=EPOCHS, batch_size=BATCH, callbacks=[early_stop], verbose=1)

recon_err = np.mean(np.square(X_test - autoencoder.predict(X_test, verbose=0)), axis=1)
threshold = np.percentile(recon_err[y_test == 0], 95)
y_pred_ae = (recon_err > threshold).astype(int)
print(f"Autoencoder F1: {f1_score(y_test, y_pred_ae):.4f}")

print("\n===== PART 3 DONE: Deep Learning =====")
print("Continue to Part 4 for XAI + MITM detection.")
# ============================================================


# ============================================================
# PART 4 — XAI (SHAP, LIME) + MITM Detection
# ============================================================
import shap
import lime
import lime.lime_tabular

print("\n===== PART 4: XAI & MITM Detection =====")

# --- SHAP ---
print("\n--- SHAP Explanations ---")
shap_sample = X_test[:200]
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(shap_sample)
shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals,
                  shap_sample, show=False, max_display=10)
plt_path = str(PROCESSED / "shap_summary.png")
plt.savefig(plt_path, bbox_inches="tight")
plt.close()
print(f"SHAP plot saved to {plt_path}")

# --- LIME ---
print("\n--- LIME Explanations ---")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=list(X_train.columns),
    class_names=["Benign", "Attack"], mode="classification", random_state=RS,
)
idx = 0
exp = lime_explainer.explain_instance(X_test.values[idx], rf.predict_proba, num_features=5)
lime_path = str(PROCESSED / "lime_explanation.html")
exp.save_to_file(lime_path)
print(f"LIME explanation saved to {lime_path}")
print("Top features:", [f[0] for f in exp.as_list()])

# --- MITM Detection Simulation ---
print("\n--- MITM Detection Evaluation ---")
# Simulated MITM indicators: we treat attack rows as potential MITM
mitm_mask = (y_test == 1).values
mitm_detected = (y_pred_rf == 1) & mitm_mask
mitm_precision = precision_score(y_test[y_test == 1], y_pred_rf[y_test == 1]) if mitm_mask.sum() > 0 else 0
mitm_recall = recall_score(y_test[y_test == 1], y_pred_rf[y_test == 1]) if mitm_mask.sum() > 0 else 0
mitm_f1 = f1_score(y_test[y_test == 1], y_pred_rf[y_test == 1]) if mitm_mask.sum() > 0 else 0
print(f"MITM Detection (attack-only view):")
print(f"  Precision: {mitm_precision:.4f}")
print(f"  Recall:    {mitm_recall:.4f}")
print(f"  F1-Score:  {mitm_f1:.4f}")

# ARP spoofing simulation
from collections import defaultdict
arp_cache = defaultdict(int)
arp_alerts = 0
for i in range(min(1000, len(y_test))):
    if y_test.iloc[i] == 1 and y_pred_rf[i] == 1:
        src_ip = f"192.168.1.{np.random.randint(1, 255)}"
        arp_cache[src_ip] += 1
        if arp_cache[src_ip] >= CONFIG["mitm"]["arp"]["alert_threshold"]:
            arp_alerts += 1
print(f"MITM ARP alerts triggered: {arp_alerts}")

print("\n===== PART 4 DONE: XAI + MITM =====")
print("Continue to Part 5 for feedback loop + evaluation.")
# ============================================================


# ============================================================
# PART 5 — Feedback Loop + Evaluation + Statistical Tests
# ============================================================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

print("\n===== PART 5: Human-in-the-Loop Feedback & Evaluation =====")

# --- Simulated Feedback Loop ---
print("\n--- Human-in-the-Loop Feedback ---")
feedback_pool = pd.DataFrame(X_test[:500]).reset_index(drop=True)
feedback_labels = y_test[:500].reset_index(drop=True)

feedback_buffer = {"X": [], "y": []}
retraining_count = 0
batch_size = 50

for start in range(0, len(feedback_pool), batch_size):
    end = start + batch_size
    batch_X = feedback_pool.iloc[start:end]
    batch_y = feedback_labels.iloc[start:end]
    preds = rf.predict(batch_X)

    for j in range(len(batch_X)):
        if preds[j] == batch_y.iloc[j]:
            feedback_buffer["X"].append(batch_X.iloc[j].values)
            feedback_buffer["y"].append(batch_y.iloc[j])
        # Human corrects false predictions (simulated)
        elif preds[j] != batch_y.iloc[j]:
            # Human corrects: add the correct label
            feedback_buffer["X"].append(batch_X.iloc[j].values)
            feedback_buffer["y"].append(batch_y.iloc[j])

    # Retrain when buffer is large enough
    if len(feedback_buffer["y"]) >= CONFIG["feedback"]["retraining_interval"]:
        X_retrain = np.array(feedback_buffer["X"])
        y_retrain = np.array(feedback_buffer["y"])
        rf.fit(X_retrain, y_retrain)
        retraining_count += 1
        feedback_buffer = {"X": [], "y": []}
        print(f"  Retraining #{retraining_count} on {len(X_retrain)} samples")

print(f"Feedback retraining cycles: {retraining_count}")

# --- Evaluation After Feedback ---
print("\n--- Post-Feedback Evaluation ---")
y_pred_post = rf.predict(X_test)
y_proba_post = rf.predict_proba(X_test)[:, 1]

def compute_metrics(y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba) if y_proba is not None else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }

# --- Statistical Tests ---
print("\n--- Statistical Significance ---")
from scipy.stats import chi2_contingency, ttest_rel

# McNemar's test
def mcnemar_test(y_true, pred1, pred2):
    a = np.sum((pred1 == y_true) & (pred2 == y_true))
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    d = np.sum((pred1 != y_true) & (pred2 != y_true))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p = 1 - chi2_contingency([[a, b], [c, d]], correction=False)[1]
    return {"statistic": stat, "p_value": p, "significant": p < 0.05}

mcnemar = mcnemar_test(y_test.values, y_pred_rf, y_pred_post)
print(f"McNemar test:  p = {mcnemar['p_value']:.4f}  significant = {mcnemar['significant']}")

# Paired t-test on accuracy per sample
ttest = ttest_rel((y_pred_rf == y_test.values).astype(float),
                  (y_pred_post == y_test.values).astype(float))
print(f"Paired t-test: p = {ttest.pvalue:.4f}  significant = {ttest.pvalue < 0.05}")

# --- Final Summary ---
print("\n" + "=" * 60)
print("            HC-IDF PIPELINE FINAL RESULTS")
print("=" * 60)

baseline_metrics = compute_metrics(y_test, y_pred_rf, y_proba_rf)
post_metrics = compute_metrics(y_test, y_pred_post, y_proba_post)

results_summary = pd.DataFrame({
    "Baseline (RF)": baseline_metrics,
    "Post-Feedback": post_metrics,
}).T
print(results_summary.round(4))

print("\n--- Deep Learning Summary ---")
print(f"LSTM      F1: {f1_score(y_te_seq, y_pred_lstm):.4f}")
print(f"CNN-LSTM  F1: {f1_score(y_te_seq, y_pred_cnn):.4f}")
print(f"Autoencoder F1: {f1_score(y_test, y_pred_ae):.4f}")

print("\n--- Feature Importance (Top 10) ---")
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
for i, idx in enumerate(top_idx):
    print(f"  {i+1}. {X_train.columns[idx]}: {importances[idx]:.4f}")

# ============================================================
# PART 6 — Comprehensive Evaluation Plots (for thesis Chapter 4)
# ============================================================
# Run this as a separate cell after Part 5 completes.
# All plots are saved to PROCESSED/ for download.

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===== PART 6: Evaluation Plots =====")
plt.ioff()
sns.set_style("whitegrid")
PLOT_DIR = PROCESSED
CMAP = "Blues"

# ── 1. ROC Curves (all models) ───────────────────────────────
print("\n[PLOT] ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 6))

# RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {roc_auc_rf:.4f})", linewidth=2)

# LSTM (use last time-step prediction)
y_proba_lstm = lstm_model.predict(X_te_seq, verbose=0).ravel()
fpr_lstm, tpr_lstm, _ = roc_curve(y_te_seq, y_proba_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
ax.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC = {roc_auc_lstm:.4f})", linewidth=2)

# CNN-LSTM
y_proba_cnn = cnn_lstm.predict(X_te_seq, verbose=0).ravel()
fpr_cnn, tpr_cnn, _ = roc_curve(y_te_seq, y_proba_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
ax.plot(fpr_cnn, tpr_cnn, label=f"CNN-LSTM (AUC = {roc_auc_cnn:.4f})", linewidth=2)

# SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
ax.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Detection Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curves_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  roc_curves_comparison.png saved")

# ── 2. Precision-Recall Curves ───────────────────────────────
print("[PLOT] Precision-Recall Curves...")
fig, ax = plt.subplots(figsize=(8, 6))

for name, y_true, y_score in [
    ("RF", y_test, y_proba_rf),
    ("SVM", y_test, y_proba_svm),
    ("LSTM", y_te_seq, y_proba_lstm),
    ("CNN-LSTM", y_te_seq, y_proba_cnn),
]:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", linewidth=2)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
plt.tight_layout()
plt.savefig(PLOT_DIR / "pr_curves_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  pr_curves_comparison.png saved")

# ── 3. Confusion Matrices ────────────────────────────────────
print("[PLOT] Confusion Matrices...")
models_cm = [
    ("Random Forest", y_test, y_pred_rf),
    ("SVM", y_test, y_pred_svm),
    ("LSTM", y_te_seq, y_pred_lstm),
    ("CNN-LSTM", y_te_seq, y_pred_cnn),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (name, y_true, y_pred) in zip(axes.flat, models_cm):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(
        ax=ax, cmap=CMAP, colorbar=False, values_format="d")
    ax.set_title(name, fontsize=12, fontweight="bold")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()
print("  confusion_matrices.png saved")

# ── 4. Model Comparison Bar Chart ────────────────────────────
print("[PLOT] Model Comparison Bar Chart...")
metrics_df = pd.DataFrame({
    "Model": ["RF", "SVM", "LSTM", "CNN-LSTM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_te_seq, y_pred_lstm),
        accuracy_score(y_te_seq, y_pred_cnn),
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_svm),
        f1_score(y_te_seq, y_pred_lstm),
        f1_score(y_te_seq, y_pred_cnn),
    ],
    "AUC-ROC": [roc_auc_rf, roc_auc_svm, roc_auc_lstm, roc_auc_cnn],
})

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_df))
width = 0.25
colors = ["#1a5276", "#27ae60", "#e74c3c"]
for i, metric in enumerate(["Accuracy", "F1-Score", "AUC-ROC"]):
    bars = ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_df["Model"], fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison_barchart.png", dpi=200, bbox_inches="tight")
plt.close()
print("  model_comparison_barchart.png saved")

# ── 5. Feature Importance Bar Chart ──────────────────────────
print("[PLOT] Feature Importance...")
importances = rf.feature_importances_
top_n = 15
top_idx = np.argsort(importances)[::-1][:top_n]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(top_n), importances[top_idx][::-1], color="#1a5276", alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels([X_train.columns[i] for i in top_idx[::-1]], fontsize=10)
ax.set_xlabel("Importance", fontsize=12)
ax.set_title(f"Top {top_n} Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "feature_importance_top15.png", dpi=200, bbox_inches="tight")
plt.close()
print("  feature_importance_top15.png saved")

# ── 6. Training History (LSTM) ──────────────────────────────
print("[PLOT] LSTM Training History...")
if hasattr(lstm_model, "history"):
    history = lstm_model.history
    if hasattr(history, "history"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.plot(history.history["loss"], label="Train Loss", color="#1a5276")
        ax.plot(history.history["val_loss"], label="Val Loss", color="#e74c3c")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("LSTM Loss", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)

        ax = axes[1]
        ax.plot(history.history["accuracy"], label="Train Acc", color="#1a5276")
        ax.plot(history.history["val_accuracy"], label="Val Acc", color="#e74c3c")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title("LSTM Accuracy", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "lstm_training_history.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  lstm_training_history.png saved")

# ── 7. Detection Latency ─────────────────────────────────────
print("[PLOT] Detection Latency...")
import time
latency_models = {}
rf_start = time.time()
rf.predict(X_test[:1000])
latency_models["RF"] = (time.time() - rf_start) / 1000 * 1000  # ms per sample

svm_start = time.time()
svm.predict(X_test[:1000])
latency_models["SVM"] = (time.time() - svm_start) / 1000 * 1000

lstm_start = time.time()
lstm_model.predict(X_te_seq[:1000], verbose=0)
latency_models["LSTM"] = (time.time() - lstm_start) / 1000 * 1000

cnn_start = time.time()
cnn_lstm.predict(X_te_seq[:1000], verbose=0)
latency_models["CNN-LSTM"] = (time.time() - cnn_start) / 1000 * 1000

fig, ax = plt.subplots(figsize=(8, 5))
models_list = list(latency_models.keys())
latency_vals = [latency_models[m] for m in models_list]
bars = ax.bar(models_list, latency_vals, color=["#1a5276", "#27ae60", "#e74c3c", "#f39c12"], alpha=0.85)
for bar, val in zip(bars, latency_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.2f} ms", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Latency (ms per sample)", fontsize=12)
ax.set_title("Detection Latency Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "detection_latency.png", dpi=200, bbox_inches="tight")
plt.close()
print("  detection_latency.png saved")

print(f"\n[PLOT] All plots saved to {PLOT_DIR}/")
print("Download them for your thesis Chapter 4.")

# ── Final output summary ─────────────────────────────────────
print("\n" + "=" * 70)
print("HC-IDF EVALUATION SUMMARY — ALL METRICS & PLOTS")
print("=" * 70)
print(f"\nModels compared: RF | SVM | LSTM | CNN-LSTM | Autoencoder")
print(f"Datasets: CICIDS2017 + UNSW-NB15 ({df.shape[0]:,} rows, {df.shape[1]} features)")
print(f"Test size: {len(y_test):,} samples")
print(f"\n--- Performance ---")
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']:10s}  Acc: {row['Accuracy']:.4f}  F1: {row['F1-Score']:.4f}  AUC: {row['AUC-ROC']:.4f}")
print(f"\n--- Latency ---")
for m, l in latency_models.items():
    print(f"  {m:10s}  {l:.2f} ms/sample")
print(f"\n--- Statistical Tests ---")
print(f"  McNemar (RF vs Post-Feedback): p = {mcnemar['p_value']:.4f}")
print(f"  Paired t-test: p = {ttest.pvalue:.4f}")
print(f"\nPlots generated: {len(list(PLOT_DIR.glob('*.png')))} files")
print("\n[DONE] Full HC-IDF evaluation complete on Colab T4 GPU.")
# ============================================================
