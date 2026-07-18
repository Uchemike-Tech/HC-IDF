"""
Prepares test session data for the HC-IDF user evaluation.
Run this once before launching the dashboard.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RS = 42
N_SAMPLES = 30000
N_TEST_ALERTS = 12  # 6 attack + 6 benign

PROCESSED = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("[PREP] Loading parquet...")
df = pd.read_parquet(PROCESSED / "combined_dataset.parquet")

print(f"[PREP] Sampling {N_SAMPLES:,} rows...")
df = df.sample(N_SAMPLES, random_state=RS).reset_index(drop=True)

# Basic cleaning
df = df.replace([np.inf, -np.inf], np.nan)
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(exclude=[np.number]).columns.drop("Label", errors="ignore")
for c in cat_cols:
    df[c] = df[c].fillna("missing").astype("category").cat.codes

# Select usable features (drop identifiers)
drop_cols = ["Label", "label", "attack_cat", "id", "Flow ID", "Timestamp", "Packet Length", "Fwd Header Length.1"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df["Label"].astype(int)

# Normalize
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X.astype(np.float64)), columns=feature_cols)

# Train RF
print("[PREP] Training RF...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, random_state=RS, n_jobs=-1, verbose=0,
)
rf.fit(X, y)
acc = rf.score(X, y)
print(f"[PREP] RF accuracy: {acc:.4f}")

# Pick diverse test samples (mix of attack and benign)
attack_idx = np.where(y == 1)[0]
benign_idx = np.where(y == 0)[0]

np.random.seed(RS)
selected = []
selected.extend(np.random.choice(attack_idx, N_TEST_ALERTS // 2, replace=False))
selected.extend(np.random.choice(benign_idx, N_TEST_ALERTS // 2, replace=False))
np.random.shuffle(selected)

test_samples = X.iloc[selected].copy()
test_labels = y.iloc[selected].copy()
test_preds = rf.predict(test_samples)
test_proba = rf.predict_proba(test_samples)[:, 1]

# Compute feature importance for each test sample (as SHAP proxy)
importances = rf.feature_importances_
top_feat_idx = np.argsort(importances)[::-1][:10]
top_features = [feature_cols[i] for i in top_feat_idx]

sample_explanations = []
for i, idx in enumerate(selected):
    contrib = []
    vals = X.iloc[idx].values
    for fi in top_feat_idx:
        contrib.append({
            "feature": feature_cols[fi],
            "value": float(vals[fi]),
            "importance": float(importances[fi]),
        })
    sample_explanations.append({
        "sample_id": int(idx),
        "true_label": int(test_labels.iloc[i]),
        "predicted": int(test_preds[i]),
        "probability": float(test_proba[i]),
        "features": contrib,
    })

# Save
print("[PREP] Saving artifacts...")
joblib.dump(rf, MODEL_DIR / "rf_test_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

test_data = {
    "samples": test_samples.values.tolist(),
    "feature_names": feature_cols,
    "labels": test_labels.tolist(),
    "predictions": test_preds.tolist(),
    "probabilities": test_proba.tolist(),
    "explanations": sample_explanations,
    "attack_idx": attack_idx.tolist()[:100],  # sample
    "benign_idx": benign_idx.tolist()[:100],
}
joblib.dump(test_data, MODEL_DIR / "test_data.pkl")

print(f"[PREP] Done. {N_TEST_ALERTS} test samples ready.")
print(f"[PREP] Top features: {top_features[:5]}")
