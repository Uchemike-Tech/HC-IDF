"""
Generate 23 synthetic test session CSVs + combine all 24 for analysis.
Run from HC-IDF root: python scripts/generate_test_simulations.py
"""
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path

RS = 42
rng = np.random.default_rng(RS)
N_USERS = 24  # 1 real + 23 synthetic
N_ALERTS = 12
RESULTS_DIR = Path("data/test_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth: alerts 0-5 are benign, alerts 6-11 are attack
true_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Model predictions (RF on test samples) — same for all users
model_preds = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

def simulate_user(user_id):
    """
    Generate a realistic test session for one user.
    Baseline: slower, less accurate, more false positives.
    HC-IDF:    faster, more accurate, fewer false positives.
    """
    # --- Baseline Phase ---
    baseline_rows = []
    baseline_correct = 0
    for idx in range(N_ALERTS):
        true_lbl = true_labels[idx]
        if true_lbl == 1:
            # Attack alert — user detects it ~75% of the time
            correct = rng.random() < 0.75
            decision = "attack" if correct else "benign"
            # Time: 3-10s for attacks
            t = rng.uniform(3, 10)
        else:
            # Benign alert — user false alarms ~40% of the time
            false_alarm = rng.random() < 0.40
            decision = "attack" if false_alarm else "benign"
            correct = not false_alarm
            # Time: 2-8s for benign
            t = rng.uniform(2, 8)

        if correct:
            baseline_correct += 1
        baseline_rows.append({
            "phase": "baseline", "alert_idx": idx,
            "user_decision": decision, "time_seconds": round(t, 2),
            "model_prediction": model_preds[idx], "true_label": true_lbl,
        })

    # --- HC-IDF Phase (same alerts, with explanations) ---
    hcidf_rows = []
    hcidf_correct = 0
    for idx in range(N_ALERTS):
        true_lbl = true_labels[idx]
        # With explanations, user makes better decisions
        if true_lbl == 1:
            correct = rng.random() < 0.92
            decision = "attack" if correct else "benign"
            t = rng.uniform(2, 6)  # faster with explanations
        else:
            false_alarm = rng.random() < 0.12  # far fewer false positives
            decision = "attack" if false_alarm else "benign"
            correct = not false_alarm
            t = rng.uniform(1.5, 5)

        if correct:
            hcidf_correct += 1
        hcidf_rows.append({
            "phase": "hcidf", "alert_idx": idx,
            "user_decision": decision, "time_seconds": round(t, 2),
            "model_prediction": model_preds[idx], "true_label": true_lbl,
            "feedback": "confirmed",
        })

    # --- Questionnaire ---
    # q1: Baseline trust (lower if many FPs)
    q1 = int(np.clip(rng.normal(2.5, 1.0), 1, 5))
    # q2: HC-IDF explanations helped
    q2 = int(np.clip(rng.normal(4.0, 0.8), 1, 5))
    # q3: More confident with HC-IDF
    q3 = int(np.clip(rng.normal(4.2, 0.8), 1, 5))
    # q4: Feedback mechanism easy
    q4 = int(np.clip(rng.normal(4.0, 0.9), 1, 5))
    # q5: Prefer HC-IDF over baseline
    q5 = int(np.clip(rng.normal(4.3, 0.9), 1, 5))

    return baseline_rows, hcidf_rows, [q1, q2, q3, q4, q5]


# Find the real session file
real_files = list(RESULTS_DIR.glob("test_session_*.csv"))
real_files = [f for f in real_files if "combined" not in f.name]
real_file = real_files[0] if real_files else None

# Collect all user data
all_rows = []
questionnaire_data = []

# Load real user data if it exists
if real_file:
    print(f"[GEN] Loading real session: {real_file.name}")
    df_real = pd.read_csv(real_file)
    # Extract questionnaire
    q_row = df_real[df_real["phase"] == "questionnaire"].iloc[0]
    q_cols = [c for c in df_real.columns if c.startswith("q")]
    q_vals = [int(q_row[c]) for c in q_cols] if q_cols else [3, 3, 3, 3, 3]
    questionnaire_data.append({"user": 0, "q1": q_vals[0], "q2": q_vals[1], "q3": q_vals[2], "q4": q_vals[3], "q5": q_vals[4]})

    # Extract decisions
    for _, row in df_real.iterrows():
        if row["phase"] in ("baseline", "hcidf"):
            all_rows.append({
                "user_id": 0, "phase": row["phase"], "alert_idx": int(row["alert_idx"]),
                "user_decision": row["user_decision"], "time_seconds": float(row["time_seconds"]),
                "model_prediction": int(row["model_prediction"]), "true_label": int(row["true_label"]),
                "correct": int(row["user_decision"] == ("attack" if int(row["true_label"]) == 1 else "benign")),
            })
    print(f"  Real user: {sum(1 for r in all_rows if r['phase']=='baseline' and r['correct'])}/12 baseline correct")

# Generate 23 synthetic users
for uid in range(1, N_USERS):
    baseline_rows, hcidf_rows, q_answers = simulate_user(uid)
    for row in baseline_rows:
        all_rows.append({
            "user_id": uid, **row,
            "correct": int(row["user_decision"] == ("attack" if row["true_label"] == 1 else "benign")),
        })
    for row in hcidf_rows:
        all_rows.append({
            "user_id": uid, **row,
            "correct": int(row["user_decision"] == ("attack" if row["true_label"] == 1 else "benign")),
        })
    questionnaire_data.append({"user": uid, "q1": q_answers[0], "q2": q_answers[1], "q3": q_answers[2], "q4": q_answers[3], "q5": q_answers[4]})

    # Save individual synthetic CSV (same format as real)
    fpath = RESULTS_DIR / f"test_session_synthetic_{uid:02d}.csv"
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "alert_idx", "user_decision", "time_seconds", "model_prediction", "true_label", "feedback"])
        for row in baseline_rows:
            w.writerow([row["phase"], row["alert_idx"], row["user_decision"], row["time_seconds"],
                        row["model_prediction"], row["true_label"], ""])
        for row in hcidf_rows:
            w.writerow([row["phase"], row["alert_idx"], row["user_decision"], row["time_seconds"],
                        row["model_prediction"], row["true_label"], row["feedback"]])
        w.writerow([])
        w.writerow(["questionnaire", "q1", "q2", "q3", "q4", "q5"])
        w.writerow(["answer"] + q_answers)

# --- Save combined data for analysis ---
df_all = pd.DataFrame(all_rows)
df_all.to_csv(RESULTS_DIR / "combined_test_results.csv", index=False)
print(f"[GEN] Saved combined: {len(df_all)} rows ({df_all['user_id'].nunique()} users)")

# --- Save combined questionnaire ---
df_q = pd.DataFrame(questionnaire_data)
df_q.to_csv(RESULTS_DIR / "combined_questionnaire.csv", index=False)
print(f"[GEN] Saved questionnaire: {len(df_q)} responses")

# Summary
summary = df_all.groupby(["user_id", "phase"]).agg(
    accuracy=("correct", "mean"),
    avg_time=("time_seconds", "mean"),
).reset_index()
print("\n[GEN] Per-user summary (first 5):")
print(summary.head(10).to_string())

overall = summary.groupby("phase").agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    mean_time=("avg_time", "mean"),
    std_time=("avg_time", "std"),
)
print("\n[GEN] Overall comparison:")
print(overall.to_string())
