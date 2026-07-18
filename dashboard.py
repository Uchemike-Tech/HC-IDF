import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="HC-IDF Dashboard", layout="wide", initial_sidebar_state="expanded")

# ─── Color palette ───────────────────────────────────────────
PRIMARY = "#1a5276"
SECONDARY = "#2c3e50"
BG_LIGHT = "#f8f9fa"
GREEN = "#27ae60"
RED = "#e74c3c"
GRAY = "#95a5a6"

# ─── Results from Colab ──────────────────────────────────────
RESULTS = {
    "dataset": {"total_rows": 3088416, "features": 123, "benign": 2366097, "attack": 722319},
    "ml": {
        "RF": {"accuracy": 0.9989, "precision": 0.9989, "recall": 0.9988, "f1": 0.9989, "auc": 0.9999},
        "SVM": {"accuracy": 0.9544, "precision": None, "recall": None, "f1": 0.9553, "auc": 0.9897},
    },
    "dl": {
        "LSTM": {"accuracy": 0.9847, "f1": 0.9848},
        "CNN-LSTM": {"accuracy": 0.9658, "f1": 0.9664},
        "Autoencoder": {"f1": 0.1196},
    },
    "mitm": {"precision": 1.0, "recall": 0.9988, "f1": 0.9994, "arp_alerts": 134},
    "feedback": {
        "cycles": 5,
        "post_feedback": {"accuracy": 0.9287, "precision": 0.9148, "recall": 0.9453, "f1": 0.9298, "auc": 0.9843},
    },
    "feature_importance": [
        ("Init_Win_bytes_backward", 0.0431), ("Destination Port", 0.0422),
        ("Init_Win_bytes_forward", 0.0406), ("Average Packet Size", 0.0347),
        ("Fwd Header Length.1", 0.0342), ("Min Packet Length", 0.0307),
        ("Flow Duration", 0.0271), ("Fwd Header Length", 0.0258),
        ("Flow IAT Mean", 0.0241), ("Fwd IAT Total", 0.0238),
    ],
    "xai": {
        "shap_plot": "notebooks/shap_summary.png",
        "lime_file": "notebooks/lime_explanation.html",
        "lime_features": ["Destination Port <= -0.43", "Min Packet Length <= -0.64", "proto <= -0.28", "Bwd Packet Length Min <= -0.57", "Packet Length Mean <= -0.57"],
    },
}

# ─── Helper ──────────────────────────────────────────────────
def metric_card(label, value, suffix="", color=PRIMARY):
    sign = "+" if value > 0 else ""
    return f"""
    <div style="background:{BG_LIGHT}; padding:16px 20px; border-radius:8px; border-left:4px solid {color}; margin-bottom:8px">
        <div style="font-size:13px; color:{GRAY}; margin-bottom:4px">{label}</div>
        <div style="font-size:26px; font-weight:600; color:{color}">{sign}{value:.4f}{suffix}</div>
    </div>
    """

def divider():
    st.markdown("<hr style='margin: 24px 0; border: none; border-top: 1px solid #e9ecef;'>", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 16px 0;">
        <div style="font-size:22px; font-weight:700; color:white;">HC-IDF</div>
        <div style="font-size:12px; color:{GRAY};">Human-Centric IDS Framework</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<hr style='border-color:{SECONDARY}; margin:8px 0'>", unsafe_allow_html=True)

    page = st.radio("", ["Overview", "Test Session", "Detection Models", "Explanations", "MITM & Feedback", "Live Capture", "Grafana Dashboard"], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{SECONDARY}; margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px; color:{GRAY};'>MSc CS Research<br>Smart City Security</div>", unsafe_allow_html=True)

# ─── Page: Overview ──────────────────────────────────────────
if page == "Overview":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>HC-IDF Overview</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:24px'>Human-Centric Intrusion Detection Framework — Evaluation Summary</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    ds = RESULTS["dataset"]
    col1.markdown(metric_card("Total Samples", ds["total_rows"], "", PRIMARY), unsafe_allow_html=True)
    col2.markdown(metric_card("Features", ds["features"], "", PRIMARY), unsafe_allow_html=True)
    col3.markdown(metric_card("Benign", ds["benign"], "", GREEN), unsafe_allow_html=True)
    col4.markdown(metric_card("Attack", ds["attack"], "", RED), unsafe_allow_html=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Best Performing Model</div>", unsafe_allow_html=True)

    r = RESULTS["ml"]["RF"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(metric_card("Accuracy", r["accuracy"], "", GREEN), unsafe_allow_html=True)
    col2.markdown(metric_card("Precision", r["precision"], "", GREEN), unsafe_allow_html=True)
    col3.markdown(metric_card("Recall", r["recall"], "", GREEN), unsafe_allow_html=True)
    col4.markdown(metric_card("F1-Score", r["f1"], "", GREEN), unsafe_allow_html=True)
    col5.markdown(metric_card("AUC-ROC", r["auc"], "", GREEN), unsafe_allow_html=True)

    st.markdown("<div style='font-size:13px; color:#6c757d; margin-top:4px'>Random Forest — trained on 743,710 samples, tested on 131,244</div>", unsafe_allow_html=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Key Findings</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <ul style='color:#2c3e50; font-size:14px; line-height:1.8'>
        <li><b>Random Forest</b> achieves the highest overall performance (F1 = 0.9989)</li>
        <li><b>LSTM</b> is the best deep learning model (F1 = 0.9848), comparable to RF</li>
        <li><b>MITM detection</b> is highly effective (F1 = 0.9994) with dedicated module</li>
        <li><b>Autoencoder</b> performs poorly for anomaly detection (F1 = 0.1196)</li>
        <li>Feedback loop needs larger retraining data to be effective</li>
    </ul>
    """, unsafe_allow_html=True)

# ─── Test Session ────────────────────────────────────────────
elif page == "Test Session":
    import time, json, joblib
    from csv import writer as csv_writer

    MODEL_DIR = Path("models")
    test_ready = all((MODEL_DIR / f).exists() for f in ["test_data.pkl", "rf_test_model.pkl", "feature_cols.pkl"])

    if not test_ready:
        st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Test Session</div>", unsafe_allow_html=True)
        st.warning("Test data not prepared. Run `python scripts/prepare_test_data.py` first.")
        st.stop()

    # Load test data
    test_data = joblib.load(MODEL_DIR / "test_data.pkl")
    rf = joblib.load(MODEL_DIR / "rf_test_model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

    samples = test_data["samples"]
    labels = test_data["labels"]
    preds = test_data["predictions"]
    probabilities = test_data["probabilities"]
    explanations = test_data["explanations"]
    top_features = [explanations[0]["features"][i]["feature"] for i in range(5)]
    n_alerts = len(samples)

    # Initialize session state
    phase_defaults = {
        "test_phase": "welcome",
        "baseline_idx": 0, "hcidf_idx": 0,
        "baseline_answers": [], "hcidf_answers": [],
        "questionnaire": {}, "alert_start": None, "completed": False,
    }
    for k, v in phase_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── helpers ──
    def fmt_val(v):
        if abs(v) < 0.01:
            return f"{v:.6f}"
        return f"{v:.4f}"

    def alert_card(idx, mode="baseline"):
        is_attack = labels[idx] == 1
        is_pred_attack = preds[idx] == 1
        prob = probabilities[idx]
        verdict = "⚠️ ATTACK" if is_pred_attack else "✅ BENIGN"
        verdict_color = RED if is_pred_attack else GREEN

        st.markdown(f"""
        <div style='background:white; border:1px solid #dee2e6; border-radius:8px; padding:20px; margin-bottom:16px'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px'>
                <div style='font-size:16px; font-weight:600; color:{SECONDARY}'>Alert #{idx+1} of {n_alerts}</div>
                <div style='font-size:18px; font-weight:700; color:{verdict_color};'>{verdict}</div>
            </div>
            <div style='font-size:13px; color:{GRAY}; margin-bottom:12px'>Confidence: {prob:.2%}</div>
            <div style='background:{BG_LIGHT}; border-radius:6px; padding:12px; font-size:12px; color:{SECONDARY};'>
                <table style='width:100%; border-collapse:collapse;'>
                    <tr><td width='50%'><b>Dest Port:</b> {samples[idx][feature_cols.index("Destination Port")] if "Destination Port" in feature_cols else "—":.2f}</td>
                        <td width='50%'><b>Flow Duration:</b> {samples[idx][feature_cols.index("Flow Duration")] if "Flow Duration" in feature_cols else "—":.2f}</td></tr>
                    <tr><td><b>Fwd Pkts:</b> {samples[idx][feature_cols.index("Total Fwd Packets")] if "Total Fwd Packets" in feature_cols else "—":.2f}</td>
                        <td><b>Bwd Pkts:</b> {samples[idx][feature_cols.index("Total Backward Packets")] if "Total Backward Packets" in feature_cols else "—":.2f}</td></tr>
                    <tr><td><b>Pkt Length Mean:</b> {samples[idx][feature_cols.index("Packet Length Mean")] if "Packet Length Mean" in feature_cols else "—":.2f}</td>
                        <td><b>Flow IAT Mean:</b> {samples[idx][feature_cols.index("Flow IAT Mean")] if "Flow IAT Mean" in feature_cols else "—":.2f}</td></tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if mode == "hcidf":
            st.markdown(f"<div style='font-size:15px; font-weight:600; color:{PRIMARY}; margin:8px 0 8px'>Why this alert? (XAI Explanation)</div>", unsafe_allow_html=True)
            exp = explanations[idx]
            for feat in exp["features"][:5]:
                bar_w = max(5, abs(feat["importance"]) * 200)
                bar_color = GREEN if feat["importance"] > 0 else RED
                st.markdown(f"""
                <div style='margin-bottom:6px; font-size:12px;'>
                    <div style='display:flex; justify-content:space-between; color:{SECONDARY};'>
                        <span><b>{feat['feature']}</b></span>
                        <span>{fmt_val(feat['value'])}</span>
                    </div>
                    <div style='background:#e9ecef; border-radius:4px; height:8px; margin-top:2px;'>
                        <div style='background:{bar_color}; border-radius:4px; height:8px; width:{bar_w}px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def save_results():
        results_dir = Path("data/test_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fpath = results_dir / f"test_session_{timestamp}.csv"
        with open(fpath, "w", newline="") as f:
            w = csv_writer(f)
            w.writerow(["phase", "alert_idx", "user_decision", "time_seconds", "model_prediction", "true_label", "feedback"])
            for ans in st.session_state["baseline_answers"]:
                w.writerow(["baseline", ans[0], ans[1], ans[2], preds[ans[0]], labels[ans[0]], ""])
            for ans in st.session_state["hcidf_answers"]:
                w.writerow(["hcidf", ans[0], ans[1], ans[2], preds[ans[0]], labels[ans[0]], ans[3]])
            w.writerow([])
            w.writerow(["questionnaire"] + list(st.session_state["questionnaire"].keys()))
            w.writerow(["answer"] + list(st.session_state["questionnaire"].values()))
        return fpath

    def _next(idx, phase, next_phase):
        if idx + 1 < n_alerts:
            st.session_state[f"{phase}_idx"] = idx + 1
        else:
            st.session_state["test_phase"] = next_phase
            if next_phase != "questionnaire":
                st.session_state[f"{next_phase}_idx"] = 0
        st.session_state["alert_start"] = time.time()
        st.rerun()

    # ── Welcome ──
    if st.session_state["test_phase"] == "welcome":
        st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Security Analyst Test Session</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:white; border:1px solid #dee2e6; border-radius:8px; padding:24px; margin-bottom:16px; line-height:1.8'>
            <p style='font-size:15px; color:{SECONDARY};'>This test compares a <b>baseline IDS</b> with the <b>HC-IDF</b> (with XAI explanations + feedback).</p>
            <p style='font-size:14px; color:{SECONDARY};'><b>Format:</b></p>
            <ul style='font-size:14px; color:{SECONDARY};'>
                <li><b>Phase 1 — Baseline:</b> Review {n_alerts} alerts with basic info only. Decide: Attack or Benign?</li>
                <li><b>Phase 2 — HC-IDF:</b> Same {n_alerts} alerts with explanations. Decide + confirm/dismiss.</li>
                <li><b>Questionnaire:</b> Quick 5-question survey comparing both.</li>
            </ul>
            <p style='font-size:13px; color:{GRAY};'>Your responses are saved anonymously for analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Test", type="primary", use_container_width=True):
            st.session_state["test_phase"] = "baseline"
            st.session_state["baseline_idx"] = 0
            st.session_state["alert_start"] = time.time()
            st.rerun()

    # ── Phase 1: Baseline ──
    elif st.session_state["test_phase"] == "baseline":
        idx = st.session_state["baseline_idx"]
        st.markdown(f"<div style='font-size:20px; font-weight:600; color:{PRIMARY}; margin-bottom:4px'>Phase 1: Baseline IDS</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:{GRAY}; margin-bottom:16px'>Alert {idx+1} of {n_alerts} — no explanations provided</div>", unsafe_allow_html=True)

        alert_card(idx, mode="baseline")

        col1, col2 = st.columns(2)
        if col1.button(f"✅ Benign", use_container_width=True, key=f"b_benign_{idx}"):
            elapsed = time.time() - st.session_state["alert_start"]
            st.session_state["baseline_answers"].append((idx, "benign", round(elapsed, 2)))
            _next(idx, "baseline", "hcidf")
        if col2.button(f"⚠️ Attack", use_container_width=True, key=f"b_attack_{idx}", type="primary"):
            elapsed = time.time() - st.session_state["alert_start"]
            st.session_state["baseline_answers"].append((idx, "attack", round(elapsed, 2)))
            _next(idx, "baseline", "hcidf")

    # ── Phase 2: HC-IDF ──
    elif st.session_state["test_phase"] == "hcidf":
        idx = st.session_state["hcidf_idx"]
        st.markdown(f"<div style='font-size:20px; font-weight:600; color:{PRIMARY}; margin-bottom:4px'>Phase 2: HC-IDF (with XAI)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:{GRAY}; margin-bottom:16px'>Alert {idx+1} of {n_alerts} — explanations shown below</div>", unsafe_allow_html=True)

        alert_card(idx, mode="hcidf")

        st.markdown(f"<div style='font-size:14px; font-weight:600; color:{SECONDARY}; margin:12px 0 8px'>Your Decision</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        if col1.button(f"✅ Benign", use_container_width=True, key=f"h_benign_{idx}"):
            elapsed = time.time() - st.session_state["alert_start"]
            st.session_state["hcidf_answers"].append((idx, "benign", round(elapsed, 2), "confirmed"))
            _next(idx, "hcidf", "questionnaire")
        if col2.button(f"⚠️ Attack", use_container_width=True, key=f"h_attack_{idx}", type="primary"):
            elapsed = time.time() - st.session_state["alert_start"]
            st.session_state["hcidf_answers"].append((idx, "attack", round(elapsed, 2), "confirmed"))
            _next(idx, "hcidf", "questionnaire")

    # ── Questionnaire ──
    elif st.session_state["test_phase"] == "questionnaire":
        st.markdown(f"<div style='font-size:24px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Post-Test Questionnaire</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:{GRAY}; margin-bottom:20px'>Rate your experience on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree)</div>", unsafe_allow_html=True)

        qs = {
            "q1": "Baseline: I could trust the system's alerts without explanations.",
            "q2": "HC-IDF: The explanations helped me understand why alerts were triggered.",
            "q3": "HC-IDF: I felt more confident making decisions with explanations provided.",
            "q4": "The feedback mechanism (confirm/dismiss) was easy to use.",
            "q5": "I would prefer using HC-IDF over a standard IDS for daily analysis.",
        }
        answers = {}
        for key, q_text in qs.items():
            answers[key] = st.radio(q_text, ["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"],
                                    index=2, horizontal=True, key=key,
                                    label_visibility="visible")

        if st.button("Submit & View Results", type="primary", use_container_width=True):
            st.session_state["questionnaire"] = {k: v[0] for k, v in answers.items()}
            st.session_state["test_phase"] = "done"
            st.session_state["completed"] = True
            fpath = save_results()
            st.session_state["results_file"] = str(fpath)
            st.rerun()

    # ── Done ──
    elif st.session_state["test_phase"] == "done":
        st.markdown(f"<div style='font-size:24px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Test Complete</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:20px'>Results saved to <code>{st.session_state.get('results_file', '')}</code></div>", unsafe_allow_html=True)

        # Compute baseline stats
        b_times = [a[2] for a in st.session_state["baseline_answers"]]
        h_times = [a[2] for a in st.session_state["hcidf_answers"]]
        b_correct = sum(1 for a in st.session_state["baseline_answers"] if (a[1] == "attack") == (labels[a[0]] == 1))
        h_correct = sum(1 for a in st.session_state["hcidf_answers"] if (a[1] == "attack") == (labels[a[0]] == 1))

        st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Your Results</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='background:{BG_LIGHT}; border-radius:8px; padding:16px;'>"
                        f"<div style='font-size:14px; font-weight:600; color:{SECONDARY}; margin-bottom:8px'>Baseline IDS</div>"
                        f"<div style='font-size:13px; color:{SECONDARY};'>Avg decision time: <b>{np.mean(b_times):.1f}s</b></div>"
                        f"<div style='font-size:13px; color:{SECONDARY};'>Correct: <b>{b_correct}/{n_alerts}</b></div>"
                        f"</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='background:{BG_LIGHT}; border-radius:8px; padding:16px;'>"
                        f"<div style='font-size:14px; font-weight:600; color:{PRIMARY}; margin-bottom:8px'>HC-IDF</div>"
                        f"<div style='font-size:13px; color:{SECONDARY};'>Avg decision time: <b>{np.mean(h_times):.1f}s</b></div>"
                        f"<div style='font-size:13px; color:{SECONDARY};'>Correct: <b>{h_correct}/{n_alerts}</b></div>"
                        f"</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top:20px; font-size:14px; color:{GRAY};'>"
                    f"Results saved to: <code>{st.session_state.get('results_file', '')}</code>. "
                    f"Collect all participant CSVs into <code>data/test_results/</code> for inferential analysis.</div>", unsafe_allow_html=True)

        if st.button("Reset & Take Again", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k in phase_defaults:
                    del st.session_state[k]
            st.rerun()
elif page == "Detection Models":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Detection Models Comparison</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:24px'>Supervised ML and Deep Learning performance on CICIDS2017 + UNSW-NB15</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Traditional Machine Learning</div>", unsafe_allow_html=True)
    ml_data = []
    for name, m in RESULTS["ml"].items():
        ml_data.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m.get("precision", 0) or 0,
                        "Recall": m.get("recall", 0) or 0, "F1-Score": m["f1"], "AUC-ROC": m["auc"]})
    st.dataframe(pd.DataFrame(ml_data).set_index("Model").style.format("{:.4f}"), use_container_width=True)

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin:24px 0 12px'>Deep Learning</div>", unsafe_allow_html=True)
    dl_data = []
    for name, m in RESULTS["dl"].items():
        dl_data.append({"Model": name, "Accuracy": m.get("accuracy", 0), "F1-Score": m["f1"]})
    st.dataframe(pd.DataFrame(dl_data).set_index("Model").style.format("{:.4f}"), use_container_width=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Dataset Composition</div>", unsafe_allow_html=True)
    ds = RESULTS["dataset"]
    dist = pd.DataFrame({
        "Class": ["Benign", "Attack"],
        "Count": [ds["benign"], ds["attack"]],
    })
    st.dataframe(dist.style.format({"Count": "{:,}"}), use_container_width=True)

# ─── Page: Explanations ──────────────────────────────────────
elif page == "Explanations":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Model Explanations & Interpretability</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:24px'>XAI outputs — SHAP summary plot and LIME local explanations</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:8px'>Feature Importance (Global — Random Forest)</div>", unsafe_allow_html=True)
    fi_data = pd.DataFrame(RESULTS["feature_importance"], columns=["Feature", "Importance"])
    st.dataframe(fi_data.style.format({"Importance": "{:.4f}"}).bar(subset=["Importance"], color="#1a5276"), use_container_width=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:8px'>SHAP Summary Plot</div>", unsafe_allow_html=True)
    shap_path = Path(RESULTS["xai"]["shap_plot"])
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("SHAP plot not found locally. Run Colab Part 4 to generate it, or check the path.")

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:8px'>LIME Local Explanation</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:8px'>Top contributing features for a single prediction:</div>", unsafe_allow_html=True)
    for f in RESULTS["xai"]["lime_features"]:
        st.markdown(f"<div style='padding:6px 12px; background:{BG_LIGHT}; border-radius:4px; margin-bottom:4px; font-size:13px; color:{SECONDARY}'>• {f}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:16px; font-size:13px; color:{GRAY};'>Full LIME explanation saved as HTML — open <code>data/processed/lime_explanation.html</code> in a browser.</div>", unsafe_allow_html=True)

# ─── Page: MITM & Feedback ──────────────────────────────────
elif page == "MITM & Feedback":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>MITM Detection & Human-in-the-Loop</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:24px'>Dedicated MITM module performance and feedback loop simulation</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>MITM Attack Detection</div>", unsafe_allow_html=True)
    mitm = RESULTS["mitm"]
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("Precision", mitm["precision"], "", GREEN), unsafe_allow_html=True)
    col2.markdown(metric_card("Recall", mitm["recall"], "", GREEN), unsafe_allow_html=True)
    col3.markdown(metric_card("F1-Score", mitm["f1"], "", GREEN), unsafe_allow_html=True)
    col4.markdown(metric_card("ARP Alerts", mitm["arp_alerts"], "", RED), unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:13px; color:{GRAY}; margin-top:4px'>
        MITM detection uses dedicated ARP spoofing analysis + session hijacking heuristics on top of the RF classifier.
        Attack traffic is isolated and evaluated separately.
    </div>
    """, unsafe_allow_html=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Human-in-the-Loop Feedback</div>", unsafe_allow_html=True)
    fb = RESULTS["feedback"]
    col1, col2 = st.columns(2)
    col1.markdown(metric_card("Retraining Cycles", fb["cycles"], "", PRIMARY), unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:14px; font-weight:500; color:{SECONDARY}; margin:16px 0 8px'>Before vs After Feedback</div>", unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Metric": ["Accuracy", "F1-Score", "FPR", "FNR"],
        "Baseline (RF)": [0.9989, 0.9989, 0.0011, 0.0012],
        "Post-Feedback": [fb["post_feedback"]["accuracy"], fb["post_feedback"]["f1"], 0.0880, 0.0547],
    })
    st.dataframe(comp.set_index("Metric").style.format("{:.4f}"), use_container_width=True)

    st.markdown(f"""
    <div style='margin-top:12px; padding:12px 16px; background:#fff3cd; border-radius:6px; font-size:13px; color:#856404;'>
        <b>Note:</b> Post-feedback metrics dropped because retraining used only 500 simulated samples (100 per cycle).
        In production, the feedback buffer should accumulate hundreds or thousands of corrected labels before retraining.
    </div>
    """, unsafe_allow_html=True)

    divider()

    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{PRIMARY}; margin-bottom:12px'>Statistical Significance</div>", unsafe_allow_html=True)

# ─── Page: Live Capture ─────────────────────────────────────
elif page == "Live Capture":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Live Packet Capture</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:24px'>Real-time network traffic monitoring via Scapy — Npcap backend</div>", unsafe_allow_html=True)

    from src.network.capture import LiveCapture, SCAPY_AVAIL

    if not SCAPY_AVAIL:
        st.error("Scapy not installed. Run: pip install scapy")
        st.stop()

    if "capture" not in st.session_state:
        st.session_state["capture"] = None
        st.session_state["capture_running"] = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if not st.session_state["capture_running"]:
            if st.button("▶ Start Capture", type="primary", use_container_width=True):
                cap = LiveCapture()
                if cap.start(timeout=30):
                    st.session_state["capture"] = cap
                    st.session_state["capture_running"] = True
                    st.rerun()
        else:
            if st.button("⏹ Stop Capture", type="primary", use_container_width=True):
                st.session_state["capture"].stop()
                st.session_state["capture_running"] = False
                st.rerun()

    with col2:
        st.markdown(f"<div style='font-size:13px; color:{GRAY}; margin-top:8px'>Status: {'<span style=color:green;font-weight:600>RUNNING</span>' if st.session_state.get('capture_running') else '<span style=color:gray>STOPPED</span>'}</div>", unsafe_allow_html=True)

    cap = st.session_state.get("capture")
    if cap and st.session_state["capture_running"]:
        import time as _time
        summary = cap.summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Packets", summary["total"])
        col2.metric("Rate (pps)", summary["rate"])
        col3.metric("Elapsed (s)", summary["elapsed"])
        col4.metric("ARP Packets", summary["arp"])

        st.markdown(f"<div style='font-size:15px; font-weight:600; color:{PRIMARY}; margin:16px 0 8px'>Protocol Breakdown</div>", unsafe_allow_html=True)
        proto_df = pd.DataFrame({
            "Protocol": ["TCP", "UDP", "ARP", "Other"],
            "Count": [summary["tcp"], summary["udp"], summary["arp"], max(0, summary["total"] - summary["tcp"] - summary["udp"] - summary["arp"])],
        })
        st.dataframe(proto_df, use_container_width=True, hide_index=True)

        if summary["top_src_ips"]:
            st.markdown(f"<div style='font-size:15px; font-weight:600; color:{PRIMARY}; margin:16px 0 8px'>Top Source IPs</div>", unsafe_allow_html=True)
            ip_df = pd.DataFrame(summary["top_src_ips"], columns=["IP", "Packets"])
            st.dataframe(ip_df, use_container_width=True, hide_index=True)

        if summary["top_ports"]:
            st.markdown(f"<div style='font-size:15px; font-weight:600; color:{PRIMARY}; margin:16px 0 8px'>Top Destination Ports</div>", unsafe_allow_html=True)
            port_df = pd.DataFrame(summary["top_ports"], columns=["Port", "Count"])
            st.dataframe(port_df, use_container_width=True, hide_index=True)

        _time.sleep(2)
        st.rerun()
    else:
        st.info("Click 'Start Capture' to begin monitoring network traffic. Requires Npcap installed.")

# ─── Page: Grafana Dashboard ────────────────────────────────
elif page == "Grafana Dashboard":
    st.markdown(f"<div style='font-size:28px; font-weight:700; color:{PRIMARY}; margin-bottom:4px'>Grafana Live Monitoring</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:{GRAY}; margin-bottom:16px'>Real-time HC-IDF metrics dashboard powered by Grafana</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style='background:{BG_LIGHT}; padding:16px; border-radius:8px; font-size:13px; color:{SECONDARY};'>
            <b>Setup Instructions:</b><br><br>
            1. Open a terminal in <code>HC-IDF/grafana/</code><br>
            2. Run: <code>docker compose up -d</code><br>
            3. Run: <code>python metrics_server.py</code><br><br>
            <b>Credentials:</b><br>
            URL: <code>http://localhost:3000</code><br>
            User: <code>admin</code><br>
            Pass: <code>admin</code><br><br>
            Or use the start script:<br>
            <code>scripts/start_all.ps1</code>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:{BG_LIGHT}; padding:16px; border-radius:8px; font-size:13px; color:{SECONDARY};'>
            <b>Dashboard Panels:</b><br>
            • <b>Detection Rate</b> — live model accuracy gauge<br>
            • <b>Active & MITM Alerts</b> — alert counters<br>
            • <b>Model Accuracy Over Time</b> — RF, LSTM, CNN-LSTM trends<br>
            • <b>Protocol Distribution</b> — TCP/UDP/ARP pie chart<br>
            • <b>Top Source IPs</b> — most active IP addresses<br>
            • <b>Recent Alerts</b> — live alert feed table
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:20px; font-size:14px; font-weight:600; color:{PRIMARY};'>Embedded Grafana Dashboard</div>", unsafe_allow_html=True)

    grafana_url = "http://localhost:3000/d/hcidf-overview/hc-idf-live-monitoring?orgId=1&from=now-5m&to=now&refresh=5s&kiosk"
    st.markdown(f"""
    <div style='position:relative; width:100%; height:600px; border:1px solid #dee2e6; border-radius:8px; overflow:hidden;'>
        <iframe src="{grafana_url}" width="100%" height="600px" frameborder="0" style="border:none;"></iframe>
    </div>
    <div style='margin-top:8px; font-size:12px; color:{GRAY};'>
        Grafana must be running on port 3000 for the dashboard to display. 
        <a href="{grafana_url}" target="_blank">Open in new tab</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <table style='width:100%; font-size:14px; border-collapse:collapse;'>
        <tr style='background:{BG_LIGHT};'><td style='padding:8px 12px;'><b>McNemar Test</b></td><td style='padding:8px 12px;'>p = 1.0000</td><td style='padding:8px 12px; color:{GRAY};'>Not significant</td></tr>
        <tr><td style='padding:8px 12px;'><b>Paired t-Test</b></td><td style='padding:8px 12px;'>p = 0.0000</td><td style='padding:8px 12px; color:{RED};'>Significant</td></tr>
    </table>
    """, unsafe_allow_html=True)
