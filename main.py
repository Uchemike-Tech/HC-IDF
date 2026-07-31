import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.preprocessor import DataPreprocessor
from src.detection.ml_engine import MLEngine
from src.detection.mitm_module import MITMDetectionModule
from src.xai.explainer import XAIExplainer
from src.feedback.feedback_loop import FeedbackLoop
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.statistical_tests import StatisticalTests
from src.detection.mitigation import MitigationOrchestrator


def load_data(config: dict) -> pd.DataFrame:
    parquet_path = Path("data/processed/combined_dataset.parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    print("[DATA] Parquet not found. Building from CSVs...")
    cic_path = Path(config["data"]["datasets"]["cicids2017"]["path"])
    frames = []
    for f in config["data"]["datasets"]["cicids2017"]["files"]:
        fp = cic_path / f
        if fp.exists():
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
            frames.append(df)
            print(f"  Loaded {f} ({len(df):,} rows)")

    unsw_path = Path(config["data"]["datasets"]["unsw_nb15"]["path"])
    for f in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
        fp = unsw_path / f
        if fp.exists():
            df = pd.read_csv(fp, low_memory=False)
            df.rename(columns={"attack_cat": "Label"}, inplace=True)
            df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
            frames.append(df)
            print(f"  Loaded {f} ({len(df):,} rows)")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [str(c).strip() for c in combined.columns]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False)
    print(f"[DATA] Saved {len(combined):,} rows to {parquet_path}")
    return combined


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # 1. Load and preprocess data
    df = load_data(config)
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # subsample for dev speed
    print(f"[DEV] Subsampled to {len(df):,} rows ({len(df.columns)} cols)")
    preprocessor = DataPreprocessor(config)
    df = preprocessor.clean(df)
    df = preprocessor.extract_flow_features(df)

    X = df.drop(columns=["Label", "Flow ID", "Timestamp"], errors="ignore")
    y = df["Label"].astype(int)

    X = preprocessor.normalize(X, fit=True)
    X, y = preprocessor.balance(X, y)
    X = preprocessor.select_features(X, y)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_split"],
        random_state=config["data"]["random_state"], stratify=y,
    )

    # 3. Train ML Engine (baseline)
    print("[BASELINE] Training ML Engine...")
    ml_engine = MLEngine(config)
    ml_engine.build_models()
    ml_engine.fit(X_train, y_train)

    y_pred_baseline = ml_engine.predict(X_test)
    y_proba_baseline = ml_engine.predict_proba(X_test)

    # 4. Evaluate baseline
    metrics = MetricsCalculator()
    baseline_results = metrics.compute_all(y_test, y_pred_baseline, y_proba_baseline)
    print(f"[BASELINE] Accuracy: {baseline_results['accuracy']:.4f}, "
          f"F1: {baseline_results['f1_score']:.4f}, "
          f"FPR: {baseline_results['false_positive_rate']:.4f}")

    # 5. XAI explanation on baseline
    print("[XAI] Generating explanations...")
    xai = XAIExplainer(config)
    xai.fit_shap(ml_engine.rf, X_train)
    shap_explanation = xai.explain_shap(X_test[:100])
    print(f"[XAI] Top features: {list(shap_explanation.get('top_features', {}).keys())[:3]}")

    # 6. MITM detection module evaluation
    print("[MITM] Evaluating MITM detection...")
    mitm_module = MITMDetectionModule(config)
    mitm_mask = (y_test == 1).values
    mitm_metrics = metrics.compute_mitm_specific(y_test.values, y_pred_baseline, mitm_mask)
    print(f"[MITM] F1: {mitm_metrics.get('mitm_f1', 0):.4f}")

    # 6b. Mitigation actions on detected MITM attacks
    print("[MITIGATION] Initialising response engine...")
    mitigator = MitigationOrchestrator(config)
    mitm_indices = mitm_mask.nonzero()[0][:50]
    for idx in mitm_indices:
        alert = {
            "mitm_alert": True,
            "mitm_type": ["ARP_SPOOFING"] if np.random.random() > 0.5 else ["SESSION_HIJACKING"],
            "src_ip": f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            "dst_ip": "10.0.0.1",
        }
        result = mitigator.handle_mitm_alert(alert)
        if result["action_count"] > 0:
            print(f"  [MITIGATION] {result['actions'][0]['action']} → {result['actions'][0]['target']} ({result['severity']})")
    mitigator_stats = mitigator.get_stats()
    print(f"[MITIGATION] {mitigator_stats['total_mitigations']} actions taken, {mitigator_stats['active_blocks']} IPs blocked")

    # 7. Simulate feedback loop
    print("[FEEDBACK] Simulating human-in-the-loop...")
    feedback = FeedbackLoop(config)
    X_retrain_pool, y_retrain_pool = X_test[:200], y_test[:200]
    for i in range(0, len(X_retrain_pool), 50):
        batch_X = X_retrain_pool.iloc[i:i+50]
        batch_y = y_retrain_pool.iloc[i:i+50]
        preds = ml_engine.predict(batch_X)
        for j in range(len(batch_X)):
            if preds[j] == batch_y.iloc[j]:
                feedback.record_decision(preds[j], batch_y.iloc[j], batch_X.iloc[j].values, "confirmed_tp")
            else:
                feedback.record_decision(preds[j], batch_y.iloc[j], batch_X.iloc[j].values, "dismissed_fp")
        if feedback.should_retrain():
            X_new, y_new = feedback.get_retraining_data()
            if X_new is not None:
                ml_engine.partial_retrain(X_new, y_new)
                print(f"[FEEDBACK] Retraining cycle {feedback.retraining_count} complete")

    # 8. Post-feedback evaluation
    y_pred_post = ml_engine.predict(X_test)
    post_results = metrics.compute_all(y_test, y_pred_post)
    print(f"[POST-FEEDBACK] Accuracy: {post_results['accuracy']:.4f}, "
          f"FPR: {post_results['false_positive_rate']:.4f}")

    # 9. Statistical testing
    stat_test = StatisticalTests()
    mcnemar = stat_test.mcnemar_test(y_test, y_pred_baseline, y_pred_post)
    print(f"[STATS] McNemar p-value: {mcnemar['p_value']:.4f}, "
          f"Significant: {mcnemar['significant']}")

    print("[DONE] HC-IDF pipeline complete.")


if __name__ == "__main__":
    main()
