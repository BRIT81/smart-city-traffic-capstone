"""
mlflow_tracking.py

Task 4 (Part 3): Advanced AI Technique — MLflow for Experiment Tracking.

Why this technique: recommended explicitly by the brief since it connects
directly to Task 6's MLOps component. Of the four options, it is also the
one that adds value to work already completed rather than requiring an
unrelated new model (a GAN or a self-supervised anomaly detector would
each be a standalone undertaking; model quantisation has little to solve
here since every model in this project is already small).

How it was implemented: every model trained across Tasks 1-3 is wrapped
in an MLflow run. Each model is reloaded from disk, its exact test set is
rebuilt using the same functions used to train it (imported directly from
supervised_models.py, unsupervised_models.py, and
deep_learning_explainability.py rather than duplicated), and its metrics
are recomputed fresh rather than hardcoded, so what's logged always
matches the actual saved model. Hyperparameters come from each
scikit-learn model's own get_params(); the LSTM's architecture is logged
explicitly since Keras has no equivalent method. Tracking uses a local
SQLite backend (MLflow's plain filesystem store is deprecated as of this
project's MLflow version) with an explicit artifact_location, so both
metadata and artifacts live under part3_machine_learning/mlflow/ rather
than scattered wherever a script happens to be run from.

What value it adds: a single place to compare every model built in this
capstone, side by side, with full hyperparameters, metrics, and the
model files themselves as versioned artifacts. This is also the direct
foundation for Task 6's MLOps and deployment simulation.

Limitations: this is retrospective logging of already-completed training
runs, not live instrumentation during training; the local SQLite/file
backend has no multi-user access or remote artifact storage; and runs
are still triggered manually by running this script, there is no
automated retraining pipeline behind it.
"""

import logging
from pathlib import Path

import joblib
import mlflow
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, r2_score, silhouette_score,
)

from data_prep import prepare_data
from supervised_models import FEATURE_COLS, chronological_split, build_feature_targets
from unsupervised_models import (
    add_weather_severity, CLUSTER_FEATURES, discretize_for_arm, mine_congestion_rules,
    MIN_SUPPORT, WEATHER_SEVERITY_RANK,
)
from deep_learning_explainability import (
    build_gap_aware_windows, chronological_split_windows, scale_sequences, LOOKBACK, SEQ_FEATURES,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
FIGURES_DIR = SCRIPT_DIR / "figures"
MLFLOW_DIR = SCRIPT_DIR / "mlflow"
EXPERIMENT_NAME = "smart_city_traffic_capstone"


def setup_mlflow():
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR.as_posix()}/tracking.db")

    artifact_dir = MLFLOW_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s, experiment: %s", mlflow.get_tracking_uri(), EXPERIMENT_NAME)


def log_task1_models(ml_df):
    train_df, test_df = chronological_split(ml_df)
    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = build_feature_targets(train_df, test_df)

    clf_specs = [
        ("logistic_regression_high_risk.joblib", "Logistic Regression (high_risk)"),
        ("random_forest_high_risk.joblib", "Random Forest (high_risk)"),
    ]
    for filename, run_name in clf_specs:
        model = joblib.load(MODELS_DIR / filename)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy_score(y_test_clf, y_pred))
            mlflow.log_metric("precision", precision_score(y_test_clf, y_pred))
            mlflow.log_metric("recall", recall_score(y_test_clf, y_pred))
            mlflow.log_metric("f1", f1_score(y_test_clf, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test_clf, y_proba))
            mlflow.log_artifact(str(MODELS_DIR / filename))
        logger.info("Logged MLflow run: %s", run_name)

    reg_specs = [
        ("linear_regression_traffic_volume.joblib", "Linear Regression (traffic_volume)"),
        ("random_forest_traffic_volume.joblib", "Random Forest (traffic_volume)"),
    ]
    for filename, run_name in reg_specs:
        model = joblib.load(MODELS_DIR / filename)
        y_pred = model.predict(X_test)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            mlflow.log_metric("mae", mean_absolute_error(y_test_reg, y_pred))
            mlflow.log_metric("r2", r2_score(y_test_reg, y_pred))
            mlflow.log_artifact(str(MODELS_DIR / filename))
        logger.info("Logged MLflow run: %s", run_name)


def log_task2_models(ml_df):
    kmeans_model = joblib.load(MODELS_DIR / "kmeans_traffic_clusters.joblib")
    kmeans_scaler = joblib.load(MODELS_DIR / "kmeans_scaler.joblib")

    df_with_severity = add_weather_severity(ml_df)
    X_cluster = kmeans_scaler.transform(df_with_severity[CLUSTER_FEATURES])
    cluster_labels = kmeans_model.predict(X_cluster)
    sil_score = silhouette_score(X_cluster, cluster_labels, sample_size=5000, random_state=42)

    with mlflow.start_run(run_name="K-means Clustering (k=6)"):
        mlflow.log_params(kmeans_model.get_params())
        mlflow.log_metric("inertia", kmeans_model.inertia_)
        mlflow.log_metric("silhouette", sil_score)
        mlflow.log_artifact(str(MODELS_DIR / "kmeans_traffic_clusters.joblib"))
        mlflow.log_artifact(str(MODELS_DIR / "kmeans_scaler.joblib"))
        mlflow.log_artifact(str(SCRIPT_DIR / "task2_cluster_profile.csv"))
    logger.info("Logged MLflow run: K-means Clustering (k=6)")

    arm_df = discretize_for_arm(df_with_severity)
    congestion_rules = mine_congestion_rules(arm_df)

    with mlflow.start_run(run_name="Association Rule Mining (Apriori)"):
        mlflow.log_param("min_support", MIN_SUPPORT)
        mlflow.log_metric("n_congestion_rules", len(congestion_rules))
        mlflow.log_metric("max_lift", congestion_rules["lift"].max())
        mlflow.log_artifact(str(SCRIPT_DIR / "task2_congestion_rules.csv"))
    logger.info("Logged MLflow run: Association Rule Mining (Apriori)")


def log_task3_lstm(ml_df):
    lstm_model = load_model(MODELS_DIR / "lstm_traffic_volume.keras")
    scaler_volume = joblib.load(MODELS_DIR / "lstm_scaler_volume.joblib")
    scaler_severity = joblib.load(MODELS_DIR / "lstm_scaler_severity.joblib")

    df_with_severity = ml_df.copy()
    df_with_severity["weather_severity"] = df_with_severity["weather_main"].map(WEATHER_SEVERITY_RANK)

    gap_df, X_raw, y_raw, target_dates = build_gap_aware_windows(df_with_severity)
    _, X_test_raw, _, y_test_raw, _ = chronological_split_windows(gap_df, X_raw, y_raw, target_dates)
    X_test, y_test = scale_sequences(X_test_raw, y_test_raw, scaler_volume, scaler_severity)

    y_pred_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    y_pred = scaler_volume.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_volume.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    with mlflow.start_run(run_name="LSTM (traffic_volume, 24h lookback)"):
        mlflow.log_params({
            "lookback_hours": LOOKBACK,
            "lstm_units": 64,
            "dropout": 0.2,
            "dense_units": 32,
            "batch_size": 64,
            "max_epochs": 30,
            "early_stopping_patience": 5,
            "features": ",".join(SEQ_FEATURES),
        })
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact(str(MODELS_DIR / "lstm_traffic_volume.keras"))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_scaler_volume.joblib"))
        mlflow.log_artifact(str(MODELS_DIR / "lstm_scaler_severity.joblib"))
        mlflow.log_artifact(str(FIGURES_DIR / "shap_bar_traffic_volume.png"))
        mlflow.log_artifact(str(FIGURES_DIR / "shap_beeswarm_traffic_volume.png"))
    logger.info("Logged MLflow run: LSTM (traffic_volume, 24h lookback) — MAE=%.2f, R2=%.4f", mae, r2)


def run_task4():
    logger.info("Starting Task 4: Advanced AI Technique — MLflow Experiment Tracking.")
    setup_mlflow()

    ml_df = prepare_data()
    log_task1_models(ml_df)
    log_task2_models(ml_df)
    log_task3_lstm(ml_df)

    runs_df = mlflow.search_runs(order_by=["start_time"])
    summary_cols = [c for c in runs_df.columns if c.startswith("tags.mlflow.runName") or c.startswith("metrics.")]

    logger.info("Task 4 complete. %d runs logged.", len(runs_df))
    print(f"\n{len(runs_df)} runs logged to experiment '{EXPERIMENT_NAME}'.")
    print(runs_df[summary_cols].to_string(index=False))
    print("\nTo browse visually, run this from part3_machine_learning/:")
    print(f'  mlflow ui --backend-store-uri "sqlite:///{MLFLOW_DIR.as_posix()}/tracking.db"')

    return runs_df


if __name__ == "__main__":
    from logging_config import configure_logging
    configure_logging()
    run_task4()