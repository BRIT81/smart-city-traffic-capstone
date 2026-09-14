"""
model_versioning.py

Task 6.1 & 6.2 (Part 3): Model Versioning + MLflow Model Registry.

Documents the real hyperparameter/size trade-off already made for the
traffic volume regressor while building Task 1 (see supervised_models.py's
own docstring), by retraining the two earlier, larger versions here purely
in memory and comparing them against the actual small model that was kept
and shipped:

- v1, unconstrained (n_estimators=300, max_depth=None, min_samples_leaf=1):
  scikit-learn's own defaults for tree growth, retrained here from scratch.
  Reached ~882MB serialized.
- v2, a first, milder cap (n_estimators=200, max_depth=20,
  min_samples_leaf=5): retrained here from scratch. ~92MB serialized.
- v3, the final production cap (n_estimators=150, max_depth=15,
  min_samples_leaf=10): reloaded from models/random_forest_traffic_volume.
  joblib rather than retrained, since it is the exact model already in use
  everywhere else in this project (Task 3's SHAP explainability, Task 4's
  MLflow tracking, and Task 6.3's deployment simulation).

Serialized size for every version is measured in memory with joblib.dump
into an io.BytesIO buffer, never written to disk, specifically so this
script never repeats Task 1's original mistake of committing an
oversized model file to the repository.

All three versions are logged to MLflow (reusing setup_mlflow from
mlflow_tracking.py rather than duplicating the tracking URI/experiment
setup) as plain runs with their hyperparameters and metrics. Only v3's
run also attaches the model artifact and registers it under the name
"traffic_volume_regressor", since it is the only version actually meant
to be deployed; v1 and v2 are tagged artifact_retained=false and kept as
metrics-only historical records, so this history costs a few KB in
MLflow's SQLite backend rather than another gigabyte of duplicate model
files. MLflow's own registry version counter (starting at 1 for this
model name's first and only registration) is a separate thing from the
v1/v2/v3 labels used throughout this project's own narrative, and should
not be confused with them.
"""

import io
import logging
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import prepare_data
from supervised_models import FEATURE_COLS, chronological_split, build_feature_targets
from mlflow_tracking import setup_mlflow

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
RF_REGRESSOR_PATH = MODELS_DIR / "random_forest_traffic_volume.joblib"

REGISTERED_MODEL_NAME = "traffic_volume_regressor"

VERSION_SPECS = {
    "v1": dict(n_estimators=300, max_depth=None, min_samples_leaf=1),
    "v2": dict(n_estimators=200, max_depth=20, min_samples_leaf=5),
    "v3": dict(n_estimators=150, max_depth=15, min_samples_leaf=10),
}


def serialized_size_mb(model):
    """Measure a fitted model's joblib-serialized size in memory, without
    ever writing it to disk."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.tell() / 1e6


def train_version(params, X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    size_mb = serialized_size_mb(model)

    logger.info(
        "Trained %s: MAE=%.2f, R2=%.4f, serialized size=%.1f MB, train time=%.1fs",
        params, mae, r2, size_mb, train_time,
    )
    return model, mae, r2, size_mb, train_time


def build_version_history(X_train, y_train, X_test, y_test):
    versions = {}

    for label in ("v1", "v2"):
        model, mae, r2, size_mb, train_time = train_version(
            VERSION_SPECS[label], X_train, y_train, X_test, y_test
        )
        versions[label] = dict(
            params=VERSION_SPECS[label], model=model, mae=mae, r2=r2,
            size_mb=size_mb, train_time=train_time,
        )

    v3_model = joblib.load(RF_REGRESSOR_PATH)
    logger.info("Reloaded production model (v3) from %s.", RF_REGRESSOR_PATH)
    y_pred_v3 = v3_model.predict(X_test)
    versions["v3"] = dict(
        params=VERSION_SPECS["v3"],
        model=v3_model,
        mae=mean_absolute_error(y_test, y_pred_v3),
        r2=r2_score(y_test, y_pred_v3),
        size_mb=serialized_size_mb(v3_model),
        train_time=None,
    )
    logger.info(
        "v3 (production, reloaded): MAE=%.2f, R2=%.4f, serialized size=%.1f MB",
        versions["v3"]["mae"], versions["v3"]["r2"], versions["v3"]["size_mb"],
    )
    return versions


def log_version_history(versions):
    setup_mlflow()

    for label in ("v1", "v2"):
        v = versions[label]
        with mlflow.start_run(run_name=f"Traffic Volume RF {label} ({'unconstrained' if label == 'v1' else 'milder cap'})"):
            mlflow.log_params(v["params"])
            mlflow.log_metric("mae", v["mae"])
            mlflow.log_metric("r2", v["r2"])
            mlflow.log_metric("serialized_size_mb", v["size_mb"])
            mlflow.set_tag("artifact_retained", "false")
            mlflow.set_tag("reason", "artifact intentionally not logged: file size exceeds practical repo limits")
        logger.info("Logged MLflow run for %s (no model artifact attached).", label)

    registered_version = None
    with mlflow.start_run(run_name="Traffic Volume RF v3 (production)"):
        mlflow.log_params(versions["v3"]["params"])
        mlflow.log_metric("mae", versions["v3"]["mae"])
        mlflow.log_metric("r2", versions["v3"]["r2"])
        mlflow.log_metric("serialized_size_mb", versions["v3"]["size_mb"])
        mlflow.set_tag("artifact_retained", "true")
        model_info = mlflow.sklearn.log_model(
            versions["v3"]["model"], artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        registered_version = model_info.registered_model_version
    logger.info("Logged and registered v3 as '%s' version %s.", REGISTERED_MODEL_NAME, registered_version)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", registered_version)
    logger.info("Set alias 'production' -> version %s.", registered_version)

    return registered_version


def print_version_table(versions):
    print("\nHyperparameter and size progression:")
    header = f"{'':4}{'n_estimators':>14}{'max_depth':>12}{'min_samples_leaf':>18}{'size':>12}{'MAE':>12}{'R2':>9}"
    print(header)
    for label in ("v1", "v2", "v3"):
        v = versions[label]
        p = v["params"]
        print(
            f"{label:4}{p['n_estimators']:>14}{str(p['max_depth']):>12}{p['min_samples_leaf']:>18}"
            f"{v['size_mb']:>10.1f}MB{v['mae']:>12.2f}{v['r2']:>9.4f}"
        )


def run_task6_1_2():
    logger.info("Starting Task 6.1 & 6.2: Model Versioning + MLflow Model Registry.")
    ml_df = prepare_data()
    train_df, test_df = chronological_split(ml_df)
    X_train, X_test, _, _, y_train_reg, y_test_reg = build_feature_targets(train_df, test_df)

    versions = build_version_history(X_train, y_train_reg, X_test, y_test_reg)
    registered_version = log_version_history(versions)

    logger.info("Task 6.1 & 6.2 complete. Registered production version: %s", registered_version)
    print_version_table(versions)
    print(f"\nRegistered '{REGISTERED_MODEL_NAME}' version {registered_version} in the MLflow Model Registry, "
          f"alias 'production' -> version {registered_version}.")

    return versions, registered_version


if __name__ == "__main__":
    # Re-assert this script's own directory at the front of sys.path: importing
    # data_prep above inserts part2_python ahead of it (so pipeline.py and
    # feature_engineering.py can be found), which would otherwise cause this
    # bare import to resolve to part2_python's same-named logging_config.py
    # instead of this file's own, silently redirecting the log file there.
    sys.path.insert(0, str(SCRIPT_DIR))
    from logging_config import configure_logging
    configure_logging()
    run_task6_1_2()
