"""
monitoring/drift_monitor.py

Task 6.4 & 6.5 (Part 3): Monitoring Simulation + Alerting Mechanism.

Reads every request the deployment API (deployment/app.py) has actually
served, logged to deployment/prediction_log.csv, and checks it against
the distribution of the data the production model was trained on. Two
kinds of check, run independently:

1. Feature (input) drift: for each raw request field, compares the live
   batch's average or category mix against the training set's. This is
   always computable, since it only needs the request inputs, never the
   true outcome.
2. Prediction error drift: for any live request whose timestamp happens
   to match a real, already-observed row in the dataset, looks up the
   actual traffic_volume and compares it against what the model
   predicted. This is the direct measure of whether the model is still
   accurate, but it can only be computed when ground truth exists — a
   genuinely new, future timestamp has none yet. This mirrors a real
   deployment, where an accuracy check trails behind live requests until
   the true outcome is later confirmed (e.g. a traffic sensor's actual
   hourly count becomes available after the fact).

Thresholds (deliberately simple, explainable rules rather than a formal
statistical test, per the project's console/log reporting choice):

- Continuous fields (temp, rain_1h, snow_1h, clouds_all): ALERT if the
  live batch's mean sits more than 1.0 training-set standard deviations
  away from the training mean. A live batch here is expected to be small
  (a handful of test requests, not a rolling window of thousands), so a
  looser threshold than a typical production system would use is
  appropriate and is documented here rather than hidden.
- Categorical fields (weather_main, is_holiday): ALERT if any category's
  share of the live batch is more than 3x its share of the training set,
  or if a category appears live that the training set never saw at all.
- Prediction error: ALERT if the batch's MAE, computed only over live
  requests with a matched ground-truth row, exceeds 2x the model's own
  freshly recomputed held-out test MAE. Reported as N/A, not PASS, when
  no live request in the batch has a matched ground-truth row yet.

The production model is loaded from the MLflow Model Registry's
"production" alias, the same way deployment/app.py loads it, so this
script always monitors whatever is actually serving traffic.

Known limitations, deliberate given this is a simulation rather than a
production-grade monitoring system:

- The continuous-feature z-score is oversensitive on zero-inflated fields
  (rain_1h, snow_1h are 0 for the overwhelming majority of rows). Their
  training standard deviation is tiny, so even one nonzero live value can
  produce a very large z-score. A production system would typically use a
  bucketed/rate-based check for these fields instead (e.g. "% of requests
  with nonzero snowfall") rather than a raw mean z-score.
- The categorical ratio check only flags a category being over-represented
  live relative to training; it does not flag a category that has become
  under-represented (a ratio below 1.0 always passes). Catching that would
  need a two-sided test (e.g. population stability index) rather than a
  single ratio threshold.
- With a live batch this small (a handful of test requests rather than a
  rolling production window), every check here is illustrative of the
  mechanism, not a statistically rigorous drift test; the thresholds are
  documented above and would need retuning against a much larger batch
  size before being trusted operationally.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PART3_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PART3_DIR))

from data_prep import prepare_data
from supervised_models import FEATURE_COLS, chronological_split, build_feature_targets
from mlflow_tracking import setup_mlflow

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

DEPLOYMENT_DIR = PART3_DIR / "deployment"
PREDICTION_LOG_PATH = DEPLOYMENT_DIR / "prediction_log.csv"

REGISTERED_MODEL_NAME = "traffic_volume_regressor"
MODEL_ALIAS = "production"

CONTINUOUS_FIELDS = ["temp", "rain_1h", "snow_1h", "clouds_all"]
CONTINUOUS_Z_THRESHOLD = 1.0
CATEGORY_RATIO_THRESHOLD = 3.0
ERROR_RATIO_THRESHOLD = 2.0


def build_reference_distribution(train_df):
    """Reference statistics computed from the exact training period the
    production model was fit on (not the full dataset), so "drift" always
    means "different from what the model actually learned from"."""
    continuous_stats = {}
    for field in CONTINUOUS_FIELDS:
        continuous_stats[field] = {
            "mean": train_df[field].mean(),
            "std": train_df[field].std(),
        }

    weather_share = train_df["weather_main"].value_counts(normalize=True).to_dict()
    holiday_rate = train_df["is_holiday"].mean()

    logger.info(
        "Reference distribution built from %d training rows (%s to %s).",
        len(train_df), train_df["date_time"].min(), train_df["date_time"].max(),
    )
    return {
        "continuous": continuous_stats,
        "weather_share": weather_share,
        "holiday_rate": holiday_rate,
    }


def load_production_model():
    setup_mlflow()
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")
    logger.info(
        "Loaded '%s' alias '%s' -> registry version %s for monitoring.",
        REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version,
    )
    return model, model_version.version


def compute_baseline_mae(model, test_df, X_test, y_test_reg):
    """Recomputes the production model's held-out test MAE fresh, rather
    than hardcoding the historical value, so the alerting threshold always
    reflects whichever version is actually registered as production."""
    y_pred = model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test_reg, y_pred)
    logger.info("Freshly computed baseline test MAE for production model: %.2f", baseline_mae)
    return baseline_mae


def load_live_requests(log_path=PREDICTION_LOG_PATH):
    if not log_path.exists():
        logger.warning("No prediction log found at %s. Run the deployment API and make some requests first.", log_path)
        return pd.DataFrame()

    live_df = pd.read_csv(log_path, parse_dates=["request_timestamp", "input_date_time"])
    logger.info("Loaded %d live requests from %s.", len(live_df), log_path)
    return live_df


def check_continuous_drift(live_df, reference):
    findings = []
    for field in CONTINUOUS_FIELDS:
        ref_mean = reference["continuous"][field]["mean"]
        ref_std = reference["continuous"][field]["std"]
        live_mean = live_df[field].mean()
        z = (live_mean - ref_mean) / ref_std if ref_std > 0 else 0.0
        status = "ALERT" if abs(z) > CONTINUOUS_Z_THRESHOLD else "PASS"
        findings.append({
            "check": f"feature mean: {field}",
            "live_value": round(live_mean, 3),
            "reference_value": round(ref_mean, 3),
            "detail": f"z={z:+.2f} (threshold |z|>{CONTINUOUS_Z_THRESHOLD})",
            "status": status,
        })
    return findings


def check_categorical_drift(live_df, reference):
    findings = []

    live_weather_share = live_df["weather_main"].value_counts(normalize=True).to_dict()
    for category, live_share in live_weather_share.items():
        ref_share = reference["weather_share"].get(category, 0.0)
        if ref_share == 0.0:
            status = "ALERT"
            detail = "category never seen in training data"
        else:
            ratio = live_share / ref_share
            status = "ALERT" if ratio > CATEGORY_RATIO_THRESHOLD else "PASS"
            detail = f"ratio={ratio:.2f}x training share (threshold >{CATEGORY_RATIO_THRESHOLD:.0f}x)"
        findings.append({
            "check": f"weather_main share: {category}",
            "live_value": round(live_share, 3),
            "reference_value": round(ref_share, 3),
            "detail": detail,
            "status": status,
        })

    live_holiday_rate = live_df["is_holiday"].astype(bool).mean()
    ref_holiday_rate = reference["holiday_rate"]
    if ref_holiday_rate == 0.0:
        status = "ALERT" if live_holiday_rate > 0 else "PASS"
        detail = "holiday requests never seen in training data" if live_holiday_rate > 0 else "no holidays either side"
    else:
        ratio = live_holiday_rate / ref_holiday_rate
        status = "ALERT" if ratio > CATEGORY_RATIO_THRESHOLD else "PASS"
        detail = f"ratio={ratio:.2f}x training share (threshold >{CATEGORY_RATIO_THRESHOLD:.0f}x)"
    findings.append({
        "check": "is_holiday rate",
        "live_value": round(live_holiday_rate, 3),
        "reference_value": round(ref_holiday_rate, 3),
        "detail": detail,
        "status": status,
    })

    return findings


def check_prediction_error(live_df, ml_df, baseline_mae):
    ground_truth = ml_df.set_index("date_time")["traffic_volume"]
    matched = live_df[live_df["input_date_time"].isin(ground_truth.index)].copy()

    if matched.empty:
        logger.info("No live requests matched a real, already-observed timestamp; prediction error check is N/A.")
        return {
            "check": "prediction error (matched ground truth)",
            "live_value": None,
            "reference_value": round(baseline_mae, 2),
            "detail": "N/A: no live request timestamp has a known outcome yet",
            "status": "N/A",
        }

    matched["actual_traffic_volume"] = matched["input_date_time"].map(ground_truth)
    batch_mae = mean_absolute_error(matched["actual_traffic_volume"], matched["predicted_traffic_volume"])
    ratio = batch_mae / baseline_mae if baseline_mae > 0 else float("inf")
    status = "ALERT" if ratio > ERROR_RATIO_THRESHOLD else "PASS"

    logger.info(
        "Prediction error check: %d matched requests, batch MAE=%.2f vs baseline MAE=%.2f (ratio=%.2fx).",
        len(matched), batch_mae, baseline_mae, ratio,
    )
    return {
        "check": "prediction error (matched ground truth)",
        "live_value": round(batch_mae, 2),
        "reference_value": round(baseline_mae, 2),
        "detail": f"{len(matched)} of {len(live_df)} live requests matched; ratio={ratio:.2f}x (threshold >{ERROR_RATIO_THRESHOLD:.0f}x)",
        "status": status,
    }


def print_drift_report(feature_findings, error_finding, n_live_requests):
    print(f"\n{'=' * 70}")
    print(f"DRIFT MONITORING REPORT — based on {n_live_requests} live request(s)")
    print(f"{'=' * 70}")

    print(f"\n{'Check':40}{'Live':>10}{'Reference':>12}{'Status':>9}")
    print("-" * 71)
    for finding in feature_findings:
        print(f"{finding['check']:40}{str(finding['live_value']):>10}{str(finding['reference_value']):>12}{finding['status']:>9}")
    print(f"{error_finding['check']:40}{str(error_finding['live_value']):>10}{str(error_finding['reference_value']):>12}{error_finding['status']:>9}")

    print("\nDetails:")
    for finding in feature_findings + [error_finding]:
        print(f"  [{finding['status']:>5}] {finding['check']}: {finding['detail']}")

    all_findings = feature_findings + [error_finding]
    n_alerts = sum(1 for f in all_findings if f["status"] == "ALERT")
    overall = "ALERT" if n_alerts > 0 else "PASS"
    print(f"\nOverall status: {overall} ({n_alerts} of {len(all_findings)} checks alerted)")
    print(f"{'=' * 70}\n")

    for finding in all_findings:
        if finding["status"] == "ALERT":
            logger.warning("DRIFT ALERT — %s: %s", finding["check"], finding["detail"])
    logger.info("Drift monitoring report complete. Overall status: %s (%d/%d checks alerted).", overall, n_alerts, len(all_findings))

    return overall


def run_task6_4_5():
    logger.info("Starting Task 6.4 & 6.5: Monitoring Simulation + Alerting Mechanism.")

    ml_df = prepare_data()
    train_df, test_df = chronological_split(ml_df)
    X_train, X_test, _, _, y_train_reg, y_test_reg = build_feature_targets(train_df, test_df)

    reference = build_reference_distribution(train_df)
    model, model_version = load_production_model()
    baseline_mae = compute_baseline_mae(model, test_df, X_test, y_test_reg)

    live_df = load_live_requests()
    if live_df.empty:
        print("\nNo live prediction log found. Run the deployment API (deployment/app.py) and "
              "make a few /predict requests first, then re-run this script.")
        return None

    feature_findings = check_continuous_drift(live_df, reference) + check_categorical_drift(live_df, reference)
    error_finding = check_prediction_error(live_df, ml_df, baseline_mae)

    overall = print_drift_report(feature_findings, error_finding, len(live_df))

    logger.info("Task 6.4 & 6.5 complete. Monitored production model version %s.", model_version)
    return feature_findings, error_finding, overall


if __name__ == "__main__":
    # Re-assert this script's own directory's parent (part3_machine_learning)
    # at the front of sys.path before importing logging_config, for the same
    # reason documented in every other Part 3 script: importing data_prep
    # inserts part2_python ahead of it, which would otherwise cause this bare
    # import to resolve to part2_python's same-named logging_config.py.
    sys.path.insert(0, str(PART3_DIR))
    from logging_config import configure_logging
    configure_logging()
    run_task6_4_5()
